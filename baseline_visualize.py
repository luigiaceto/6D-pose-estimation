"""  
Funzioni di visualizzazione per 6D pose.
"""

from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
import yaml
import os
import re
from ultralytics import YOLO

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils.pose_utils import quaternion_to_rotation_matrix
from utils.visualization import draw_3d_bbox_colored, draw_axis_colored

def visualize_baseline_predictions(
    dataset_root,
    cam_k,
    image_path,
    yolo_checkpoint=str(Path("checkpoints") / "best_yolo_model.pt"),
    pose_checkpoint=str(Path("checkpoints") / "best_pose_model.pt"),
    device='cuda',
    figsize=(12, 8),
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    """
    Pipeline completa: YOLO -> Crop+Padding -> ResNet -> Visualizza GT e predizione con confronto numerico.
    """
    
    # Load object diameters
    models_info_path = str(dataset_root / "models" / "models_info.yml")
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    object_diameters = {obj_id: info['diameter'] for obj_id, info in models_info.items()}
    
    # Load models
    yolo_model = YOLO(yolo_checkpoint)
    
    checkpoint = torch.load(pose_checkpoint, map_location=device, weights_only=False)
    pose_model = ResNetPose().to(device)
    pose_model.load_state_dict(checkpoint['model_state_dict'])
    pose_model.eval()
    
    # Pinhole camera
    pinhole = PinholeCamera(cam_k=cam_k)
    
    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=img_mean,
            std=img_std
        )
    ])
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    match = re.search(r'data[/\\](\d+)[/\\]rgb[/\\](\d+)\.png', image_path)
    if not match:
        raise ValueError(f"Path immagine non valido: {image_path}")
    
    obj_folder = match.group(1)
    img_name = match.group(2)
    
    
    gt_file = image_path.replace(
        str(Path("data") / f"{obj_folder}" / "rgb" / f"{img_name}.png"),
        f'{obj_folder}_gt.yml'
    )
    
    with open(gt_file, 'r') as f:
        gt_data = yaml.load(f, Loader=yaml.CLoader)
    
    # YOLO detection
    results = yolo_model(image_path, verbose=False)
    
    print("\n" + "="*70)
    print(f"VISUALIZATION: {os.path.basename(image_path)}")
    print("="*70)
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Bounding box extraction
            bbox = boxes.xywh[i].cpu().numpy()
            x_c, y_c, w, h = bbox
            x_min = int(x_c - w/2)
            y_min = int(y_c - h/2)
            x_max = int(x_c + w/2)
            y_max = int(y_c + h/2)
            
            # Crop handling
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)
            
            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
            
            # Converti in PIL RGB
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_pil = Image.fromarray(cropped_rgb)

            # LETTERBOX PADDING
            w_crop, h_crop = cropped_pil.size
            max_dim = max(w_crop, h_crop)

            # Creiamo una nuova immagine quadrata nera
            square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

            # Calcoliamo offset per centrare l'immagine
            offset_x = (max_dim - w_crop) // 2
            offset_y = (max_dim - h_crop) // 2
            
            # Incolliamo l'immagine al centro
            square_img.paste(cropped_pil, (offset_x, offset_y))
            
            # Resize alla dimensione di input della ResNet (224x224) 
            final_input = square_img.resize((224, 224), Image.BILINEAR)
            
            cropped_tensor = transform(final_input).unsqueeze(0).to(device)

            # Predict quaternion
            with torch.no_grad():
                pred_quaternion = pose_model(cropped_tensor)
            
            # Calculate translation
            class_id = int(boxes.cls[i])
            obj_id = class_id + 1
            diameter = object_diameters[obj_id]
            
            bbox_xyxy = torch.tensor([[x_min, y_min, x_max, y_max]], device=device, dtype=torch.float32)
            center_2d_pixels = torch.tensor([[(x_min + x_max) / 2, (y_min + y_max) / 2]], device=device, dtype=torch.float32)
            batch_diameters = torch.tensor([diameter], device=device, dtype=torch.float32)
            
            depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
            pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)[0].cpu().numpy()
            pred_rotation = quaternion_to_rotation_matrix(pred_quaternion)[0].cpu().numpy()
            pred_quat = pred_quaternion[0].cpu().numpy()
            
            # Estrazione ground truth per questa immagine
            img_idx = int(img_name)
            if img_idx in gt_data:
                gt_info = gt_data[img_idx][0]  
                gt_rotation = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                gt_translation = np.array(gt_info['cam_t_m2c']) / 1000.0  
                gt_quat = np.array(gt_info['quaternion'])
                
                # Draw GROUND TRUTH
                img = draw_3d_bbox_colored(img, gt_rotation, gt_translation, cam_k, obj_id, models_info, color=(0, 255, 0))
                img = draw_axis_colored(img, gt_rotation, gt_translation, cam_k, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                
                # Draw PREDICTION 
                img = draw_3d_bbox_colored(img, pred_rotation, pred_translation, cam_k, obj_id, models_info, color=(255, 165, 0))
                img = draw_axis_colored(img, pred_rotation, pred_translation, cam_k, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])
                
                # Print confronto numerico
                print(f"\n Object {obj_id} (Class {class_id})")
                print(f"\n GROUND TRUTH:")
                print(f"     Translation: [{gt_translation[0]:7.4f}, {gt_translation[1]:7.4f}, {gt_translation[2]:7.4f}] m")
                print(f"     Quaternion:  [{gt_quat[0]:7.4f}, {gt_quat[1]:7.4f}, {gt_quat[2]:7.4f}, {gt_quat[3]:7.4f}]")
                
                print(f"\n PREDICTION:")
                print(f"     Translation: [{pred_translation[0]:7.4f}, {pred_translation[1]:7.4f}, {pred_translation[2]:7.4f}] m")
                print(f"     Quaternion:  [{pred_quat[0]:7.4f}, {pred_quat[1]:7.4f}, {pred_quat[2]:7.4f}, {pred_quat[3]:7.4f}]")
                
                # Calcola IoU del bounding box
                gt_bbox = gt_info['obj_bb'] 
                gt_x1, gt_y1, gt_w, gt_h = gt_bbox
                gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
                
                # Calcola intersezione
                x1_inter = max(x_min, gt_x1)
                y1_inter = max(y_min, gt_y1)
                x2_inter = min(x_max, gt_x2)
                y2_inter = min(y_max, gt_y2)
                
                if x2_inter > x1_inter and y2_inter > y1_inter:
                    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                else:
                    intersection = 0
                
                # Calcola union
                pred_area = (x_max - x_min) * (y_max - y_min)
                gt_area = gt_w * gt_h
                union = pred_area + gt_area - intersection
                
                bbox_iou = intersection / union if union > 0 else 0
                
                # Calcola errori pose
                trans_diff = (pred_translation - gt_translation) * 100 
                trans_error = np.linalg.norm(trans_diff)
                R_diff = pred_rotation.T @ gt_rotation
                rot_error = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)))
                
                print(f"\n ERRORS:")
                print(f"     BBox IoU (2D) by YOLO: {bbox_iou:.2%}")
                print(f"     Translation Error: {trans_error:.2f} cm")
                print(f"       - X error: {trans_diff[0]:6.2f} cm")
                print(f"       - Y error: {trans_diff[1]:6.2f} cm")
                print(f"       - Z error (depth): {trans_diff[2]:6.2f} cm")
                print(f"     Rotation Error: {rot_error:.2f}°")
    
    print("\n" + "="*70 + "\n")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Green = Ground Truth | Cyan = Prediction', fontsize=14)
    plt.show()