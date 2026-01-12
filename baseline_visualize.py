"""  
Funzioni di visualizzazione per 6D pose.
"""

from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from utils.pose_utils import (
    quaternion_to_rotation_matrix, 
    YOLO_TO_LINEMOD_MAP
)
from utils.visualization import draw_3d_bbox_colored, draw_axis_colored 


def visualize_baseline(
    dataset_root,
    cam_k,
    test_dataset=None,
    yolo_checkpoint=str(Path("checkpoints") / "best_yolo_model.pt"),
    model_checkpoint=str(Path("checkpoints") / "best_pose_model.pt"),
    device='cuda',
    figsize=(18, 6),
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    """
    Pipeline completa: YOLO -> Crop+Padding -> ResNet -> Visualizza GT e predizione.
    Mostra 3 immagini casuali dal test set in un unico plot.
    
    Args:
        test_dataset: Dataset di test per selezionare campioni validi
        num_samples: Numero di immagini da visualizzare (default: 3)
    """
    num_samples = 3

    if test_dataset is None:
        raise ValueError("Devi fornire 'test_dataset'.")
    
    # Seleziona 3 sample casuali dal test set
    test_samples = test_dataset.get_samples_id()
    random_indices = np.random.choice(len(test_samples), size=num_samples, replace=False)
    
    selected_samples = [test_samples[idx] for idx in random_indices]
    
    # Load model info and models
    models_info_path = str(dataset_root / "models" / "models_info.yml")
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    object_diameters = {obj_id: info['diameter'] for obj_id, info in models_info.items()}
    
    # Load YOLO and Pose models
    yolo_model = YOLO(yolo_checkpoint)
    
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    pose_model = ResNetPose().to(device)
    pose_model.load_state_dict(checkpoint['model_state_dict'])
    pose_model.eval()

    pinhole = PinholeCamera(cam_k=cam_k)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    
    # Process each selected sample
    processed_images = []
    
    for folder_id, sample_id in selected_samples:
        image_path = str(dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping {image_path} - not found")
            continue
        
        # Load ground truth
        gt_file = str(dataset_root / f"{folder_id:02d}_gt.yml")
        with open(gt_file, 'r') as f:
            gt_data = yaml.load(f, Loader=yaml.CLoader)
        
        # YOLO inference
        results = yolo_model(image_path, verbose=False)
        
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
                
                # Usa il metodo del dataset per crop e padding
                img_pil = Image.open(image_path).convert("RGB")
                bbox_xywh = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                    continue
                
                # Crop e pad usando il metodo del dataset
                preprocessed_img = test_dataset._crop_and_pad_image(img_pil, bbox_xywh, resample=Image.BILINEAR)
                cropped_tensor = transform(preprocessed_img).unsqueeze(0).to(device)

                # Predict quaternion
                with torch.no_grad():
                    pred_quaternion = pose_model(cropped_tensor)
                
                # Calculate translation
                class_id = int(boxes.cls[i])
                obj_id = YOLO_TO_LINEMOD_MAP[class_id]
                diameter = object_diameters[obj_id]
                
                bbox_xyxy = torch.tensor([[x_min, y_min, x_max, y_max]], device=device, dtype=torch.float32)
                center_2d = torch.tensor([[(x_min + x_max) / 2, (y_min + y_max) / 2]], device=device)
                batch_diam = torch.tensor([diameter], device=device)
                
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diam)
                pred_trans_tensor = pinhole.unproject_2d_to_3d(center_2d, depth)
                
                # Convert to Numpy for Drawing
                pred_translation = pred_trans_tensor[0].cpu().numpy()
                pred_rotation = quaternion_to_rotation_matrix(pred_quaternion)[0].cpu().numpy()
                
                # Extract ground truth
                if sample_id in gt_data:
                    gt_info = gt_data[sample_id][0]  
                    gt_rotation = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                    gt_translation = np.array(gt_info['cam_t_m2c']) / 1000.0
                    
                    # Draw GROUND TRUTH (Green)
                    img = draw_3d_bbox_colored(img, gt_rotation, gt_translation, cam_k, obj_id, models_info, color=(0, 255, 0))
                    img = draw_axis_colored(img, gt_rotation, gt_translation, cam_k, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                    
                    # Draw PREDICTION (Cyan)
                    img = draw_3d_bbox_colored(img, pred_rotation, pred_translation, cam_k, obj_id, models_info, color=(255, 165, 0))
                    img = draw_axis_colored(img, pred_rotation, pred_translation, cam_k, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])
        
        # Convert to RGB and store
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_images.append(img_rgb)
    
    # Display all images in a single row
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_rgb in enumerate(processed_images):
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'Sample {idx+1}', fontsize=12)
    
    plt.suptitle('Green = Ground Truth | Cyan = Prediction', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()