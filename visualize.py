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
from ultralytics import YOLO

from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

def draw_axis(img, R, t, K, scale=0.05):
    """
    Disegna assi 3D (X=rosso, Y=verde, Z=blu) sull'immagine.
    
    Args:
        img: immagine BGR
        R: (3,3) rotation matrix
        t: (3,) translation vector
        K: (3,3) camera intrinsics
        scale: lunghezza assi in metri
    """
    # Punti 3D degli assi
    points_3d = np.array([
        [0, 0, 0],
        [scale, 0, 0],  # X rosso
        [0, scale, 0],  # Y verde
        [0, 0, scale]   # Z blu
    ], dtype=np.float32)
    
    # Trasforma in camera coords
    points_cam = (R @ points_3d.T).T + t
    
    # Proietta a 2D
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    points_2d = []
    for p in points_cam:
        if p[2] > 0:  # Solo se davanti alla camera
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            points_2d.append((u, v))
        else:
            points_2d.append(None)
    
    if all(p is not None for p in points_2d):
        origin = points_2d[0]
        cv2.line(img, origin, points_2d[1], (0, 0, 255), 3)    # X rosso
        cv2.line(img, origin, points_2d[2], (0, 255, 0), 3)    # Y verde
        cv2.line(img, origin, points_2d[3], (255, 0, 0), 3)    # Z blu
    
    return img


def draw_3d_bbox(img, R, t, K, obj_id, models_info):
    """
    Disegna bounding box 3D dell'oggetto sull'immagine.
    
    Args:
        img: immagine BGR
        R: (3,3) rotation matrix
        t: (3,) translation vector
        K: (3,3) camera intrinsics
        obj_id: ID oggetto
        models_info: dizionario con info modelli
    """
    # Carica corner points 3D del modello
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners del bounding box 3D (in mm)
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32)
    
    # Trasforma in camera coords: (R @ p + t) per ogni punto
    corners_cam = (R @ corners_3d.T).T + t * 1000  # t è in metri, convertiamo in mm
    
    # Proietta a 2D
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    corners_2d = []
    for p in corners_cam:
        if p[2] > 0:  # Solo se davanti alla camera
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            corners_2d.append((u, v))
        else:
            return img  # Se qualche corner è dietro la camera, non disegnare
    
    # Disegna il bounding box 3D
    # Bottom face (0,1,2,3)
    cv2.line(img, corners_2d[0], corners_2d[1], (255, 255, 0), 2)
    cv2.line(img, corners_2d[1], corners_2d[2], (255, 255, 0), 2)
    cv2.line(img, corners_2d[2], corners_2d[3], (255, 255, 0), 2)
    cv2.line(img, corners_2d[3], corners_2d[0], (255, 255, 0), 2)
    
    # Top face (4,5,6,7)
    cv2.line(img, corners_2d[4], corners_2d[5], (255, 255, 0), 2)
    cv2.line(img, corners_2d[5], corners_2d[6], (255, 255, 0), 2)
    cv2.line(img, corners_2d[6], corners_2d[7], (255, 255, 0), 2)
    cv2.line(img, corners_2d[7], corners_2d[4], (255, 255, 0), 2)
    
    # Vertical edges
    cv2.line(img, corners_2d[0], corners_2d[4], (255, 255, 0), 2)
    cv2.line(img, corners_2d[1], corners_2d[5], (255, 255, 0), 2)
    cv2.line(img, corners_2d[2], corners_2d[6], (255, 255, 0), 2)
    cv2.line(img, corners_2d[3], corners_2d[7], (255, 255, 0), 2)
    
    return img


def draw_3d_bbox_colored(img, R, t, K, obj_id, models_info, color=(255, 255, 0)):
    """Versione con colore personalizzabile per GT vs Pred."""
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32)
    
    corners_cam = (R @ corners_3d.T).T + t * 1000
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    corners_2d = []
    for p in corners_cam:
        if p[2] > 0:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            corners_2d.append((u, v))
        else:
            return img
    
    # Draw con colore specificato
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # bottom
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # vertical
    ]
    for e in edges:
        cv2.line(img, corners_2d[e[0]], corners_2d[e[1]], color, 2)
    
    return img


def draw_axis_colored(img, R, t, K, scale=0.05, colors=None):
    """Versione con colori personalizzabili per GT vs Pred."""
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # default: RGB
    
    points_3d = np.array([
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ], dtype=np.float32)
    
    points_cam = (R @ points_3d.T).T + t
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    points_2d = []
    for p in points_cam:
        if p[2] > 0:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            points_2d.append((u, v))
        else:
            points_2d.append(None)
    
    if all(p is not None for p in points_2d):
        origin = points_2d[0]
        cv2.line(img, origin, points_2d[1], colors[0], 3)  # X
        cv2.line(img, origin, points_2d[2], colors[1], 3)  # Y
        cv2.line(img, origin, points_2d[3], colors[2], 3)  # Z
    
    return img


def visualize_predictions(
    dataset_root,
    cam_k,
    image_path,
    yolo_checkpoint=str(Path("checkpoints") / "best.pt"),
    pose_checkpoint=str(Path("checkpoints") / "best_pose_model_with_stats.pt"),
    device='cuda',
    figsize=(12, 8),
    show_gt=True
):
    """
    Pipeline completa: YOLO -> Crop -> Pose -> Visualizza GT e predizione con confronto numerico.
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
    K = pinhole.get_intrinsics_matrix()
    
    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint['image_mean'],
            std=checkpoint['image_std']
        )
    ])
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    # Load ground truth per confronto
    import re
    match = re.search(r'data/(\d+)/rgb/(\d+)\.png', image_path)
    if not match:
        raise ValueError("Path immagine non valido")
    
    obj_folder = match.group(1)
    img_name = match.group(2)
    
    gt_file = image_path.replace(f'data/{obj_folder}/rgb/{img_name}.png', f'{obj_folder}_gt.yml')
    
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
            
            # RGB conversion for model input
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_pil = Image.fromarray(cropped_rgb)
            cropped_tensor = transform(cropped_pil).unsqueeze(0).to(device)
            
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
            
            # Get ground truth per questa immagine
            img_idx = int(img_name)
            if img_idx in gt_data:
                gt_info = gt_data[img_idx][0]  # Primo oggetto
                gt_rotation = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                gt_translation = np.array(gt_info['cam_t_m2c']) / 1000.0  # mm -> m
                gt_quat = np.array(gt_info['quaternion'])
                
                # Draw GROUND TRUTH (verde)
                img = draw_3d_bbox_colored(img, gt_rotation, gt_translation, K, obj_id, models_info, color=(0, 255, 0))
                img = draw_axis_colored(img, gt_rotation, gt_translation, K, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                
                # Draw PREDICTION (ciano/blu)
                img = draw_3d_bbox_colored(img, pred_rotation, pred_translation, K, obj_id, models_info, color=(255, 165, 0))
                img = draw_axis_colored(img, pred_rotation, pred_translation, K, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])
                
                # Print confronto numerico
                print(f"\n Object {obj_id} (Class {class_id})")
                print(f"\n GROUND TRUTH:")
                print(f"     Translation: [{gt_translation[0]:7.4f}, {gt_translation[1]:7.4f}, {gt_translation[2]:7.4f}] m")
                print(f"     Quaternion:  [{gt_quat[0]:7.4f}, {gt_quat[1]:7.4f}, {gt_quat[2]:7.4f}, {gt_quat[3]:7.4f}]")
                
                print(f"\n PREDICTION:")
                print(f"     Translation: [{pred_translation[0]:7.4f}, {pred_translation[1]:7.4f}, {pred_translation[2]:7.4f}] m")
                print(f"     Quaternion:  [{pred_quat[0]:7.4f}, {pred_quat[1]:7.4f}, {pred_quat[2]:7.4f}, {pred_quat[3]:7.4f}]")
                
                # Calcola IoU del bounding box
                gt_bbox = gt_info['obj_bb']  # [x_min, y_min, width, height]
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
                trans_diff = (pred_translation - gt_translation) * 100  # m -> cm
                trans_error = np.linalg.norm(trans_diff)
                R_diff = pred_rotation.T @ gt_rotation
                rot_error = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)))
                
                print(f"\n ERRORS:")
                print(f"     BBox IoU (2D):        {bbox_iou:.2%}  ← YOLO detection accuracy")
                print(f"     Translation Error:    {trans_error:.2f} cm")
                print(f"       - X error: {trans_diff[0]:6.2f} cm")
                print(f"       - Y error: {trans_diff[1]:6.2f} cm")
                print(f"       - Z error (depth): {trans_diff[2]:6.2f} cm  ← Main issue!")
                print(f"     Rotation Error:       {rot_error:.2f}°")
    
    print("\n" + "="*70 + "\n")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Green = Ground Truth | Cyan = Prediction', fontsize=14)
    plt.show()
