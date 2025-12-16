"""  
Funzioni di visualizzazione per 6D pose.
"""

import cv2
import numpy as np
import torch
import yaml
import os
from ultralytics import YOLO

from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera
import torchvision.transforms as transforms


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


def visualize_predictions(
    image_path,
    yolo_checkpoint='./checkpoints/best.pt',
    pose_checkpoint='./checkpoints/best_pose_model.pt',
    output_path=None,
    device='cuda'
):
    """
    Pipeline completa: YOLO -> Crop -> Pose -> Visualizza.
    
    Args:
        image_path: path immagine RGB
        yolo_checkpoint: checkpoint YOLO
        pose_checkpoint: checkpoint pose model
        output_path: dove salvare risultato
        device: 'cuda' o 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load object diameters
    dataset_root = "./datasets/linemod/DenseFusion/Linemod_preprocessed"
    models_info_path = os.path.join(dataset_root, 'models', 'models_info.yml')
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    object_diameters = {obj_id: info['diameter'] for obj_id, info in models_info.items()}
    
    # Load models
    yolo_model = YOLO(yolo_checkpoint)
    
    checkpoint = torch.load(pose_checkpoint, map_location=device)
    pose_model = ResNetPose(pretrained=False).to(device)
    pose_model.load_state_dict(checkpoint['model_state_dict'])
    pose_model.eval()
    
    # Pinhole camera
    cam_params = checkpoint['camera_params']
    pinhole = PinholeCamera(
        cam_params['fx'], cam_params['fy'],
        cam_params['cx'], cam_params['cy']
    )
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
    IMG_WIDTH, IMG_HEIGHT = 640, 480
    
    # YOLO detection
    results = yolo_model(image_path, verbose=False)
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Bounding box
            bbox = boxes.xywh[i].cpu().numpy()
            x_c, y_c, w, h = bbox
            x_min = int(x_c - w/2)
            y_min = int(y_c - h/2)
            x_max = int(x_c + w/2)
            y_max = int(y_c + h/2)
            
            # Crop
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)
            
            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
            
            # RGB e transform
            from PIL import Image
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_pil = Image.fromarray(cropped_rgb)
            cropped_tensor = transform(cropped_pil).unsqueeze(0).to(device)
            
            # Predict quaternion
            with torch.no_grad():
                pred_quaternion = pose_model(cropped_tensor)  # (1, 4)
            
            # Calcola translation da bbox + diametro
            # YOLO class_id va da 0-14, obj_id dataset va da 1-15
            class_id = int(boxes.cls[i])
            obj_id = class_id + 1  # mapping YOLO -> dataset
            diameter = object_diameters[obj_id]
            
            bbox_xyxy = torch.tensor(
                [[x_min, y_min, x_max, y_max]], 
                device=device, dtype=torch.float32
            )
            
            center_2d_pixels = torch.tensor(
                [[(x_min + x_max) / 2, (y_min + y_max) / 2]],
                device=device, dtype=torch.float32
            )
            
            batch_diameters = torch.tensor(
                [diameter], device=device, dtype=torch.float32
            )
            
            depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
            translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)[0].cpu().numpy()
            
            # Rotation matrix
            rotation = quaternion_to_rotation_matrix(pred_quaternion)[0].cpu().numpy()
            
            # Draw 3D bounding box
            img = draw_3d_bbox(img, rotation, translation, K, obj_id, models_info)
            
            # Draw 3D axes
            img = draw_axis(img, rotation, translation, K, scale=0.05)
            
            # Info text
            conf = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            cv2.putText(
                img, f"Class {class_id} ({conf:.2f})",
                (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )
    
    # Save o show
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved to {output_path}")
    
    return img


if __name__ == "__main__":
    # Esempio
    visualize_predictions(
        image_path="./datasets/linemod/DenseFusion/Linemod_preprocessed/data/01/rgb/0000.png",
        output_path="./visualization_result.png",
        device='cuda'
    )
