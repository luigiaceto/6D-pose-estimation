import torch
import numpy as np
import cv2
from PIL import Image
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from ultralytics import YOLO

from models.FusionPoseNet import FusionPoseNet
from models.ResNetPose import quaternion_to_rotation_matrix

# =============================================================================
# FUNZIONI DI DISEGNO (Uguali a baseline_visualize.py)
# =============================================================================

def draw_3d_bbox_colored(img, R, t, K, obj_id, models_info, color=(255, 255, 0)):
    """Disegna il bounding box 3D proiettato sull'immagine 2D."""
    # Recupera info modello 3D
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners in mm
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
    
    # Proiezione: t è in metri, convertiamo in mm per coerenza con corners_3d (o viceversa)
    # Qui convertiamo t in mm: t * 1000
    corners_cam = (R @ corners_3d.T).T + t * 1000.0
    
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    corners_2d = []
    for p in corners_cam:
        # p[2] è Z in mm. Deve essere > 0
        if p[2] > 0:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            corners_2d.append((u, v))
        else:
            return img # Bbox dietro la camera
    
    # Disegna linee
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # verticali
    ]
    for e in edges:
        if e[0] < len(corners_2d) and e[1] < len(corners_2d):
            cv2.line(img, corners_2d[e[0]], corners_2d[e[1]], color, 2)
    
    return img

def draw_axis_colored(img, R, t, K, scale=0.05, colors=None):
    """Disegna gli assi XYZ."""
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB default
    
    # Scale in metri (0.05 = 5cm)
    points_3d = np.array([
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ], dtype=np.float32)
    
    # t è già in metri
    points_cam = (R @ points_3d.T).T + t
    
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
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
        cv2.line(img, origin, points_2d[1], colors[0], 3) # X
        cv2.line(img, origin, points_2d[2], colors[1], 3) # Y
        cv2.line(img, origin, points_2d[3], colors[2], 3) # Z
    
    return img

# =============================================================================
# HELPER PER DEPTH PROCESSING
# =============================================================================

def process_depth_crop(depth_path, bbox, target_size=(224, 224)):
    """
    Carica, croppa, pad, resize e normalizza la depth map
    replicando la logica di CustomDatasetPoseExtension.
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # 1. Load (16-bit PNG, valori in mm)
    depth_img = Image.open(depth_path) # Mode 'I'
    
    # 2. Crop
    x, y, w, h = bbox
    cropped_depth = depth_img.crop((x, y, x+w, y+h))
    
    # 3. Padding (Square)
    w_crop, h_crop = cropped_depth.size
    max_dim = max(w_crop, h_crop)
    square_depth = Image.new('I', (max_dim, max_dim), 0)
    
    offset_x = (max_dim - w_crop) // 2
    offset_y = (max_dim - h_crop) // 2
    square_depth.paste(cropped_depth, (offset_x, offset_y))
    
    # 4. Resize (Bilinear come nel tuo dataset)
    square_depth = square_depth.resize(target_size, Image.BILINEAR)
    
    # 5. To Tensor & Normalize (mm -> metri)
    depth_tensor = torch.tensor(np.array(square_depth), dtype=torch.float32)
    depth_tensor = depth_tensor / 1000.0 
    
    # 6. Add Channel dim: (H, W) -> (1, H, W)
    depth_tensor = depth_tensor.unsqueeze(0)
    
    return depth_tensor

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def visualize_fusion_predictions(
    dataset_root,
    cam_k,
    image_path,
    yolo_checkpoint=str(Path("checkpoints") / "best.pt"),
    fusion_checkpoint=str(Path("checkpoints") / "best_fusion_model.pt"), # Checkpoint Extension
    device='cuda',
    figsize=(12, 8),
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    
    # 1. Carica Info Modelli
    models_info_path = dataset_root / "models" / "models_info.yml"
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    # 2. Load Models
    print(f"Loading YOLO: {yolo_checkpoint}")
    yolo_model = YOLO(yolo_checkpoint)
    
    print(f"Loading FusionPoseNet: {fusion_checkpoint}")
    # Nota: FusionPoseNet richiede cam_k nell'init
    pose_model = FusionPoseNet(cam_k=cam_k).to(device)
    checkpoint = torch.load(fusion_checkpoint, map_location=device)
    pose_model.load_state_dict(checkpoint['model_state_dict']) # Assumendo salvataggio standard
    pose_model.eval()
    
    # 3. Trasformazioni RGB
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    
    # 4. Carica Immagine RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb_pil = Image.open(image_path).convert("RGB")
    
    # 5. Deduci Path Depth
    # Assumiamo struttura: data/XX/rgb/YYYY.png -> data/XX/depth/YYYY.png
    depth_path = image_path.replace("rgb", "depth")
    
    # 6. Carica Ground Truth per confronto
    # Path: data/XX/rgb/YYYY.png -> data/XX_gt.yml
    import re
    match = re.search(r'data[\\/](\d+)[\\/]rgb[\\/](\d+)\.png', image_path)
    if not match:
        raise ValueError("Cannot parse image path structure.")
    
    folder_id_str = match.group(1)
    img_id_str = match.group(2)
    img_idx = int(img_id_str)
    
    gt_file = dataset_root / f"{folder_id_str}_gt.yml"
    with open(gt_file, 'r') as f:
        gt_data_all = yaml.load(f, Loader=yaml.CLoader)
    
    # 7. YOLO Inference
    results = yolo_model(image_path, verbose=False)
    
    print("\n" + "="*70)
    print(f"VISUALIZATION (RGB-D FUSION): {os.path.basename(image_path)}")
    print("="*70)
    
    output_img = img_bgr.copy()
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # --- A. BBox Extraction & Logic ---
            bbox = boxes.xywh[i].cpu().numpy()
            x_c, y_c, w, h = bbox
            x_min, y_min = int(x_c - w/2), int(y_c - h/2)
            x_max, y_max = int(x_c + w/2), int(y_c + h/2)
            
            # Clamp coordinate
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_bgr.shape[1], x_max), min(img_bgr.shape[0], y_max)
            
            # --- B. Crop & Preprocess RGB ---
            cropped_pil = img_rgb_pil.crop((x_min, y_min, x_max, y_max))
            
            # Letterbox Padding RGB
            w_crop, h_crop = cropped_pil.size
            max_dim = max(w_crop, h_crop)
            square_rgb = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
            offset_x = (max_dim - w_crop) // 2
            offset_y = (max_dim - h_crop) // 2
            square_rgb.paste(cropped_pil, (offset_x, offset_y))
            
            input_rgb = square_rgb.resize((224, 224), Image.BILINEAR)
            tensor_rgb = rgb_transform(input_rgb).unsqueeze(0).to(device)
            
            # --- C. Crop & Preprocess Depth ---
            # Passiamo il bbox (x, y, w, h) in pixel interi
            bbox_crop_args = (x_min, y_min, x_max - x_min, y_max - y_min)
            tensor_depth = process_depth_crop(depth_path, bbox_crop_args).to(device)
            tensor_depth = tensor_depth.unsqueeze(0) # (1, 1, 224, 224)
            
            # --- D. Prepare BBox Center ---
            # Il centro serve al modello per il Pinhole Layer
            # Deve essere il centro del BBOX rilevato da YOLO (in pixel originali)
            center_x = x_min + (x_max - x_min) / 2.0
            center_y = y_min + (y_max - y_min) / 2.0
            tensor_center = torch.tensor([[center_x, center_y]], dtype=torch.float32).to(device)
            
            # --- E. Inference ---
            with torch.no_grad():
                # Forward Pass: RGB + Depth + Center -> Quaternion + Translation
                pred_quat, pred_trans, pred_2d = pose_model(tensor_rgb, tensor_depth, tensor_center)
            
            # Conversioni per visualizzazione
            pred_t_np = pred_trans[0].cpu().numpy() # [x, y, z] in metri
            pred_q_np = pred_quat[0].cpu().numpy()
            pred_R_np = quaternion_to_rotation_matrix(pred_quat)[0].cpu().numpy()
            pred_uv_np = pred_2d[0].cpu().numpy() # serve ???
            
            # --- F. Recupera Ground Truth ---
            # Usiamo la class ID di YOLO per mappare l'obj_id (class_id + 1 per LineMod solitamente)
            class_id = int(boxes.cls[i])
            obj_id = class_id + 1 
            
            # Cerchiamo nel GT se esiste questo oggetto per questa immagine
            if img_idx in gt_data_all:
                gt_list = gt_data_all[img_idx]
                
                # Semplificazione: prendiamo il primo oggetto GT che corrisponde all'ID
                # (In scene complesse servirebbe matching IoU, ma LineMod è simple)
                gt_info = None
                for obj in gt_list: # gt_list è una lista se ci sono più oggetti, o dict se processato
                     # Nel tuo extract_ground_truth sembrava un dizionario, ma il raw yaml è lista
                     # Adattiamo al formato standard
                     if isinstance(obj, dict) and obj.get('obj_id', -1) == obj_id:
                         gt_info = obj
                         break
                     # Fallback se il tuo yaml preprocessato ha struttura diversa
                     elif isinstance(gt_list, dict) and gt_list.get('obj_id') == obj_id:
                         gt_info = gt_list
                         break
                
                if gt_info:
                    gt_R = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                    gt_t = np.array(gt_info['cam_t_m2c']) / 1000.0 # mm -> m

                    # --- G. Visualizzazione e Metriche ---
                    
                    # 1. Disegna GT (Verde)
                    output_img = draw_3d_bbox_colored(output_img, gt_R, gt_t, cam_k, obj_id, models_info, color=(0, 255, 0))
                    output_img = draw_axis_colored(output_img, gt_R, gt_t, cam_k, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                    
                    # 2. Disegna PRED (Ciano/Arancio)
                    output_img = draw_3d_bbox_colored(output_img, pred_R_np, pred_t_np, cam_k, obj_id, models_info, color=(255, 165, 0))
                    output_img = draw_axis_colored(output_img, pred_R_np, pred_t_np, cam_k, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])

                    # disegna dove la rete pensa sia il centro dell'oggetto
                    cv2.circle(output_img, (int(pred_uv_np[0]), int(pred_uv_np[1])), 5, (0, 255, 255), -1)

                    # 3. Calcolo Errori
                    trans_diff_cm = (pred_t_np - gt_t) * 100
                    trans_err = np.linalg.norm(trans_diff_cm)
                    
                    # Errore Rotazione (Geodesic)
                    R_diff = pred_R_np.T @ gt_R
                    trace = np.trace(R_diff)
                    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
                    rot_err = np.degrees(np.arccos(cos_angle))
                    
                    print(f"\n OBJECT {obj_id} (Detected as Class {class_id})")
                    print(f" --------------------------------------------------")
                    print(f" PREDICTION (RGB-D):")
                    print(f"   T: [{pred_t_np[0]:.4f}, {pred_t_np[1]:.4f}, {pred_t_np[2]:.4f}] m")
                    print(f"   Q: [{pred_q_np[0]:.3f}, {pred_q_np[1]:.3f}, {pred_q_np[2]:.3f}, {pred_q_np[3]:.3f}]")
                    print(f" GROUND TRUTH:")
                    print(f"   T: [{gt_t[0]:.4f}, {gt_t[1]:.4f}, {gt_t[2]:.4f}] m")
                    print(f" ERRORS:")
                    print(f"   Translation: {trans_err:.2f} cm")
                    print(f"     X: {trans_diff_cm[0]:.2f} cm")
                    print(f"     Y: {trans_diff_cm[1]:.2f} cm")
                    print(f"     Z: {trans_diff_cm[2]:.2f} cm (Depth Error)")
                    print(f"   Rotation:    {rot_err:.2f} deg")

    # Mostra risultato
    img_final_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_final_rgb)
    plt.axis('off')
    plt.title("RGB-D Fusion Result: Green=GT, Orange=Pred", fontsize=14)
    plt.show()
