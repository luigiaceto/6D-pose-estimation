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

from models.TridentNetPose import TridentNetPose
from utils.pose_utils import quaternion_to_rotation_matrix
from utils.visualization import draw_3d_bbox_colored, draw_axis_colored

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
    test_dataset=None,
    sample_idx=None,
    image_path=None,
    yolo_checkpoint=str(Path("checkpoints") / "best.pt"),
    fusion_checkpoint=str(Path("checkpoints") / "best_fusion_model.pt"), # Checkpoint Extension
    device='cuda',
    figsize=(12, 8),
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    
    # 1. Gestione Selezione Immagine dal Test Set
    if test_dataset is not None:
        test_samples = test_dataset.get_samples_id()
        
        if sample_idx is not None:
            # Usa sample specifico dal test set
            if sample_idx < 0 or sample_idx >= len(test_samples):
                raise ValueError(f"sample_idx {sample_idx} fuori range. Test set ha {len(test_samples)} samples.")
            folder_id, sample_id = test_samples[sample_idx]
        else:
            # Selezione casuale dal test set
            folder_id, sample_id = test_samples[np.random.randint(len(test_samples))]
        
        image_path = str(dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        print(f"📊 Visualizzando sample dal TEST SET: folder {folder_id:02d}, image {sample_id:04d}")
    
    elif image_path is None:
        raise ValueError("Devi fornire 'test_dataset' oppure 'image_path'.")
    
    else:
        # Warning: path manuale
        print("⚠️  ATTENZIONE: image_path fornito manualmente. Impossibile verificare se è nel test set.")
        print("   Raccomandazione: usa 'test_dataset' per garantire selezione dal test set.")
    
    # 2. Carica Info Modelli
    models_info_path = dataset_root / "models" / "models_info.yml"
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    # 3. Load Models
    print(f"Loading YOLO: {yolo_checkpoint}")
    yolo_model = YOLO(yolo_checkpoint)
    
    print(f"Loading TridentNetPose: {fusion_checkpoint}")
    pose_model = TridentNetPose(cam_k=cam_k).to(device)
    checkpoint = torch.load(fusion_checkpoint, map_location=device)
    pose_model.load_state_dict(checkpoint['model_state_dict']) # Assumendo salvataggio standard
    pose_model.eval()
    
    # 4. Trasformazioni RGB
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    
    # 5. Carica Immagine RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb_pil = Image.open(image_path).convert("RGB")
    
    # 6. Deduci Path Depth
    # Assumiamo struttura: data/XX/rgb/YYYY.png -> data/XX/depth/YYYY.png
    depth_path = image_path.replace("rgb", "depth")
    
    # 7. Carica Ground Truth per confronto
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
    
    # 8. YOLO Inference
    results = yolo_model(image_path, verbose=False)
    
    print("\n" + "="*70)
    print(f"VISUALIZATION: {os.path.basename(image_path)}")
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

            img_h_orig, img_w_orig = img_bgr.shape[:2]
            w_norm = w / img_w_orig
            h_norm = h / img_h_orig
            tensor_bbox_dims = torch.tensor([[w_norm, h_norm]], dtype=torch.float32).to(device)
            
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
                pred_quat, pred_trans, pred_2d = pose_model(tensor_rgb, tensor_depth, tensor_center, tensor_bbox_dims)
            
            # Conversioni per visualizzazione
            pred_t_np = pred_trans[0].cpu().numpy() # [x, y, z] in metri
            pred_q_np = pred_quat[0].cpu().numpy()
            pred_R_np = quaternion_to_rotation_matrix(pred_quat)[0].cpu().numpy()
            pred_uv_np = pred_2d[0].cpu().numpy() # serve ???
            
            # --- F. Recupera Ground Truth ---
            class_id = int(boxes.cls[i])
            obj_id = class_id + 1 
            
            # Estrazione ground truth per questa immagine (come baseline)
            if img_idx in gt_data_all:
                gt_info = gt_data_all[img_idx][0]  
                gt_R = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                gt_t = np.array(gt_info['cam_t_m2c']) / 1000.0  
                gt_quat = np.array(gt_info['quaternion'])
                
                # Draw GROUND TRUTH (Verde)
                output_img = draw_3d_bbox_colored(output_img, gt_R, gt_t, cam_k, obj_id, models_info, color=(0, 255, 0))
                output_img = draw_axis_colored(output_img, gt_R, gt_t, cam_k, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                
                # Draw PREDICTION (Arancione)
                output_img = draw_3d_bbox_colored(output_img, pred_R_np, pred_t_np, cam_k, obj_id, models_info, color=(255, 165, 0))
                output_img = draw_axis_colored(output_img, pred_R_np, pred_t_np, cam_k, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])
                
                # Print confronto numerico (come baseline)
                print(f"\n Object {obj_id} (Class {class_id})")
                print(f"\n GROUND TRUTH:")
                print(f"     Translation: [{gt_t[0]:7.4f}, {gt_t[1]:7.4f}, {gt_t[2]:7.4f}] m")
                print(f"     Quaternion:  [{gt_quat[0]:7.4f}, {gt_quat[1]:7.4f}, {gt_quat[2]:7.4f}, {gt_quat[3]:7.4f}]")
                
                print(f"\n PREDICTION:")
                print(f"     Translation: [{pred_t_np[0]:7.4f}, {pred_t_np[1]:7.4f}, {pred_t_np[2]:7.4f}] m")
                print(f"     Quaternion:  [{pred_q_np[0]:7.4f}, {pred_q_np[1]:7.4f}, {pred_q_np[2]:7.4f}, {pred_q_np[3]:7.4f}]")
                
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
                trans_diff = (pred_t_np - gt_t) * 100 
                trans_error = np.linalg.norm(trans_diff)
                R_diff = pred_R_np.T @ gt_R
                rot_error = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)))
                
                print(f"\n ERRORS:")
                print(f"     BBox IoU (2D) by YOLO: {bbox_iou:.2%}")
                print(f"     Translation Error: {trans_error:.2f} cm")
                print(f"       - X error: {trans_diff[0]:6.2f} cm")
                print(f"       - Y error: {trans_diff[1]:6.2f} cm")
                print(f"       - Z error (depth): {trans_diff[2]:6.2f} cm")
                print(f"     Rotation Error: {rot_error:.2f}°")

    # Mostra risultato
    img_final_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_final_rgb)
    plt.axis('off')
    plt.title('Green = Ground Truth | Cyan = Prediction', fontsize=14)
    plt.show()