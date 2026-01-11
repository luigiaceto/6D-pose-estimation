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
from utils.pose_utils import quaternion_to_rotation_matrix, YOLO_TO_LINEMOD_MAP
from utils.visualization import draw_3d_bbox_colored, draw_axis_colored

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def visualize_extension(
    dataset_root,
    cam_k,
    test_dataset=None,
    yolo_checkpoint=str(Path("checkpoints") / "best_yolo_model.pt"),
    model_checkpoint=str(Path("checkpoints") / "best_extension_model.pt"),
    device='cuda',
    figsize=(18, 6),
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    """
    Pipeline completa: YOLO -> Crop+Padding -> TridentNetPose -> Visualizza GT e predizione.
    Mostra 3 immagini casuali dal test set in un unico plot.
    
    Args:
        test_dataset: Dataset di test per selezionare campioni validi
    """
    num_samples = 3
    
    if test_dataset is None:
        raise ValueError("Devi fornire 'test_dataset'.")
    
    # Seleziona 3 sample casuali dal test set
    test_samples = test_dataset.get_samples_id()
    random_indices = np.random.choice(len(test_samples), size=num_samples, replace=False)
    
    selected_samples = [test_samples[idx] for idx in random_indices]
    
    # Load model info
    models_info_path = dataset_root / "models" / "models_info.yml"
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    # Load YOLO and Pose models
    yolo_model = YOLO(yolo_checkpoint)
    
    pose_model = TridentNetPose(cam_k=cam_k).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    pose_model.load_state_dict(checkpoint['model_state_dict'])
    pose_model.eval()
    
    # RGB transformation
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    
    # Process each selected sample
    processed_images = []
    
    for folder_id, sample_id in selected_samples:
        image_path = str(dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        depth_path = str(dataset_root / "data" / f"{folder_id:02d}" / "depth" / f"{sample_id:04d}.png")
        
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Skipping {image_path} - not found")
            continue
        
        img_rgb_pil = Image.open(image_path).convert("RGB")
        
        # Load ground truth
        gt_file = str(dataset_root / f"{folder_id:02d}_gt.yml")
        with open(gt_file, 'r') as f:
            gt_data = yaml.load(f, Loader=yaml.CLoader)
        
        # YOLO inference
        results = yolo_model(image_path, verbose=False)
        
        output_img = img_bgr.copy()
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # BBox Extraction
                bbox = boxes.xywh[i].cpu().numpy()
                x_c, y_c, w, h = bbox
                x_min, y_min = int(x_c - w/2), int(y_c - h/2)
                x_max, y_max = int(x_c + w/2), int(y_c + h/2)
                
                # Clamp coordinates
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(output_img.shape[1], x_max), min(output_img.shape[0], y_max)
                
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Prepara bbox in formato (x, y, w, h)
                bbox_xywh = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                    continue
                
                img_h_orig, img_w_orig = output_img.shape[:2]
                w_norm = bbox_xywh[2] / img_w_orig
                h_norm = bbox_xywh[3] / img_h_orig
                tensor_bbox_dims = torch.tensor([[w_norm, h_norm]], dtype=torch.float32).to(device)
                
                # Crop & Preprocess RGB usando il metodo del dataset
                preprocessed_rgb = test_dataset._crop_and_pad_image(img_rgb_pil, bbox_xywh, resample=Image.BILINEAR)
                tensor_rgb = rgb_transform(preprocessed_rgb).unsqueeze(0).to(device)
                
                # Crop & Preprocess Depth usando il metodo del dataset
                depth_pil = Image.open(depth_path)  # Mode 'I' (16-bit)
                preprocessed_depth = test_dataset._crop_and_pad_image(depth_pil, bbox_xywh, resample=Image.NEAREST)
                
                # Converti depth da PIL a tensor e normalizza (mm -> metri)
                depth_array = np.array(preprocessed_depth, dtype=np.float32) / 1000.0
                tensor_depth = torch.tensor(depth_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 224, 224)
                
                # Prepare BBox Center
                center_x = x_min + (x_max - x_min) / 2.0
                center_y = y_min + (y_max - y_min) / 2.0
                tensor_center = torch.tensor([[center_x, center_y]], dtype=torch.float32).to(device)
                
                # Inference
                with torch.no_grad():
                    pred_quat, pred_trans, _ = pose_model(
                        tensor_rgb, 
                        tensor_depth, 
                        tensor_center, 
                        tensor_bbox_dims
                    )
                
                # Convert to Numpy for Drawing
                pred_t_np = pred_trans[0].cpu().numpy()
                pred_R_np = quaternion_to_rotation_matrix(pred_quat)[0].cpu().numpy()
                
                # Extract ground truth
                class_id = int(boxes.cls[i])
                obj_id = YOLO_TO_LINEMOD_MAP[class_id]
                
                if sample_id in gt_data:
                    gt_info = gt_data[sample_id][0]  
                    gt_R = np.array(gt_info['cam_R_m2c']).reshape(3, 3)
                    gt_t = np.array(gt_info['cam_t_m2c']) / 1000.0
                    
                    # Draw GROUND TRUTH (Green)
                    output_img = draw_3d_bbox_colored(output_img, gt_R, gt_t, cam_k, obj_id, models_info, color=(0, 255, 0))
                    output_img = draw_axis_colored(output_img, gt_R, gt_t, cam_k, scale=0.05, colors=[(0, 200, 0), (0, 255, 0), (0, 180, 0)])
                    
                    # Draw PREDICTION (Cyan)
                    output_img = draw_3d_bbox_colored(output_img, pred_R_np, pred_t_np, cam_k, obj_id, models_info, color=(255, 165, 0))
                    output_img = draw_axis_colored(output_img, pred_R_np, pred_t_np, cam_k, scale=0.05, colors=[(255, 100, 0), (255, 165, 0), (200, 130, 0)])
                
        
        # Convert to RGB and store
        img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
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