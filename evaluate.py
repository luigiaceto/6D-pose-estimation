"""
Evaluation script per 6D Pose Estimation.

Calcola metriche:
- ADD (Average Distance of Model Points)
- ADD-S (per oggetti simmetrici)
- Rotation error (gradi)
- Translation error (cm)
"""

import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera
from models.losses import compute_add_metric, compute_add_s_metric
from data.CustomDatasetPose import CustomDatasetPose
from data.DataLoaderCollating import rgb_collate_fn


def load_model_points(dataset_root, obj_id):
    """Carica corner points 3D del modello."""
    models_info_path = os.path.join(dataset_root, 'models', 'models_info.yml')
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners del bounding box
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32) / 1000.0  # mm -> m
    
    return corners


def compute_rotation_error(pred_R, gt_R):
    """Errore di rotazione in gradi."""
    R_diff = pred_R.T @ gt_R
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_translation_error(pred_t, gt_t):
    """Errore di translation in cm."""
    return np.linalg.norm(pred_t - gt_t) * 100  # m -> cm


def evaluate(
    checkpoint_path='./checkpoints/best_pose_model.pt',
    dataset_root="./datasets/linemod/DenseFusion/Linemod_preprocessed",
    batch_size=16,
    device='cuda'
):
    """
    Evaluation del modello.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cam_params = checkpoint['camera_params']
    
    # Pinhole model
    pinhole = PinholeCamera(
        cam_params['fx'], cam_params['fy'],
        cam_params['cx'], cam_params['cy']
    )
    
    # Dataset
    cam_K = np.array([
        cam_params['fx'], 0, cam_params['cx'],
        0, cam_params['fy'], cam_params['cy'],
        0, 0, 1
    ], dtype=np.float32)
    
    test_dataset = CustomDatasetPose(
        dataset_root=dataset_root,
        split='test',
        train_ratio=0.7,
        seed=42,
        device=device,
        cam_K=cam_K,
        img_mean=checkpoint['image_mean'],
        img_std=checkpoint['image_std']
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=rgb_collate_fn, pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Get object diameters
    object_diameters = test_dataset.get_object_diameters()
    
    # Model
    model = ResNetPose(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Metriche
    symmetric_objects = [2, 10]  # eggbox, glue
    all_add = []
    all_add_s = []
    all_rot_errors = []
    all_trans_errors = []
    all_object_ids = []  # Per breakdown per oggetto
    all_diameters = []   # Per calcolare accuracy @ 10%
    
    IMG_WIDTH, IMG_HEIGHT = 640, 480
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            cropped_img = batch['cropped_img'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            gt_translation = batch['translation'].to(device)
            gt_rotation = batch['rotation'].to(device)
            bbox_base = batch['bbox_base'].to(device)
            obj_id = batch['obj_id'].to(device).long()
            obj_ids = obj_id.cpu().numpy()
            
            # Forward: ResNet predice SOLO quaternion
            pred_quaternion = model(cropped_img)  # (B, 4)
            
            # Calcola translation da bbox + diametro (come in train.py)
            bbox_xyxy = torch.stack([
                bbox_base[:, 0],
                bbox_base[:, 1],
                bbox_base[:, 0] + bbox_base[:, 2],
                bbox_base[:, 1] + bbox_base[:, 3]
            ], dim=1)
            
            center_2d_pixels = torch.stack([
                (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
            ], dim=1)
            
            batch_diameters = torch.tensor(
                [object_diameters[int(oid)] for oid in obj_id.cpu()],
                device=device, dtype=torch.float32
            )
            
            depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
            pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
            
            # Rotation matrix
            pred_rotation = quaternion_to_rotation_matrix(pred_quaternion)
            
            # Converti a numpy
            pred_R = pred_rotation.cpu().numpy()
            pred_t = pred_translation.cpu().numpy()
            gt_R = gt_rotation.cpu().numpy()
            gt_t = gt_translation.cpu().numpy()
            
            # Per ogni sample nel batch
            for i in range(len(obj_ids)):
                obj_id = int(obj_ids[i])
                
                # Carica model points
                model_points = load_model_points(dataset_root, obj_id)
                
                # Rotation e translation errors
                rot_err = compute_rotation_error(pred_R[i], gt_R[i])
                trans_err = compute_translation_error(pred_t[i], gt_t[i])
                
                all_rot_errors.append(rot_err)
                all_trans_errors.append(trans_err)
                all_object_ids.append(obj_id)
                all_diameters.append(object_diameters[obj_id])
                
                # ADD o ADD-S
                if obj_id in symmetric_objects:
                    add_s = compute_add_s_metric(
                        pred_R[i], pred_t[i], gt_R[i], gt_t[i], model_points
                    )
                    all_add_s.append(add_s * 100)  # m -> cm
                    all_add.append(add_s * 100)  # Per calcolo complessivo
                else:
                    add = compute_add_metric(
                        pred_R[i], pred_t[i], gt_R[i], gt_t[i], model_points
                    )
                    all_add.append(add * 100)  # m -> cm
    
    # Risultati
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    all_add_np = np.array(all_add)
    all_diameters_np = np.array(all_diameters)
    
    # Converti diametri da mm a cm per confronto
    all_diameters_cm = all_diameters_np / 10.0
    
    # Calcola accuracy a diverse soglie
    threshold_10 = all_diameters_cm * 0.1  # 10% del diametro
    threshold_5 = all_diameters_cm * 0.05   # 5% del diametro
    threshold_2 = all_diameters_cm * 0.02   # 2% del diametro
    
    acc_10 = np.mean(all_add_np < threshold_10) * 100
    acc_5 = np.mean(all_add_np < threshold_5) * 100
    acc_2 = np.mean(all_add_np < threshold_2) * 100
    
    print(f"\n ADD ACCURACY (% predictions below threshold):")
    print(f"  @ 10% diameter: {acc_10:.2f}% ← STANDARD METRIC")
    print(f"  @ 5% diameter:  {acc_5:.2f}%")
    print(f"  @ 2% diameter:  {acc_2:.2f}%")
    
    # Interpretazione
    print(f"\Performance Level:")
    if acc_10 >= 80:
        print(f" EXCELLENT (≥80%)")
    elif acc_10 >= 60:
        print(f"GOOD (60-80%)")
    elif acc_10 >= 40:
        print(f"MODERATE (40-60%)")
    elif acc_10 >= 20:
        print(f"WEAK (20-40%)")
    else:
        print(f"POOR (<20%)")
    
    if all_add:
        print(f"\nADD Mean Error (all objects):")
        print(f"  Mean: {np.mean(all_add):.2f} cm")
        print(f"  Median: {np.median(all_add):.2f} cm")
        print(f"  Std Dev: {np.std(all_add):.2f} cm")
    
    if all_add_s:
        non_sym_add = [all_add[i] for i, oid in enumerate(all_object_ids) if oid not in symmetric_objects]
        if non_sym_add:
            print(f"\n  ADD (non-symmetric only):")
            print(f"    Mean: {np.mean(non_sym_add):.2f} cm")
        
        sym_add = [all_add_s[i] for i in range(len(all_add_s))]
        if sym_add:
            print(f"\n  ADD-S (symmetric only):")
            print(f"    Mean: {np.mean(sym_add):.2f} cm")
    
    print(f"\nRotation Error:")
    print(f"  Mean: {np.mean(all_rot_errors):.2f}°")
    print(f"  Median: {np.median(all_rot_errors):.2f}°")
    print(f"  % < 5°:  {np.mean(np.array(all_rot_errors) < 5) * 100:.1f}%")
    print(f"  % < 10°: {np.mean(np.array(all_rot_errors) < 10) * 100:.1f}%")
    
    print(f"\n Translation Error:")
    print(f"  Mean: {np.mean(all_trans_errors):.2f} cm")
    print(f"  Median: {np.median(all_trans_errors):.2f} cm")
    print(f"  % < 2 cm:  {np.mean(np.array(all_trans_errors) < 2) * 100:.1f}%")
    print(f"  % < 5 cm:  {np.mean(np.array(all_trans_errors) < 5) * 100:.1f}%")
    
    print(f"\n Per-Object Breakdown:")
    unique_objs = sorted(set(all_object_ids))
    
    for obj_id in unique_objs:
        obj_indices = [i for i, o in enumerate(all_object_ids) if o == obj_id]
        obj_add = [all_add[i] for i in obj_indices]
        obj_rot = [all_rot_errors[i] for i in obj_indices]
        obj_diameter = all_diameters[obj_id] / 10.0  # mm → cm
        
        obj_acc_10 = np.mean(np.array(obj_add) < (obj_diameter * 0.1)) * 100
        
        sym_marker = " (SYM)" if obj_id in symmetric_objects else ""
        print(f"  Object {obj_id:2d}{sym_marker}: "
              f"ADD={np.mean(obj_add):5.2f}cm, "
              f"Rot={np.mean(obj_rot):5.2f}°, "
              f"Acc@10%={obj_acc_10:5.1f}% "
              f"(n={len(obj_indices)})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    evaluate(
        checkpoint_path='./checkpoints/best_pose_model.pt',
        dataset_root="./datasets/linemod/DenseFusion/Linemod_preprocessed",
        batch_size=16,
        device='cuda'
    )
