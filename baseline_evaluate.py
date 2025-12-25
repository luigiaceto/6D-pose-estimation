"""
Evaluation script per 6D Pose Estimation.

Calcola metriche:
- ADD (Average Distance of Model Points)
- ADD-S (per oggetti simmetrici)
- Rotation error (gradi)
- Translation error (cm)
"""

import os
from pathlib import Path
import torch
import numpy as np
import yaml
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera
from models.losses import compute_add_metric, compute_add_rotation_only, compute_add_s_metric, compute_add_s_rotation_only


def load_model_points(dataset_root, obj_id):
    """Carica corner points 3D del modello."""
    models_info_path = str(dataset_root / "models" / "models_info.yml")
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
    dataset_root,
    test_dataset,
    test_loader,
    cam_k,
    checkpoint_path=str(Path("checkpoints") / "best_pose_model.pt"), 
    device='cuda',
    save_table=False,
    table_path="evaluation_results.csv"
):
    """
    Evaluation del modello baseline.
    """
    
    # Model & load checkpoint
    model = ResNetPose().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Pinhole model
    pinhole = PinholeCamera(cam_k)
    
    # Get object diameters
    object_diameters = test_dataset.get_object_diameters()
    
    # Metriche
    symmetric_objects = [2, 10]  # eggbox, glue
    all_add = []
    all_add_rotation_only = []
    all_add_s = []
    all_rot_errors = []
    all_trans_errors = []
    all_object_ids = []  # Per breakdown per oggetto
    all_diameters = []   # Per calcolare accuracy @ 10%
    
    IMG_WIDTH, IMG_HEIGHT = 640, 480

    # collect metrics per classe
    per_class_metrics= defaultdict(list)

    print("Evaluating...")
    with torch.no_grad():
        
        for batch in tqdm(test_loader):
            cropped_img = batch['cropped_img'].to(device)
            gt_translation = batch['translation'].to(device)
            gt_rotation = batch['rotation'].to(device)
            bbox_base = batch['bbox_base'].to(device)
            obj_id = batch['obj_id'].to(device).long()
            obj_ids = obj_id.cpu().numpy()
            
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
            
            pred_quaternion = model(cropped_img)  # (B, 4)
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
                model_points = load_model_points(str(dataset_root), obj_id)
                
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
                    add_s_rotation_only = compute_add_s_rotation_only(
                        pred_R[i], gt_R[i], model_points
                    )
                    all_add_rotation_only.append(add_s_rotation_only * 100)
                    per_class_metrics[obj_id].append({ 'rotation': rot_err, 'translation': trans_err, 'add': add_s * 100, 'add_rotation_only': add_s_rotation_only * 100 })
                else:
                    add = compute_add_metric(
                        pred_R[i], pred_t[i], gt_R[i], gt_t[i], model_points
                    )
                    all_add.append(add * 100)  # m -> cm
                    add_rotation_only = compute_add_rotation_only(
                        pred_R[i], gt_R[i], model_points
                    )
                    all_add_rotation_only.append(add_rotation_only * 100)
                    per_class_metrics[obj_id].append({ 'rotation': rot_err, 'translation': trans_err, 'add': add * 100, 'add_rotation_only': add_rotation_only * 100 })
    
    all_add_np = np.array(all_add)
    all_diameters_np = np.array(all_diameters)
  

    # Converti diametri da mm a cm per confronto
    all_diameters_cm = all_diameters_np / 10.0
    
    # Accuracy @ 10% diameter (metrica standard)
    threshold_10 = all_diameters_cm * 0.1
    accuracy = np.mean(all_add_np < threshold_10) * 100


    # Interpretazione
    if accuracy >= 80:
        level = "EXCELLENT"
    elif accuracy >= 60:
        level = "GOOD"
    elif accuracy >= 40:
        level = "MODERATE"
    else:
        level = "POOR"
    print(f"Performance Level: {level}\n")


    per_class_results=[]
    for class_id, metrics in per_class_metrics.items():
        if len(metrics) == 0:
            continue

        rot_errors = np.array([m['rotation'] for m in metrics])
        trans_errors = np.array([m['translation'] for m in metrics])
        add_errors = np.array([m['add'] for m in metrics])
        add_rotation_only_errors = np.array([m['add_rotation_only'] for m in metrics])
        
        # accuracy @ 10% diameter
        class_diameter_cm = object_diameters[class_id] / 10.0
        threshold = 0.1 * class_diameter_cm
        accuracy = np.mean(add_errors < threshold) * 100
        
        per_class_results.append({
        'class_id': class_id,
        'num_samples': len(metrics),
        'accuracy_10p': accuracy,
        'rot_mean': rot_errors.mean(),
        'trans_mean': trans_errors.mean(),
        'add_mean': add_errors.mean(),
        'add_rot_only_mean': add_rotation_only_errors.mean(),
        })
    
    # add total avg last row
    per_class_results.append({
        'class_id': 'ALL',
        'num_samples': len(all_add),
        'accuracy_10p': accuracy,
        'rot_mean': np.mean(all_rot_errors),
        'trans_mean': np.mean(all_trans_errors),
        'add_mean': np.mean(all_add),
        'add_rot_only_mean': np.mean(all_add_rotation_only),
    })

    return print_evaluation_results_table(per_class_results, save_table, table_path)   


def print_evaluation_results_table(metrics_per_class, save_table=False, table_path="evaluation_results.csv"):
    
    LINEMOD_OBJECT_NAMES = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
    "ALL": "ALL"
    }

    df = pd.DataFrame(metrics_per_class)
    df['Object Name'] = df['class_id'].map(LINEMOD_OBJECT_NAMES)
    df = df.drop(columns=['class_id'])
    df = df.rename(columns={
        'object_name': 'Object Name',
        'num_samples': '#Samples',
        'accuracy_10p': 'Accuracy @10% (%)',
        'rot_mean': 'Rotation Error (deg)',
        'trans_mean': 'Translation Error (cm)',
        'add_mean': 'ADD / ADD-S (cm)',
        'add_rot_only_mean': ' ADD (rot only) (cm)',
    })

    df = df[
        [
            'Object Name',
            '#Samples',
            'Accuracy @10% (%)',
            'Rotation Error (deg)',
            'Translation Error (cm)',
            'ADD / ADD-S (cm)',
            ' ADD (rot only) (cm)',
        ]
    ]

    df = df.round(2)
    #df = df.sort_values(by='Object ID', ascending=True)

    if save_table:
        df.to_csv(table_path, index=False)
        print(f"Saved CSV to {table_path}")
    return df
