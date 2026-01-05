"""
Evaluation script per 6D Pose Estimation.

Calcola metriche:
- ADD (Average Distance of Model Points)
- ADD-S (per oggetti simmetrici)
- Rotation error (gradi)
- Translation error (cm)
"""

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from utils.pose_utils import (
    quaternion_to_rotation_matrix,  
    compute_add_metric, 
    compute_add_rotation_only, 
    compute_add_s_metric, 
    compute_add_s_rotation_only, 
    compute_rotation_error,
    compute_translation_error,
    load_all_models_points,
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS, 
    )


def evaluate_baseline(
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
    
    # --- FIX CRITICO: CARICA I 1000 PUNTI QUI ---
    print(">>> 📦 Preloading HIGH RES models (1000 points per object)...")
    model_points_dict = load_all_models_points(dataset_root, num_points=1000)
    print(f"    Loaded {len(model_points_dict)} objects with 1000 surface points each")
    
    # Metriche
    symmetric_objects = SYMMETRIC_OBJECTS
    all_add = []
    all_add_rotation_only = []
    all_add_s = []
    all_rot_errors = []
    all_trans_errors = []
    all_object_ids = []  # Per breakdown per oggetto
    all_diameters = []   # Per calcolare accuracy @ 10%
    

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
                
                # --- FIX CRITICO: USA IL DIZIONARIO CON 1000 PUNTI ---
                # NON usare load_model_points che restituisce solo gli 8 punti del bbox!
                model_points = model_points_dict[obj_id].cpu().numpy()  # Converti tensor -> numpy
                
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
                    per_class_metrics[obj_id].append(
                        {
                            'rotation': rot_err,
                            'translation': trans_err,
                            'add': add_s * 100,
                            'add_rotation_only': add_s_rotation_only * 100 
                        }
                    )
                else:
                    add = compute_add_metric(
                        pred_R[i], pred_t[i], gt_R[i], gt_t[i], model_points
                    )
                    all_add.append(add * 100)  # m -> cm
                    add_rotation_only = compute_add_rotation_only(
                        pred_R[i], gt_R[i], model_points
                    )
                    all_add_rotation_only.append(add_rotation_only * 100)
                    per_class_metrics[obj_id].append(
                        { 
                            'rotation': rot_err,
                            'translation': trans_err,
                            'add': add * 100,
                            'add_rotation_only': add_rotation_only * 100
                        }
                    )
    
    all_add_np = np.array(all_add)
    all_diameters_np = np.array(all_diameters)
  

    # Converti diametri da mm a cm per confronto
    all_diameters_cm = all_diameters_np / 10.0
    
    # Accuracy @ 10% diameter (metrica standard)
    threshold_10 = all_diameters_cm * 0.1
    accuracy = np.mean(all_add_np < threshold_10) * 100

    per_class_results=[]
    for class_id, metrics in per_class_metrics.items():
        if len(metrics) == 0:
            continue

        class_rot_errors = np.array([m['rotation'] for m in metrics])
        class_trans_errors = np.array([m['translation'] for m in metrics])
        class_add_errors = np.array([m['add'] for m in metrics])
        class_add_rotation_only_errors = np.array([m['add_rotation_only'] for m in metrics])
        
        # accuracy @ 10% diameter
        class_diameter_cm = object_diameters[class_id] / 10.0
        class_threshold = 0.1 * class_diameter_cm
        class_accuracy = np.mean(class_add_errors < class_threshold) * 100
        
        # ADD-R accuracy @ 10% diameter (rotation only)
        class_add_r_accuracy = np.mean(class_add_rotation_only_errors < class_threshold) * 100
        
        per_class_results.append(
            {
                'class_id': class_id,
                'num_samples': len(metrics),
                'accuracy_10p': class_accuracy,
                'add_r_accuracy_10p': class_add_r_accuracy,
                'rot_mean': class_rot_errors.mean(),
                'trans_mean': class_trans_errors.mean(),
                'add_mean': class_add_errors.mean(),
                'add_rot_only_mean': class_add_rotation_only_errors.mean()
            }
        )
    
    # ADD-R accuracy @ 10% diameter (rotation only)
    all_add_rot_only_np = np.array(all_add_rotation_only)
    add_r_accuracy = np.mean(all_add_rot_only_np < threshold_10) * 100
    
    per_class_results.append(
        {
            'class_id': 'MEAN',
            'num_samples': len(all_add),
            'accuracy_10p': accuracy,
            'add_r_accuracy_10p': add_r_accuracy,
            'rot_mean': np.mean(all_rot_errors),
            'trans_mean': np.mean(all_trans_errors),
            'add_mean': np.mean(all_add),
            'add_rot_only_mean': np.mean(all_add_rotation_only)
        }
    )

    return print_evaluation_results_table(per_class_results, save_table, table_path)   
