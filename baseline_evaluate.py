from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from utils.pose_utils import (
    quaternion_to_rotation_matrix,  
    load_all_models_points,
    print_evaluation_results_table,
    batch_add_loss,
    batch_adds_loss,
    compute_rotation_error,
    compute_translation_error,
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
    
    model = ResNetPose().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Pinhole model per calcolo traslazione
    pinhole = PinholeCamera(cam_k)
    
    # Get object diameters
    object_diameters = test_dataset.get_object_diameters()
    
    print("⏳ Preloading mesh points for batch evaluation...")
    mesh_points_cache = load_all_models_points(dataset_root, num_points=1000)
    print(f"✅ Loaded {len(mesh_points_cache)} objects with 1000 surface points each")
    
    # Spostiamo tutti i punti sulla GPU subito per velocità
    for k, v in mesh_points_cache.items():
        mesh_points_cache[k] = v.to(device)
    
    # Accumulatori
    all_add = []
    all_rot_errors = []
    all_trans_errors = []
    all_diameters = []
    per_class_metrics = defaultdict(list)

    print("Evaluating Baseline (BATCH MODE)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Batch"):
            # Dati su GPU
            cropped_img = batch['cropped_img'].to(device)
            bbox_base = batch['bbox_base'].to(device)
            
            gt_trans = batch['translation'].to(device)    # (B, 3)
            gt_rot_matrix = batch['rotation'].to(device)  # (B, 3, 3)
            obj_ids = batch['obj_id'].to(device)          # (B,)
            
            # Ricostruiamo bbox xyxy per il pinhole
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
            
            # Recuperiamo i diametri corretti per il batch
            batch_diameters = torch.tensor(
                [object_diameters[int(oid)] for oid in obj_ids.cpu()],
                device=device, dtype=torch.float32
            )
            
            depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
            pred_trans = pinhole.unproject_2d_to_3d(center_2d_pixels, depth) # (B, 3)
            
            pred_quaternion = model(cropped_img)  # (B, 4)
            pred_rot_matrix = quaternion_to_rotation_matrix(pred_quaternion) # (B, 3, 3)
            
            # Costruiamo il tensore dei punti 3D per questo batch
            # (B, N, 3)
            batch_points = torch.stack([mesh_points_cache[int(oid)] for oid in obj_ids])
            
            # Reshape translation per broadcasting: (B, 3) -> (B, 3, 1)
            pred_t_b = pred_trans.unsqueeze(-1)
            gt_t_b = gt_trans.unsqueeze(-1)
            
            # Calcola ADD (Asimmetrico) per TUTTI
            add_losses = batch_add_loss(pred_rot_matrix, pred_t_b, gt_rot_matrix, gt_t_b, batch_points)
            
            # Calcola ADD-S (Simmetrico) per TUTTI
            adds_losses = batch_adds_loss(pred_rot_matrix, pred_t_b, gt_rot_matrix, gt_t_b, batch_points)
            
            # Portiamo tutto su CPU per logging
            # Convertiamo in cm (* 100) subito
            add_res = (add_losses * 100).cpu().numpy()
            adds_res = (adds_losses * 100).cpu().numpy()
            
            # Calcolo errori classici (rot in deg, trans in cm)
            batch_size = len(obj_ids)
            pred_R_np = pred_rot_matrix.cpu().numpy()
            gt_R_np = gt_rot_matrix.cpu().numpy()
            pred_t_np = pred_trans.cpu().numpy()
            gt_t_np = gt_trans.cpu().numpy()
            ids_np = obj_ids.cpu().numpy()
            
            for i in range(batch_size):
                oid = int(ids_np[i])
                diameter = object_diameters[oid]
                
                # Selezione Metrica Corretta (ADD vs ADD-S)
                if oid in SYMMETRIC_OBJECTS:
                    final_add = adds_res[i]
                else:
                    final_add = add_res[i]
                
                # Calcolo errori classici (rot/trans)
                rot_err = compute_rotation_error(pred_R_np[i], gt_R_np[i])
                trans_err = compute_translation_error(pred_t_np[i], gt_t_np[i])
                
                all_add.append(final_add)
                all_rot_errors.append(rot_err)
                all_trans_errors.append(trans_err)
                all_diameters.append(diameter)
                
                per_class_metrics[oid].append({
                    'rotation': rot_err,
                    'translation': trans_err,
                    'add': final_add
                })
    
    # --- TABELLA FINALE ---
    all_add_np = np.array(all_add)
    all_diameters_cm = np.array(all_diameters) / 10.0 # diametri convertiti da mm a cm

    # ADD Accuracy @ 10% diametro oggetti
    thresholds = all_diameters_cm * 0.1
    accuracy = np.mean(all_add_np < thresholds) * 100

    per_class_results = []
    for cls_id, metrics in sorted(per_class_metrics.items()):
        metrics_df = pd.DataFrame(metrics)
        cls_diam_cm = object_diameters[cls_id] / 10.0
        cls_thresh = cls_diam_cm * 0.1
        cls_acc = np.mean(metrics_df['add'] < cls_thresh) * 100
        
        per_class_results.append({
            'class_id': cls_id,
            'num_samples': len(metrics),
            'rot_mean': metrics_df['rotation'].mean(),
            'trans_mean': metrics_df['translation'].mean(),
            'add_mean': metrics_df['add'].mean(),
            'accuracy_10p': cls_acc
        })
        
    per_class_results.append({
        'class_id': 'MEAN',
        'num_samples': len(all_add),
        'rot_mean': np.mean(all_rot_errors),
        'trans_mean': np.mean(all_trans_errors),
        'add_mean': np.mean(all_add),
        'accuracy_10p': accuracy
    })

    return print_evaluation_results_table(per_class_results, save_table, table_path)