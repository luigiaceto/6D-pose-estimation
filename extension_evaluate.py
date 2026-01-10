import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from models.TridentNetPose import TridentNetPose
from utils.pose_utils import (
    quaternion_to_rotation_matrix,  
    load_models_points, 
    print_evaluation_results_table,
    compute_rotation_error,
    compute_ADD,
    compute_ADDS,
    compute_translation_error,
    SYMMETRIC_OBJECTS,
    N_POINTS_TO_LOAD
)


def evaluate_extension_batch(
    dataset_root,
    test_dataset,
    test_loader,
    cam_k,
    checkpoint_path, 
    device='cuda',
    save_table=False,
    table_path="extension_results.csv"
):
    
    model = TridentNetPose(cam_k=cam_k).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    object_diameters = test_dataset.get_object_diameters() 
    num_points = N_POINTS_TO_LOAD
    print("Preloading mesh points for batch evaluation...")
    mesh_points_cache = load_models_points(dataset_root, num_points=num_points)
    
    # Spostiamo tutti i punti sulla GPU subito per velocità
    for k, v in mesh_points_cache.items():
        mesh_points_cache[k] = v.to(device)

    # Accumulatori
    all_add = []
    all_rot_errors = []
    all_trans_errors = []
    all_diameters = []
    per_class_metrics = defaultdict(list)
    
    print("Evaluating RGB-D Extension (BATCH MODE)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Batch"):
            # Dati su GPU
            rgb = batch['cropped_img'].to(device)
            depth = batch['cropped_depth'].to(device)
            bbox_center = batch['bbox_center_pixel'].to(device)
            bbox_dims = batch['bbox_dims'].to(device)
            
            gt_trans = batch['translation'].to(device)     # (B, 3)
            gt_rot_matrix = batch['rotation'].to(device)   # (B, 3, 3)
            obj_ids = batch['obj_id'].to(device)           # (B,)

            # Forward - Modello restituisce (pred_quat, pred_trans, pred_uv)
            pred_quat, pred_trans, _ = model(
                rgb, 
                depth, 
                bbox_center, 
                bbox_dims
            )
            
            pred_rot_matrix = quaternion_to_rotation_matrix(pred_quat) # (B, 3, 3)
            
            # Costruiamo il tensore dei punti per questo batch specifico.
            # Prende i punti corretti per ogni oggetto nel batch e li impila.
            # Risultato: (B, N, 3)
            batch_points = torch.stack([mesh_points_cache[int(oid)] for oid in obj_ids])
            
            # Reshape translation per broadcasting: (B, 3) -> (B, 3, 1)
            pred_t_b = pred_trans.unsqueeze(-1)
            gt_t_b = gt_trans.unsqueeze(-1)
            
            # Calcola ADD (Asimmetrico)
            add_losses = compute_ADD(pred_rot_matrix, gt_rot_matrix, batch_points, pred_t_b, gt_t_b)
            
            # Calcola ADD-S (Simmetrico)
            adds_losses = compute_ADDS(pred_rot_matrix, gt_rot_matrix, batch_points, pred_t_b, gt_t_b)
            
            # Portiamo tutto su CPU per logging e calcoli finali leggeri
            add_res = (add_losses).cpu().numpy()
            adds_res = (adds_losses).cpu().numpy()
            
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
                # Nota: compute_rotation_error è leggero, si può lasciare in numpy
                batch_points_np = batch_points.cpu().numpy()  # Converti batch in numpy
                rot_err = compute_rotation_error(pred_R_np[i], gt_R_np[i], oid, batch_points_np[i])
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
    all_diameters_m = np.array(all_diameters) / 1000.0 # diametri convertiti da mm a cm

    # ADD Accuracy @ 10% diametro oggetti
    thresholds = all_diameters_m * 0.1
    accuracy = np.mean(all_add_np < thresholds) * 100

    per_class_results = []
    for cls_id, metrics in sorted(per_class_metrics.items()):
        metrics_df = pd.DataFrame(metrics)
        cls_diam_m = object_diameters[cls_id] / 1000.0  
        cls_thresh_m = cls_diam_m * 0.1  
        cls_acc = np.mean(metrics_df['add'] < cls_thresh_m) * 100 
        
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