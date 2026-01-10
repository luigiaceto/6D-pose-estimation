from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from utils.pose_utils import (
    quaternion_to_rotation_matrix,  
    compute_ADD, 
    compute_ADDS,
    load_models_points,
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS, 
    N_POINTS_TO_LOAD
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
    
    # Carica i punti del modello 
    num_points = N_POINTS_TO_LOAD
    print(f" Preloading HIGH RES models ({num_points} points per object)...")
    model_points_dict = load_models_points(dataset_root, num_points=num_points)

    max_id = max(model_points_dict.keys())
    n_pts = list(model_points_dict.values())[0].shape[0]
    point_bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32)
    for oid, pts in model_points_dict.items():
        point_bank[oid] = pts.to(device)
            
    # symmetry lookup table (True for symmetric objects)
    symmetry_lookup = torch.zeros(max_id + 1, dtype=torch.bool)
    for obj_id in SYMMETRIC_OBJECTS:
        if obj_id <= max_id:
            symmetry_lookup[obj_id] = True
    
    results_list = []

    print("Evaluating Baseline (BATCH MODE)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Batch"):
            # Dati su GPU
            cropped_img = batch['cropped_img'].to(device)
            bbox_base = batch['bbox_base'].to(device)
            gt_trans = batch['translation'].to(device)    
            gt_rot_matrix = batch['rotation'].to(device) 
            obj_ids = batch['obj_id'].to(device)          
            
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
            pred_rotation_matrix = quaternion_to_rotation_matrix(pred_quaternion)
            
            batch_points = point_bank[obj_ids]
            
            # Reshape per batch functions: (B, 3) -> (B, 3, 1)
            pred_t_batch = pred_trans.unsqueeze(-1)
            gt_t_batch = gt_trans.unsqueeze(-1)
            
            # --- METRICA ADD(-S) ---
            add_batch = compute_ADD(pred_rotation_matrix, gt_rot_matrix, batch_points, pred_t_batch, gt_t_batch)
            adds_batch = compute_ADDS(pred_rotation_matrix, gt_rot_matrix, batch_points, pred_t_batch, gt_t_batch)
            is_symmetric = symmetry_lookup[obj_ids]
            final_add = torch.where(is_symmetric, adds_batch, add_batch)
            
            # --- ERRORE TRASLAZIONE ---
            trans_errors = torch.norm(pred_trans - gt_trans, dim=1) * 100

            # --- ERRORE ROTAZIONE ---
            R_diff = torch.bmm(pred_rotation_matrix.transpose(1, 2), gt_rot_matrix)
            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            cos_theta = ((trace - 1) / 2.0).clamp(-1.0, 1.0)
            rot_errors_deg = torch.rad2deg(torch.acos(cos_theta))
            if is_symmetric.any():
                # Raggio medio approssimato (diametro / 2)
                radii = batch_diameters / 2.0 / 1000.0 # mm -> m
                ratio = (adds_batch / (2 * radii)).clamp(-1.0, 1.0)
                sym_rot_errors = torch.rad2deg(2 * torch.asin(ratio))
                rot_errors_deg = torch.where(is_symmetric, sym_rot_errors, rot_errors_deg)

            # Salvo su CPU
            B = len(obj_ids)
            ids_cpu = obj_ids.cpu().numpy()
            add_cpu = (final_add * 100).cpu().numpy() # m -> cm
            trans_cpu = trans_errors.cpu().numpy()
            rot_cpu = rot_errors_deg.cpu().numpy()
            
            for i in range(B):
                results_list.append({
                    'class_id': int(ids_cpu[i]),
                    'rotation': rot_cpu[i],
                    'translation': trans_cpu[i],
                    'add': add_cpu[i]
                })
               
    
    df_raw = pd.DataFrame(results_list)
    
    # Calcolo statistiche per classe
    per_class_results = []
    
    # Raggruppa per class_id
    grouped = df_raw.groupby('class_id')
    
    for cls_id, group in grouped:
        # Diametro in cm
        cls_diam_cm = object_diameters[cls_id] / 10.0
        threshold = cls_diam_cm * 0.1
        
        accuracy = (group['add'] < threshold).mean() * 100
        
        per_class_results.append({
            'class_id': cls_id,
            'num_samples': len(group),
            'rot_mean': group['rotation'].mean(),
            'trans_mean': group['translation'].mean(),
            'add_mean': group['add'].mean() / 100.0, # torna a metri per compatibilità con print_table
            'accuracy_10p': accuracy
        })

    # Calcolo accuracy globale pesata sui singoli sample
    # (ogni sample ha una soglia diversa in base al suo oggetto)
    valid_samples = 0
    correct_samples = 0
    
    for i, row in df_raw.iterrows():
        cls_id = row['class_id']
        thresh = (object_diameters[cls_id] / 10.0) * 0.1
        if row['add'] < thresh:
            correct_samples += 1
        valid_samples += 1
            
    total_accuracy = (correct_samples / valid_samples * 100) if valid_samples > 0 else 0

    per_class_results.append({
        'class_id': 'MEAN',
        'num_samples': len(df_raw),
        'rot_mean': df_raw['rotation'].mean(),
        'trans_mean': df_raw['translation'].mean(),
        'add_mean': df_raw['add'].mean() / 100.0, # metri
        'accuracy_10p': total_accuracy
    })

    return print_evaluation_results_table(per_class_results, save_table, table_path)