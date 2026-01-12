import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

from models.TridentNetPose import TridentNetPose


from utils.pose_utils import (
    compute_rotation_error,
    compute_translation_error,
    compute_ADD,
    compute_ADDS,
    load_models_points, 
    quaternion_to_rotation_matrix,
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS,
    YOLO_TO_LINEMOD_MAP,
    N_POINTS_TO_LOAD,
    IMG_WIDTH,
    IMG_HEIGHT
)

def evaluate_extension_pipeline(
    dataset_root,
    test_dataset,
    test_loader,
    cam_k,
    yolo_checkpoint=str(Path("checkpoints") / "best_yolo_model.pt"),
    model_checkpoint=str(Path("checkpoints") / "best_fusion_model.pt"),
    device="cuda"
):
    
    if test_loader.batch_size != 1:
        print("WARNING: Full Pipeline Evaluation richiede batch_size=1.")
    
    object_diameters = test_dataset.get_object_diameters()
    
    num_points = N_POINTS_TO_LOAD
    mesh_points_cache = load_models_points(dataset_root)
    # Spostiamo su GPU subito
    for k, v in mesh_points_cache.items():
        mesh_points_cache[k] = v.to(device)
    
    # Pre-build lookup tables per compute_rotation_error (efficienza)
    max_id = max(mesh_points_cache.keys())
    symmetry_lookup = torch.zeros(max_id + 1, dtype=torch.bool, device=device)
    for obj_id in SYMMETRIC_OBJECTS:
        symmetry_lookup[obj_id] = True
    
    model_points_bank = torch.zeros((max_id + 1, num_points, 3), device=device)
    for k, v in mesh_points_cache.items():
        model_points_bank[k] = v

    yolo = YOLO(yolo_checkpoint)
    
    pose_model = TridentNetPose(cam_k=cam_k).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    all_metrics = {
        'rot_err': [], 'trans_err': [], 'add': [], 'diameters': []
    }
    per_class_metrics = defaultdict(list)
    stats = {
        'total': 0,
        'valid': 0,
        'missed': 0, 
        'wrong_class': 0
    }
    
    for batch in tqdm(test_loader, desc="Testing Extension Pipeline"):
        stats['total'] += 1
        
        # Batch size 1 -> prendiamo indice [0] solo per obj_id
        gt_R = batch["rotation"].to(device)       # (1, 3, 3) - keep batch dim!
        gt_t = batch["translation"].to(device)    # (1, 3) - keep batch dim!
        gt_obj_id = int(batch["obj_id"][0])       # scalar
        
        folder_id = int(batch['sample_id'][0][0])
        sample_id = int(batch['sample_id'][0][1])
        rgb_path = dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png"
        
        results = yolo(str(rgb_path), verbose=False)[0]
        
        if len(results.boxes) == 0:
            stats['missed'] += 1
            continue
            
        best_idx = results.boxes.conf.argmax()
        pred_yolo_class = int(results.boxes.cls[best_idx].item())
        pred_obj_id = YOLO_TO_LINEMOD_MAP.get(pred_yolo_class, -1)
        
        # Check Classe
        if pred_obj_id != gt_obj_id:
            stats['wrong_class'] += 1
            continue 
            
        stats['valid'] += 1

        # Estrazione BBox YOLO (Center, w, h)
        box = results.boxes.xywh[best_idx].cpu().numpy() 
        x_c, y_c, w, h = box
        
        # BBox Top-Left per il cropping
        x_tl = x_c - (w / 2)
        y_tl = y_c - (h / 2)
        bbox_yolo_tl = [x_tl, y_tl, w, h]
        
        try:
            rgb_tensor = test_dataset.load_cropped_image(str(rgb_path), bbox_yolo_tl)
            rgb_batch = rgb_tensor.unsqueeze(0).to(device)
            
            depth_tensor = test_dataset.load_cropped_depth(folder_id, sample_id, bbox_yolo_tl)
            depth_batch = depth_tensor.unsqueeze(0).to(device)
            
            bbox_center_tensor = torch.tensor([[x_c, y_c]], dtype=torch.float32).to(device)
            
            w_norm = w / IMG_WIDTH
            h_norm = h / IMG_HEIGHT
            bbox_dims_tensor = torch.tensor([[w_norm, h_norm]], dtype=torch.float32).to(device)

        except Exception as e:
            print(f"Error processing sample {folder_id}/{sample_id}: {e}")
            stats['missed'] += 1
            continue

        with torch.no_grad():
            # Forward pass: RGB + Depth + Box Info (modello restituisce anche pred_uv)
            pred_q, pred_t, _ = pose_model(rgb_batch, depth_batch, bbox_center_tensor, bbox_dims_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q) # (1, 3, 3)

        # Reshape translation per le funzioni batch (1, 3, 1)
        pred_t_b = pred_t.unsqueeze(-1)
        gt_t_b = gt_t.unsqueeze(-1)
        
        # Recupera punti mesh dalla cache
        model_points_torch = mesh_points_cache[gt_obj_id].unsqueeze(0)  # (1, N, 3)

        if gt_obj_id in SYMMETRIC_OBJECTS:
            add_val = compute_ADDS(pred_R, gt_R, model_points_torch, pred_t_b, gt_t_b).item()  # METRI
        else:
            add_val = compute_ADD(pred_R, gt_R, model_points_torch, pred_t_b, gt_t_b).item()  # METRI
            
        add_cm = add_val * 100  # METRI -> CM
        
        # Rotation error
        gt_quat = batch["quaternion"].to(device)  # (1, 4)
        class_id_tensor = torch.tensor([gt_obj_id], device=device, dtype=torch.long)
        r_err = compute_rotation_error(
            pred_q, gt_quat, class_id_tensor, symmetry_lookup, model_points_bank
        )[0].item()  # Extract first element from (1,) tensor
        
        # Translation error
        t_err = compute_translation_error(pred_t, gt_t).item()
        diameter = object_diameters[gt_obj_id]
        
        per_class_metrics[gt_obj_id].append({
            'rotation': r_err, 'translation': t_err, 'add': add_cm
        })
        
        all_metrics['add'].append(add_cm)
        all_metrics['rot_err'].append(r_err)
        all_metrics['trans_err'].append(t_err)
        all_metrics['diameters'].append(diameter)

    if len(all_metrics['add']) == 0:
        print("⚠️ Nessun risultato valido.")
        return None

    print(f"\n{'='*60}")
    print(f"📊 RGB-D PIPELINE STATISTICS")
    print(f"Total Images:       {stats['total']}")
    print(f"✅ Valid Detect:    {stats['valid']} ({(stats['valid']/stats['total'])*100:.1f}%)")
    print(f"⁉️ Missed:          {stats['missed']}")
    print(f"❌ Wrong Class:     {stats['wrong_class']}")
    print(f"{'='*60}\n")

    per_class_results = []
    for class_id, metrics in sorted(per_class_metrics.items()):
        metrics_df = pd.DataFrame(metrics)
        diam_m = object_diameters[class_id] / 1000.0  # MM -> METRI
        threshold_m = diam_m * 0.1  # 10% in METRI
        # metrics_df['add'] è in CM, convertiamo in METRI per confronto
        add_m = metrics_df['add'] / 100.0
        acc_10p = np.mean(add_m < threshold_m) * 100
        acc_2cm = np.mean(add_m < 0.02) * 100  # 2 cm threshold
        
        per_class_results.append({
            'class_id': class_id, 
            'num_samples': len(metrics),
            'rot_mean': metrics_df['rotation'].mean(), 
            'trans_mean': metrics_df['translation'].mean(),  # Already in cm
            'add_mean': metrics_df['add'].mean() / 100.0,  # cm -> meters for print_table
            'accuracy_10p': acc_10p,
            'accuracy_2cm': acc_2cm
        })
        
    all_adds_cm = np.array(all_metrics['add'])  # CM
    all_adds_m = all_adds_cm / 100.0  # CM -> METRI
    all_diams_m = np.array(all_metrics['diameters']) / 1000.0  # MM -> METRI
    acc_all = np.mean(all_adds_m < (all_diams_m * 0.1)) * 100
    acc_2cm_all= np.mean(all_adds_cm < 2) * 100
    
    per_class_results.append({
        'class_id': 'MEAN', 
        'num_samples': len(all_adds_cm),
        'rot_mean': np.mean(all_metrics['rot_err']), 
        'trans_mean': np.mean(all_metrics['trans_err']),  # Already in cm
        'add_mean': np.mean(all_adds_cm) / 100.0,  # cm -> meters for print_table
        'accuracy_10p': acc_all,
        'accuracy_2cm': acc_2cm_all
    })
    
    return print_evaluation_results_table(per_class_results)