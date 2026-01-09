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
    batch_add_loss,
    batch_adds_loss,
    load_all_models_points, 
    quaternion_to_rotation_matrix,
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS,
    YOLO_TO_LINEMOD_MAP
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
        print("⚠️ WARNING: Full Pipeline Evaluation richiede batch_size=1.")
    
    object_diameters = test_dataset.get_object_diameters()
    
    print("⏳ Preloading mesh points for GPU evaluation...")
    mesh_points_cache = load_all_models_points(dataset_root, num_points=1000)
    # Spostiamo su GPU subito
    for k, v in mesh_points_cache.items():
        mesh_points_cache[k] = v.to(device)
    print(f"✅ Loaded {len(mesh_points_cache)} objects.")

    yolo = YOLO(yolo_checkpoint)
    
    pose_model = TridentNetPose(cam_k=cam_k).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
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
        
        # Batch size 1 -> prendiamo indice [0]
        gt_R = batch["rotation"].to(device)       # (1, 3, 3)
        gt_t = batch["translation"].to(device)    # (1, 3)
        gt_obj_id = int(batch["obj_id"][0])
        
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
            bbox_dims_tensor = torch.tensor([[w, h]], dtype=torch.float32).to(device)

        except Exception as e:
            print(f"Error processing sample {folder_id}/{sample_id}: {e}")
            stats['missed'] += 1
            continue

        with torch.no_grad():
            # Forward pass: RGB + Depth + Box Info
            pred_q, pred_t, _ = pose_model(rgb_batch, depth_batch, bbox_center_tensor, bbox_dims_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q) # (1, 3, 3)

        # Reshape translaton per le funzioni batch (1, 3, 1)
        pred_t_b = pred_t.unsqueeze(-1)
        gt_t_b = gt_t.unsqueeze(-1)
        
        # Recupera punti mesh dalla cache (1, N, 3)
        batch_points = mesh_points_cache[gt_obj_id].unsqueeze(0) 

        if gt_obj_id in SYMMETRIC_OBJECTS:
            loss_val = batch_adds_loss(pred_R, pred_t_b, gt_R, gt_t_b, batch_points)
        else:
            loss_val = batch_add_loss(pred_R, pred_t_b, gt_R, gt_t_b, batch_points)
            
        add_cm = loss_val.item() * 100 # m -> cm
        r_err = compute_rotation_error(pred_R[0].cpu().numpy(), gt_R[0].cpu().numpy())
        t_err = compute_translation_error(pred_t[0].cpu().numpy(), gt_t[0].cpu().numpy())
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
        diam_cm = object_diameters[class_id] / 10.0
        acc_10p = np.mean(metrics_df['add'] < (diam_cm * 0.1)) * 100
        
        per_class_results.append({
            'class_id': class_id, 
            'num_samples': len(metrics),
            'rot_mean': metrics_df['rotation'].mean(), 
            'trans_mean': metrics_df['translation'].mean(),
            'add_mean': metrics_df['add'].mean(), 
            'accuracy_10p': acc_10p
        })
        
    all_adds = np.array(all_metrics['add'])
    all_diams_cm = np.array(all_metrics['diameters']) / 10.0
    acc_all = np.mean(all_adds < (all_diams_cm * 0.1)) * 100
    
    per_class_results.append({
        'class_id': 'MEAN', 
        'num_samples': len(all_adds),
        'rot_mean': np.mean(all_metrics['rot_err']), 
        'trans_mean': np.mean(all_metrics['trans_err']),
        'add_mean': np.mean(all_adds), 
        'accuracy_10p': acc_all
    })
    
    return print_evaluation_results_table(per_class_results)