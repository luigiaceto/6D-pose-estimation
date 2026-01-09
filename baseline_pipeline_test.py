import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera

from utils.pose_utils import (
    compute_rotation_error, 
    compute_translation_error,
    batch_compute_add_metric, 
    batch_compute_add_s_metric,
    load_models_points, 
    quaternion_to_rotation_matrix,
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS,
    YOLO_TO_LINEMOD_MAP,
    N_POINTS_TO_LOAD
)

def evaluate_baseline_pipeline(
    dataset_root,
    test_dataset,
    test_loader,
    cam_k,
    yolo_checkpoint=str(Path("checkpoints") / "best_yolo_model.pt"),
    model_checkpoint=str(Path("checkpoints") / "best_fusion_model.pt"),
    device="cuda"
):
    
    if test_loader.batch_size != 1:
        print(" WARNING: Full Pipeline Evaluation richiede batch_size=1 per gestire detection variabili.")

    num_points = N_POINTS_TO_LOAD
    print("Preloading mesh points for batch evaluation...")
    mesh_points_cache = load_models_points(dataset_root, num_points=num_points)
    # Spostiamo su GPU subito
    for k, v in mesh_points_cache.items():
        mesh_points_cache[k] = v.to(device)
    print(f"Loaded {len(mesh_points_cache)} objects with {num_points} surface points each")
    
    object_diameters = test_dataset.get_object_diameters()

    yolo = YOLO(yolo_checkpoint)
    
    pose_model = ResNetPose().to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    pinhole = PinholeCamera(cam_k)
    
    # Metriche Container
    all_metrics = {
        'rot_err': [], 'trans_err': [], 'add': [], 'diameters': []
    }
    per_class_metrics = defaultdict(list)
    
    # Statistiche
    stats = {
        'total': 0,
        'valid': 0,      
        'missed': 0,        
        'wrong_class': 0  
    }

    for batch in tqdm(test_loader, desc="Testing Baseline Pipeline"):
        stats['total'] += 1
        
        # Essendo batch=1, prendiamo l'indice [0]
        gt_R = batch["rotation"][0].cpu().numpy()
        gt_t = batch["translation"][0].cpu().numpy()
        gt_obj_id = int(batch["obj_id"][0])
        
        folder_id = int(batch['sample_id'][0][0])
        sample_id = int(batch['sample_id'][0][1])
        img_path = dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png"
        
        results = yolo(str(img_path), verbose=False)[0]
        
        if len(results.boxes) == 0:
            stats['missed'] += 1
            continue
            
        # trova la detection con confidence maggiore
        best_idx = results.boxes.conf.argmax()
        pred_yolo_class = int(results.boxes.cls[best_idx].item())
        
        # Mapping YOLO ID -> LINEMOD ID
        pred_obj_id = YOLO_TO_LINEMOD_MAP.get(pred_yolo_class, -1)
        
        # Check coerenza classe
        if pred_obj_id != gt_obj_id:
            stats['wrong_class'] += 1
            continue 
            
        stats['valid'] += 1

        # Estrazione Bbox YOLO (xywh)
        box = results.boxes.xywh[best_idx].cpu().numpy() # centro_x, centro_y, w, h
        x_c, y_c, w, h = box
        
        # Convertiamo in Top-Left (x, y, w, h)
        x_tl = x_c - (w / 2)
        y_tl = y_c - (h / 2)
        bbox_yolo_tl = [x_tl, y_tl, w, h]

        # Preprocessing
        #
        # Passiamo il BBox predetto da YOLO, non il GT
        try:
            # Ritorna tensore (3, 224, 224) normalizzato
            crop_tensor = test_dataset.load_cropped_image(str(img_path), bbox_yolo_tl)
            crop_batch = crop_tensor.unsqueeze(0).to(device) # (1, 3, 224, 224)
        except Exception as e:
            print(f"Error cropping {img_path}: {e}")
            stats['missed'] += 1
            continue

        # ResNet
        with torch.no_grad():
            pred_q = pose_model(crop_batch)
            pred_R = quaternion_to_rotation_matrix(pred_q)[0].cpu().numpy()

        diameter = object_diameters[gt_obj_id]
        
        # Pinhole richiede bbox formato xyxy
        bbox_xyxy = torch.tensor([[x_tl, y_tl, x_tl+w, y_tl+h]], device=device)
        center_2d = torch.tensor([[x_c, y_c]], device=device)
        batch_diam = torch.tensor([diameter], device=device)
        
        # Calcolo Z e traslazione
        pred_z = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diam)
        pred_t = pinhole.unproject_2d_to_3d(center_2d, pred_z)[0].cpu().numpy()

        # Carica punti modello (già cached su GPU)
        model_points_np = mesh_points_cache[gt_obj_id].cpu().numpy()
        
        r_err = compute_rotation_error(pred_R, gt_R)
        t_err = compute_translation_error(pred_t, gt_t)
        
        # Converti a batch format per le funzioni batch_compute_add_*
        pred_R_torch = torch.from_numpy(pred_R).unsqueeze(0).to(device)  # (1, 3, 3)
        pred_t_torch = torch.from_numpy(pred_t).unsqueeze(0).unsqueeze(-1).to(device)  # (1, 3, 1)
        gt_R_torch = torch.from_numpy(gt_R).unsqueeze(0).to(device)  # (1, 3, 3)
        gt_t_torch = torch.from_numpy(gt_t).unsqueeze(0).unsqueeze(-1).to(device)  # (1, 3, 1)
        model_points_torch = torch.from_numpy(model_points_np).unsqueeze(0).to(device)  # (1, N, 3)
        
        if gt_obj_id in SYMMETRIC_OBJECTS:
            add_val = batch_compute_add_s_metric(pred_R_torch, pred_t_torch, gt_R_torch, gt_t_torch, model_points_torch).item() * 100  # cm
        else:
            add_val = batch_compute_add_metric(pred_R_torch, pred_t_torch, gt_R_torch, gt_t_torch, model_points_torch).item() * 100  # cm
            
        per_class_metrics[gt_obj_id].append({
            'rotation': r_err, 'translation': t_err, 'add': add_val
        })
        
        all_metrics['add'].append(add_val)
        all_metrics['rot_err'].append(r_err)
        all_metrics['trans_err'].append(t_err)
        all_metrics['diameters'].append(diameter)

    if len(all_metrics['add']) == 0:
        print("Nessun risultato valido.")
        return None

    print(f"\n{'='*60}")
    print(f"📊 PIPELINE STATISTICS")
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