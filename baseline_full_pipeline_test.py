import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# Imports dai tuoi moduli
from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera

# Importiamo le metriche dal file di evaluate esistente per non riscriverle
from baseline_evaluate import (
    compute_rotation_error, 
    compute_translation_error,
    compute_add_metric, 
    compute_add_s_metric, 
    compute_add_rotation_only,
    compute_add_s_rotation_only,
    load_model_points, 
    print_evaluation_results_table
)

def evaluate_full_pipeline(
    test_dataset,
    test_loader,
    yolo_model_path,
    pose_model_path,
    cam_k,
    symmetric_objects=None,
    device="cuda"
):
    """
    Valuta la pipeline completa: YOLO -> Crop (via Dataset) -> ResNet -> Pinhole.
    """
    
    # 1. Setup Modelli
    yolo = YOLO(yolo_model_path)
    
    pose_model = ResNetPose().to(device)
    checkpoint = torch.load(pose_model_path, map_location=device, weights_only=False)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    pinhole = PinholeCamera(cam_k)
    
    # 2. Setup Dati
    if symmetric_objects is None:
        symmetric_objects = [10, 11] # Default fallback (eggbox, glue)
        
    object_diameters = test_dataset.get_object_diameters()
    
    # Metriche Container
    all_metrics = {
        'rot_err': [], 'trans_err': [], 
        'add': [], 'add_rot_only': [], 
        'diameters': []
    }
    per_class_metrics = defaultdict(list)
    
    # Contatori statistici
    stats = {
        'processed': 0,
        'skipped_yolo_miss': 0,
        'skipped_invalid_id': 0
    }

    
    # Iteriamo sul DataLoader
    for batch in tqdm(test_loader, desc="Pipeline Eval"):
        
        # Ground Truth
        gt_R = batch["rotation"][0].cpu().numpy()
        gt_t = batch["translation"][0].cpu().numpy()
        gt_obj_id = int(batch["obj_id"][0])
        
        # --- FIX QUI SOTTO ---
        # Recuperiamo Folder ID e Sample ID dal tensore combinato
        raw_sample_info = batch['sample_id'][0] # Shape [2]: [folder_id, sample_id]
        
        folder_val = int(raw_sample_info[0].item())
        sample_val = int(raw_sample_info[1].item())
        
        folder_str = f"{folder_val:02d}"
        sample_str = f"{sample_val:04d}"
        
        img_path = test_dataset.dataset_root / "data" / folder_str / "rgb" / f"{sample_str}.png"
        # ---------------------
        
        # ---------------------------------------------------------
        # STEP 1: YOLO DETECTION
        # ---------------------------------------------------------
        results = yolo(str(img_path), verbose=False)[0]
        
        if len(results.boxes) == 0:
            stats['skipped_yolo_miss'] += 1
            continue
            
        # Trova la box migliore (highest confidence)
        best_box_idx = results.boxes.conf.argmax()
        box = results.boxes.xywh[best_box_idx].cpu().numpy() # x_c, y_c, w, h
        
        x_c, y_c, w, h = box
        x_tl = x_c - (w / 2)
        y_tl = y_c - (h / 2)
        bbox_for_dataset = [x_tl, y_tl, w, h]
        
        # ---------------------------------------------------------
        # STEP 2: CROP & PREPROCESSING (RE-USE DATASET LOGIC!)
        # ---------------------------------------------------------
        try:
            crop_tensor = test_dataset.load_cropped_image(str(img_path), bbox_for_dataset)
            crop_tensor = crop_tensor.unsqueeze(0).to(device)
            
        except Exception as e:
            stats['skipped_yolo_miss'] += 1
            continue

        # ---------------------------------------------------------
        # STEP 3: POSE ESTIMATION (RESNET)
        # ---------------------------------------------------------
        with torch.no_grad():
            pred_q = pose_model(crop_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q)[0].cpu().numpy()

        # ---------------------------------------------------------
        # STEP 4: GEOMETRIC TRANSLATION (PINHOLE)
        # ---------------------------------------------------------
        diameter = object_diameters[gt_obj_id]
        
        x, y, w, h = bbox_for_dataset
        bbox_xyxy = torch.tensor([[x, y, x+w, y+h]], device=device)
        
        depth = pinhole.compute_depth_from_bbox(
            bbox_xyxy, 
            torch.tensor([diameter], device=device)
        )
        
        center_2d = torch.tensor([[x_c, y_c]], device=device)
        pred_t = pinhole.unproject_2d_to_3d(center_2d, depth)[0].cpu().numpy()

        # ---------------------------------------------------------
        # STEP 5: METRICS
        # ---------------------------------------------------------
        model_points = load_model_points(test_dataset.dataset_root, gt_obj_id)
        
        r_err = compute_rotation_error(pred_R, gt_R)
        t_err = compute_translation_error(pred_t, gt_t)
        
        if gt_obj_id in symmetric_objects:
            add_val = compute_add_s_metric(pred_R, pred_t, gt_R, gt_t, model_points) * 100
            add_rot_only = compute_add_s_rotation_only(pred_R, gt_R, model_points) * 100
        else:
            add_val = compute_add_metric(pred_R, pred_t, gt_R, gt_t, model_points) * 100
            add_rot_only = compute_add_rotation_only(pred_R, gt_R, model_points) * 100
            
        per_class_metrics[gt_obj_id].append({
            'rotation': r_err, 
            'translation': t_err, 
            'add': add_val, 
            'add_rot_only': add_rot_only
        })
        
        all_metrics['add'].append(add_val)
        all_metrics['add_rot_only'].append(add_rot_only)
        all_metrics['diameters'].append(object_diameters[gt_obj_id])
        all_metrics['rot_err'].append(r_err)
        all_metrics['trans_err'].append(t_err)
        
        stats['processed'] += 1

    # ---------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------
    per_class_results = []
    
    for class_id, metrics in per_class_metrics.items():
        m_df = list_to_df(metrics)
        
        diam_cm = object_diameters[class_id] / 10.0
        thresh = diam_cm * 0.1
        acc = np.mean(np.array(m_df['add']) < thresh) * 100
        acc_r = np.mean(np.array(m_df['add_rot_only']) < thresh) * 100
        
        per_class_results.append({
            'class_id': class_id,
            'num_samples': len(metrics),
            'accuracy_10p': acc,
            'add_r_accuracy_10p': acc_r,
            'rot_mean': np.mean(m_df['rotation']),
            'trans_mean': np.mean(m_df['translation']),
            'add_mean': np.mean(m_df['add']),
            'add_rot_only_mean': np.mean(m_df['add_rot_only'])
        })
        
    all_diams_cm = np.array(all_metrics['diameters']) / 10.0
    all_adds = np.array(all_metrics['add'])
    all_adds_r = np.array(all_metrics['add_rot_only'])
    thresh_all = all_diams_cm * 0.1
    
    acc_all = np.mean(all_adds < thresh_all) * 100
    acc_r_all = np.mean(all_adds_r < thresh_all) * 100
    
    per_class_results.append({
        'class_id': 'ALL',
        'num_samples': len(all_adds),
        'accuracy_10p': acc_all,
        'add_r_accuracy_10p': acc_r_all,
        'rot_mean': np.mean(all_metrics['rot_err']),
        'trans_mean': np.mean(all_metrics['trans_err']),
        'add_mean': np.mean(all_adds),
        'add_rot_only_mean': np.mean(all_adds_r)
    })
    
    print(f"\n{'='*80}")
    print(f"PIPELINE RESULTS: {stats['processed']} processed, {stats['skipped_yolo_miss']} missed by YOLO")
    print(f"{'='*80}\n")
    
    return print_evaluation_results_table(per_class_results)

def list_to_df(metrics_list):
    """Helper veloce per convertire lista di dict in dict di liste/array"""
    return {
        'rotation': [m['rotation'] for m in metrics_list],
        'translation': [m['translation'] for m in metrics_list],
        'add': [m['add'] for m in metrics_list],
        'add_rot_only': [m['add_rot_only'] for m in metrics_list]
    }