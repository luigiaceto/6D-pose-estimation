import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# Imports dai tuoi moduli
from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera

# Importiamo le metriche e la MAPPA dal file centrale
from baseline_evaluate import (
    compute_rotation_error, 
    compute_translation_error,
    compute_add_metric, 
    compute_add_s_metric, 
    compute_add_rotation_only,
    compute_add_s_rotation_only,
    load_model_points, 
    print_evaluation_results_table,
    YOLO_TO_LINEMOD_MAP  # <--- IMPORTIAMO LA MAPPA CENTRALIZZATA
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
    Pipeline STRICT (Severa):
    1. Usa YOLO per rilevare l'oggetto.
    2. Controlla se la classe predetta (mappata) coincide con il Ground Truth.
    3. Se coincideno: Calcola la posa.
    4. Se NON coincidono: Scarta l'immagine (Errore di Classificazione).
    """
    
    # 1. Setup Modelli
    yolo = YOLO(yolo_model_path)
    
    pose_model = ResNetPose().to(device)
    checkpoint = torch.load(pose_model_path, map_location=device, weights_only=False)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    pinhole = PinholeCamera(cam_k)
    
    if symmetric_objects is None:
        symmetric_objects = [10, 11] 
        
    object_diameters = test_dataset.get_object_diameters()
    
    # Metriche Container
    all_metrics = {
        'rot_err': [], 'trans_err': [], 'add': [], 'add_rot_only': [], 'diameters': []
    }
    per_class_metrics = defaultdict(list)
    
    # Statistiche
    stats = {
        'processed_total': 0,
        'valid_prediction': 0,      # YOLO ha azzeccato tutto (Box + Classe)
        'discarded_miss': 0,        # YOLO non ha trovato nulla
        'discarded_wrong_class': 0  # YOLO ha trovato box ma sbagliato classe
    }

    # Loop
    for batch in tqdm(test_loader, desc="Full Pipeline Eval"):
        
        # Dati Ground Truth
        gt_R = batch["rotation"][0].cpu().numpy()
        gt_t = batch["translation"][0].cpu().numpy()
        gt_obj_id = int(batch["obj_id"][0])
        stats['processed_total'] += 1
        
        # Recupero Path Immagine (Fix per Tensore)
        raw_sample_info = batch['sample_id'][0] 
        folder_str = f"{int(raw_sample_info[0].item()):02d}"
        sample_str = f"{int(raw_sample_info[1].item()):04d}"
        img_path = test_dataset.dataset_root / "data" / folder_str / "rgb" / f"{sample_str}.png"
        
        # ---------------------------------------------------------
        # STEP 1: YOLO DETECTION
        # ---------------------------------------------------------
        results = yolo(str(img_path), verbose=False)[0]
        
        if len(results.boxes) == 0:
            stats['discarded_miss'] += 1
            continue
            
        # Analisi della predizione migliore
        best_idx = results.boxes.conf.argmax()
        
        # MAPPING: ID YOLO (0..12) -> ID LINEMOD (1..15)
        pred_raw_id = int(results.boxes.cls[best_idx].item())
        pred_linemod_id = YOLO_TO_LINEMOD_MAP.get(pred_raw_id, -1)
        
        # --- CHECK SEVERO (STRICT) ---
        if pred_linemod_id != gt_obj_id:
            # Se la classe è sbagliata, SCARTIAMO!
            stats['discarded_wrong_class'] += 1
            continue 
        
        # Se siamo qui, YOLO ha indovinato la classe!
        stats['valid_prediction'] += 1

        # Estrazione Box
        box = results.boxes.xywh[best_idx].cpu().numpy()
        x_c, y_c, w, h = box
        
        # Prepariamo coordinate per il dataset (Top-Left)
        x_tl = x_c - (w / 2)
        y_tl = y_c - (h / 2)
        bbox_for_dataset = [x_tl, y_tl, w, h]
        
        # ---------------------------------------------------------
        # STEP 2: CROP & PREPROCESSING (via Dataset)
        # ---------------------------------------------------------
        try:
            # Usiamo il metodo del dataset per garantire coerenza col training
            crop_tensor = test_dataset.load_cropped_image(str(img_path), bbox_for_dataset)
            crop_tensor = crop_tensor.unsqueeze(0).to(device)
        except:
            # Se il crop fallisce (es. box fuori immagine), contiamolo come miss
            stats['discarded_miss'] += 1
            continue

        # ---------------------------------------------------------
        # STEP 3: POSE ESTIMATION (RESNET)
        # ---------------------------------------------------------
        with torch.no_grad():
            pred_q = pose_model(crop_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q)[0].cpu().numpy()

        # ---------------------------------------------------------
        # STEP 4: GEOMETRY (PINHOLE)
        # ---------------------------------------------------------
        diameter = object_diameters[gt_obj_id]
        
        # Pinhole vuole xyxy
        bbox_xyxy = torch.tensor([[x_tl, y_tl, x_tl+w, y_tl+h]], device=device)
        
        # Calcolo Depth
        depth = pinhole.compute_depth_from_bbox(bbox_xyxy, torch.tensor([diameter], device=device))
        
        # Unprojection Centro
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
            'rotation': r_err, 'translation': t_err, 'add': add_val, 'add_rot_only': add_rot_only
        })
        
        all_metrics['add'].append(add_val)
        all_metrics['add_rot_only'].append(add_rot_only)
        all_metrics['diameters'].append(object_diameters[gt_obj_id])
        all_metrics['rot_err'].append(r_err)
        all_metrics['trans_err'].append(t_err)

    # ---------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE DIAGNOSTICS:")
    print(f"Total Samples: {stats['processed_total']}")
    print(f"✅ Valid Predictions (Box+Class OK): {stats['valid_prediction']}")
    print(f"❌ Discarded (Wrong Class):        {stats['discarded_wrong_class']}")
    print(f"❌ Discarded (No Detection):       {stats['discarded_miss']}")
    print(f"{'='*80}\n")
    
    # Preparazione DataFrame per stampa
    if len(all_metrics['add']) == 0:
        print("⚠️ Nessun campione valido trovato!")
        return None

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
        
    # Totale
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
    
    return print_evaluation_results_table(per_class_results)

def list_to_df(metrics_list):
    return {k: [m[k] for m in metrics_list] for k in metrics_list[0].keys()}