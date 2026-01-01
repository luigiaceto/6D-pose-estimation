import torch
import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from models.FusionPoseNet import FusionPoseNet
from utils.pose_utils import (
    quaternion_to_rotation_matrix,  
    compute_add_metric, 
    compute_add_rotation_only, 
    compute_add_s_metric, 
    compute_add_s_rotation_only, 
    compute_rotation_error,
    compute_translation_error,
    load_model_points, 
    print_evaluation_results_table,
    SYMMETRIC_OBJECTS, 
    )


def evaluate_extension(
    dataset_root,
    test_dataset,
    test_loader,
    cam_k,
    checkpoint_path, 
    device='cuda',
    save_table=False,
    table_path="extension_results.csv"
):
    """
    Evaluation del modello RGB-D Fusion.
    """
    
    model = FusionPoseNet(
        cam_k=cam_k
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Info dataset
    object_diameters = test_dataset.get_object_diameters()
    symmetric_objects = [10, 11] # Eggbox, Glue (verifica gli ID corretti per il tuo dataset LineMod)
    
    # Accumulatori per metriche globali
    all_add = []
    all_add_rotation_only = []
    all_rot_errors = []
    all_trans_errors = []
    all_diameters = []
    
    # Accumulatori per classe
    per_class_metrics = defaultdict(list)
    
    print("Evaluating RGB-D Extension...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Batch"):
            # Sposta dati su GPU
            rgb = batch['cropped_img'].to(device)
            depth = batch['cropped_depth'].to(device)
            bbox_center = batch['bbox_center_pixel'].to(device)
            
            gt_translation = batch['translation'].to(device)
            gt_rotation = batch['rotation'].to(device) # Matrice 3x3
            
            obj_ids = batch['obj_id'].to(device).long().cpu().numpy()
            
            # --- FORWARD PASS ---
            # Il modello ora restituisce direttamente la traslazione finale!
            pred_quat, pred_translation, _ = model(rgb, depth, bbox_center)
            
            # Conversione quaternioni -> matrice rotazione
            pred_rotation = quaternion_to_rotation_matrix(pred_quat)
            
            # Converti a numpy per calcoli metriche
            pred_R_np = pred_rotation.cpu().numpy()
            pred_t_np = pred_translation.cpu().numpy()
            gt_R_np = gt_rotation.cpu().numpy()
            gt_t_np = gt_translation.cpu().numpy()
            
            # Loop sugli oggetti nel batch
            for i in range(len(obj_ids)):
                obj_id = int(obj_ids[i])
                diameter = object_diameters[obj_id]
                
                # Errori base
                rot_err = compute_rotation_error(pred_R_np[i], gt_R_np[i])
                trans_err = compute_translation_error(pred_t_np[i], gt_t_np[i])
                
                # ADD / ADD-S Metric
                model_points = load_model_points(dataset_root, obj_id)
                
                if obj_id in symmetric_objects:
                    # Usa ADD-S per simmetrici
                    add_val = compute_add_s_metric(
                        pred_R_np[i], pred_t_np[i], gt_R_np[i], gt_t_np[i], model_points
                    )
                    add_rotation_only_val = compute_add_s_rotation_only(
                        pred_R_np[i], gt_R_np[i], model_points
                    )
                else:
                    # Usa ADD standard
                    add_val = compute_add_metric(
                        pred_R_np[i], pred_t_np[i], gt_R_np[i], gt_t_np[i], model_points
                    )
                    add_rotation_only_val = compute_add_rotation_only(
                        pred_R_np[i], gt_R_np[i], model_points
                    )
                
                # Salvataggio metriche (ADD in cm per coerenza col print finale)
                add_cm = add_val * 100
                add_rotation_only_cm = add_rotation_only_val * 100
                
                all_rot_errors.append(rot_err)
                all_trans_errors.append(trans_err)
                all_add.append(add_cm)
                all_add_rotation_only.append(add_rotation_only_cm)
                all_diameters.append(diameter)
                
                per_class_metrics[obj_id].append({
                    'rotation': rot_err,
                    'translation': trans_err,
                    'add': add_cm,
                    'add_rotation_only': add_rotation_only_cm
                })

    # --- CALCOLO RISULTATI FINALI ---
    
    # 1. Accuracy @ 10% Diameter
    # (Quanti oggetti hanno errore ADD < 10% del loro diametro?)
    all_add_np = np.array(all_add) # cm
    all_add_rotation_only_np = np.array(all_add_rotation_only) # cm
    all_diameters_cm = np.array(all_diameters) / 10.0 # mm -> cm
    thresholds = all_diameters_cm * 0.1
    
    accuracy = np.mean(all_add_np < thresholds) * 100
    add_r_accuracy = np.mean(all_add_rotation_only_np < thresholds) * 100

    # Generazione Tabella per Classe
    per_class_results = []
    for cls_id, metrics in sorted(per_class_metrics.items()):
        metrics_df = pd.DataFrame(metrics)
        
        cls_diam_cm = object_diameters[cls_id] / 10.0
        cls_thresh = cls_diam_cm * 0.1
        cls_acc = np.mean(metrics_df['add'] < cls_thresh) * 100
        cls_add_r_acc = np.mean(metrics_df['add_rotation_only'] < cls_thresh) * 100
        
        per_class_results.append({
            'class_id': cls_id,
            'num_samples': len(metrics),
            'accuracy_10p': cls_acc,
            'add_r_accuracy_10p': cls_add_r_acc,
            'rot_mean': metrics_df['rotation'].mean(),
            'trans_mean': metrics_df['translation'].mean(),
            'add_mean': metrics_df['add'].mean(),
            'add_rot_only_mean': metrics_df['add_rotation_only'].mean()
        })
        
    # Aggiungi riga "ALL" (Media globale)
    per_class_results.append({
        'class_id': 'MEAN',
        'num_samples': len(all_add),
        'accuracy_10p': accuracy,
        'add_r_accuracy_10p': add_r_accuracy,
        'rot_mean': np.mean(all_rot_errors),
        'trans_mean': np.mean(all_trans_errors),
        'add_mean': np.mean(all_add),
        'add_rot_only_mean': np.mean(all_add_rotation_only)
    })

    # Stampa e Salva CSV
    return print_evaluation_results_table(per_class_results, save_table, table_path)


