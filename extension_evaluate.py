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
    compute_add_s_metric
)


def load_model_points(dataset_root, obj_id):
    """
    Carica i punti 3D del modello (bounding box corners) per il calcolo della metrica ADD.
    """
    models_info_path = str(dataset_root / "models" / "models_info.yml")
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners del bounding box 3D
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32) / 1000.0  # mm -> metri
    
    return corners

def compute_rotation_error(pred_R, gt_R):
    """Calcola l'errore di rotazione in gradi."""
    R_diff = pred_R.T @ gt_R
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def compute_translation_error(pred_t, gt_t):
    """Calcola l'errore di traslazione in cm."""
    return np.linalg.norm(pred_t - gt_t) * 100  # metri -> cm

def evaluate(
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
                else:
                    # Usa ADD standard
                    add_val = compute_add_metric(
                        pred_R_np[i], pred_t_np[i], gt_R_np[i], gt_t_np[i], model_points
                    )
                
                # Salvataggio metriche (ADD in cm per coerenza col print finale)
                add_cm = add_val * 100
                
                all_rot_errors.append(rot_err)
                all_trans_errors.append(trans_err)
                all_add.append(add_cm)
                all_diameters.append(diameter)
                
                per_class_metrics[obj_id].append({
                    'rotation': rot_err,
                    'translation': trans_err,
                    'add': add_cm
                })

    # --- CALCOLO RISULTATI FINALI ---
    
    # 1. Accuracy @ 10% Diameter
    # (Quanti oggetti hanno errore ADD < 10% del loro diametro?)
    all_add_np = np.array(all_add) # cm
    all_diameters_cm = np.array(all_diameters) / 10.0 # mm -> cm
    thresholds = all_diameters_cm * 0.1
    
    accuracy = np.mean(all_add_np < thresholds) * 100
    
    print("\n" + "="*50)
    print(f"GLOBAL RESULTS (RGB-D Fusion)")
    print("="*50)
    print(f"Accuracy @ 10% diam: {accuracy:.2f}%")
    print(f"Mean Translation Err: {np.mean(all_trans_errors):.2f} cm")
    print(f"Mean Rotation Err:    {np.mean(all_rot_errors):.2f} deg")
    print(f"Mean ADD:             {np.mean(all_add):.2f} cm")
    print("="*50 + "\n")

    # Generazione Tabella per Classe
    per_class_results = []
    for cls_id, metrics in sorted(per_class_metrics.items()):
        metrics_df = pd.DataFrame(metrics)
        
        cls_diam_cm = object_diameters[cls_id] / 10.0
        cls_thresh = cls_diam_cm * 0.1
        cls_acc = np.mean(metrics_df['add'] < cls_thresh) * 100
        
        per_class_results.append({
            'class_id': cls_id,
            'num_samples': len(metrics),
            'accuracy_10p': cls_acc,
            'rot_mean': metrics_df['rotation'].mean(),
            'trans_mean': metrics_df['translation'].mean(),
            'add_mean': metrics_df['add'].mean()
        })
        
    # Aggiungi riga "ALL" (Media globale)
    per_class_results.append({
        'class_id': 'ALL',
        'num_samples': len(all_add),
        'accuracy_10p': accuracy,
        'rot_mean': np.mean(all_rot_errors),
        'trans_mean': np.mean(all_trans_errors),
        'add_mean': np.mean(all_add)
    })

    # Stampa e Salva CSV
    return print_evaluation_results_table(per_class_results, save_table, table_path)


def print_evaluation_results_table(metrics_per_class, save_table=False, table_path="evaluation_results.csv"):
    """Formatta e stampa la tabella dei risultati."""
    
    # Mappa ID -> Nomi (Adatta se il tuo dataset usa ID diversi)
    LINEMOD_OBJECT_NAMES = {
        1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can", 
        6: "cat", 7: "cup", 8: "driller", 9: "duck", 10: "eggbox", 
        11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone",
        "ALL": "AVERAGE"
    }

    df = pd.DataFrame(metrics_per_class)
    
    # Mapping nomi
    df['Object Name'] = df['class_id'].apply(lambda x: LINEMOD_OBJECT_NAMES.get(x, str(x)))
    
    # Rinomina colonne PRIMA di selezionarle
    df = df.rename(columns={
        'num_samples': '#Samples',
        'accuracy_10p': 'Accuracy @10% (%)',
        'rot_mean': 'Rotation Error (deg)',
        'trans_mean': 'Translation Error (cm)',
        'add_mean': 'ADD (cm)'
    })
    
    # Poi seleziona le colonne rinominate
    df = df[['Object Name', '#Samples', 'Accuracy @10% (%)', 'Rotation Error (deg)', 
             'Translation Error (cm)', 'ADD (cm)']]

    df = df.round(2)
    
    print("\nPer-Class Breakdown:")
    print(df.to_string(index=False))

    if save_table:
        df.to_csv(table_path, index=False)
        print(f"\nResults saved to {table_path}")
        
    return df