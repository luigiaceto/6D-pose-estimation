import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import trimesh

IMG_WIDTH = 640
IMG_HEIGHT = 480

SYMMETRIC_OBJECTS = [10, 11]
N_POINTS_TO_LOAD = 2000

# Traduzione da YOLO (0, 1, 2, ...) a LINEMOD (1, 2, 4, ...)
YOLO_TO_LINEMOD_MAP = { 
    0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15 
}

LINEMOD_OBJECT_NAMES = {
    1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
    12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone",
    "MEAN": "MEAN"
}


def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]


def quaternion_to_rotation_matrix(quaternion):
    """
    Converte quaternion (B, 4) a rotation matrix (B, 3, 3).
    Normalizza l'input per sicurezza numerica.
    """
    # Normalizza per sicurezza (eps evita divisioni per zero)
    quaternion = F.normalize(quaternion, p=2, dim=1, eps=1e-8)
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    batch_size = quaternion.shape[0]
    R = torch.zeros(batch_size, 3, 3, device=quaternion.device, dtype=quaternion.dtype)
    
    # Formula di Rodrigues semplificata per quaternioni unitari
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R


def compute_translation_from_depth_crop(cropped_depth, pred_uv, cam_k):
    """
    Calcola la coordinata Z (depth) usando un ROBUSTO PERCENTILE vettorizzato.
    
    Bug #2 Fix: Invece di campionare solo il centro fisso (che fallisce con jitter),
    usa il 10° percentile dell'intero crop valido (assume l'oggetto sia la cosa più vicina).
    
    IMPORTANTE: Assume che cropped_depth sia GIÀ IN METRI (garantito dal dataset).
    """
    B, _, H, W = cropped_depth.shape 
    
    depth_m = cropped_depth.clone().detach() # metri
    
    # --- Sampling Robusto su TUTTO il Crop (non solo centro) ---
    # Con il jitter del bbox, l'oggetto può essere decentrato.
    # Prendo quindi il PERCENTILE dei valori più vicini (assume oggetto > background)
    
    # Flatten dell'intera depth map per ogni sample
    flat_depth = depth_m[:, 0, :, :].reshape(B, -1)  # (B, H*W)
    
    # --- Filtro Background e Outlier (Vettorizzato con NaN) ---
    # Creiamo una maschera dei valori validi
    valid_mask = (flat_depth > 0.05) & (flat_depth < 4.0)
    
    # Sostituiamo i valori invalidi con NaN (Not a Number)
    depth_with_nans = flat_depth.clone()
    depth_with_nans[~valid_mask] = float('nan')
    
    # --- Calcolo Percentile Robusto (10% dei valori più vicini) ---
    # Strategia: L'oggetto è tipicamente la cosa più vicina nel crop.
    # Prendiamo il 10° percentile (ignora outlier come background lontano o pixel nulli)
    
    # Ordino i valori ignorando i NaN
    sorted_depths = torch.sort(depth_with_nans, dim=1).values  # (B, H*W)
    
    # Calcola l'indice del 10° percentile.
    # Conta quanti valori validi ci sono per ogni sample.
    valid_counts = torch.sum(~torch.isnan(depth_with_nans), dim=1)  # (B,)
    percentile_idx = (valid_counts * 0.10).long().clamp(min=0, max=H*W-1)  # 10° percentile
    
    # Estraggo il valore del percentile per ogni batch
    z_finals = sorted_depths[torch.arange(B), percentile_idx]
    
    # --- Fallback per righe completamente invalide ---
    invalid_batch_mask = torch.isnan(z_finals)
    
    if invalid_batch_mask.any():
        z_finals[invalid_batch_mask] = 0.5 

    z_final = z_finals.unsqueeze(1) # (B, 1)

    # --- Back-Projection ---
    # Mantengo dimensionalità (B, 1) per corretto broadcasting
    fx, fy = cam_k[:, 0:1], cam_k[:, 1:2]
    cx, cy = cam_k[:, 2:3], cam_k[:, 3:4]
    
    tx = (pred_uv[:, 0:1] - cx) * z_final / fx
    ty = (pred_uv[:, 1:2] - cy) * z_final / fy
    
    return torch.cat([tx, ty, z_final], dim=1)


def load_models_points(dataset_root, num_points=1000):
    """
    Carica i modelli 3D dal disco usando Farthest Point Sampling (FPS) 
    per una copertura geometrica ottimale.
    """
    cache = {}
    models_dir = dataset_root / "models"
    obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    
    print(f"⏳ Preloading 3D models with FPS (points={num_points}) from {models_dir}...")
    
    for obj_id in obj_ids:
        ply_path = models_dir / f"obj_{obj_id:02d}.ply"
        
        if ply_path.exists():
            mesh = trimesh.load(str(ply_path))
            vertices = np.array(mesh.vertices)
            
            # --- ALGORITMO FARTHEST POINT SAMPLING ---
            n_vertices = vertices.shape[0]
            if n_vertices > num_points:
                sampled_idxs = np.zeros(num_points, dtype=np.int32)
                # Scegliamo il primo punto a caso o il primo vertice
                sampled_idxs[0] = 0 
                # Distanze minime di ogni vertice dai punti già selezionati
                min_distances = np.full(n_vertices, np.inf)
                
                curr_point = vertices[sampled_idxs[0]]
                
                for i in range(1, num_points):
                    # Calcola distanza euclidea tra l'ultimo punto scelto e tutti gli altri
                    dist = np.linalg.norm(vertices - curr_point, axis=1)
                    # Aggiorna la distanza minima per ogni vertice
                    min_distances = np.minimum(min_distances, dist)
                    # Il prossimo punto è quello con la massima tra le distanze minime
                    sampled_idxs[i] = np.argmax(min_distances)
                    curr_point = vertices[sampled_idxs[i]]
                
                sampled_points = vertices[sampled_idxs]
            else:
                # Se l'oggetto ha meno punti del richiesto, prendiamo tutto
                sampled_points = vertices

            # LineMOD .ply sono in mm. Converto in metri
            tensor_points = torch.tensor(sampled_points, dtype=torch.float32) / 1000.0
            cache[obj_id] = tensor_points
        else:
            print(f"⚠️ Warning: Model {ply_path} not found.")
            
    print(f"✅ Loaded {len(cache)} models with FPS.")
    return cache


def compute_ADD(pred_R, gt_R, points, pred_t=None, gt_t=None):
    """ Calcola la metrica ADD (Average Distance of Model Points) """
    # Caso Rotation-Only
    if pred_t is None: pred_t = torch.zeros((pred_R.shape[0], 3, 1), device=pred_R.device)
    if gt_t is None: gt_t = torch.zeros((gt_R.shape[0], 3, 1), device=gt_R.device)

    # Trasposizione i punti per la moltiplicazione: (B, N, 3) -> (B, 3, N)
    points_t = points.transpose(1, 2)

    # Applicazione trasformazione: R * p + t
    # (B, 3, 3) x (B, 3, N) -> (B, 3, N) + (B, 3, 1) -> (B, 3, N)
    p_pred = torch.bmm(pred_R, points_t) + pred_t
    p_gt = torch.bmm(gt_R, points_t) + gt_t

    # Calcolo distanza Euclidea per ogni punto: Norm su dim=1 (x,y,z)
    dists = torch.norm(p_pred - p_gt, dim=1) # (B, N)

    # Media su tutti i punti dell'oggetto: (B,)
    return dists.mean(dim=1)


def compute_ADDS(pred_R, gt_R, points, pred_t=None, gt_t=None):
    """ Calcola la metrica ADD (Average Distance of Model Points) """
    if pred_t is None: pred_t = torch.zeros((pred_R.shape[0], 3, 1), device=pred_R.device)
    if gt_t is None:   gt_t = torch.zeros((gt_R.shape[0], 3, 1), device=gt_R.device)

    points_t = points.transpose(1, 2)

    p_pred = (torch.bmm(pred_R, points_t) + pred_t).transpose(1, 2) # (B, N, 3)
    p_gt = (torch.bmm(gt_R, points_t) + gt_t).transpose(1, 2)     # (B, N, 3)

    # Calcolo matrice distanze tutti-contro-tutti: (B, N, N)
    dist_matrix = torch.cdist(p_pred, p_gt, p=2)

    # Per ogni punto predetto, trova il punto GT più vicino (min su dim=2)
    min_dists, _ = torch.min(dist_matrix, dim=2) # (B, N)

    return min_dists.mean(dim=1)


def compute_rotation_error(pred_quat, gt_quat, class_ids, symmetry_lookup, model_points):
    """
    Calcola l'errore di rotazione medio in GRADI per l'intero batch,
    gestendo correttamente le simmetrie.
    """
    with torch.no_grad():
        B = pred_quat.shape[0]
        is_sym = symmetry_lookup[class_ids.long()] # Maschera Booleana (B,)
        errors = torch.zeros(B, device=pred_quat.device)
        
        if (~is_sym).any():
            # errore rotazionale calcolato come errore geodesico dei quaternioni
            p_q = F.normalize(pred_quat[~is_sym], p=2, dim=1)
            g_q = F.normalize(gt_quat[~is_sym], p=2, dim=1)
            dot = torch.abs(torch.sum(p_q * g_q, dim=1))
            dot = torch.clamp(dot, -1.0, 1.0)
            errors[~is_sym] = torch.rad2deg(2 * torch.acos(dot))
            
        if is_sym.any():
            # errore rotazionale calcolato coi modelli 3D (usa una logica simile alla ADDS)
            p_R = quaternion_to_rotation_matrix(pred_quat[is_sym])
            g_R = quaternion_to_rotation_matrix(gt_quat[is_sym])
            pts = model_points[class_ids[is_sym].long()] # (B_sym, N, 3)
            
            # Applichiamo rotazione ai punti (senza traslazione)
            pts_t = pts.transpose(1, 2)
            p_pts = torch.bmm(p_R, pts_t).permute(0, 2, 1) # (B_sym, N, 3)
            g_pts = torch.bmm(g_R, pts_t).permute(0, 2, 1) # (B_sym, N, 3)
            
            # Distanza minima punto-a-punto (Symmetry-safe)
            dist_matrix = torch.cdist(p_pts, g_pts, p=2) 
            min_dists, _ = torch.min(dist_matrix, dim=2) # (B_sym, N)
            mean_dist = torch.mean(min_dists, dim=1)     # (B_sym,)
            
            # Approssimazione Distanza -> Gradi
            errors[is_sym] = torch.rad2deg(2 * torch.asin(torch.clamp(mean_dist / 2.0, 0.0, 1.0)))
            
        return errors.mean()


def compute_translation_error(pred_t, gt_t):
    """Errore di translation in CM (già convertito per compatibilità con codice esistente)."""
    if pred_t.ndim == 3: pred_t = pred_t.squeeze(-1)
    if gt_t.ndim == 3:   gt_t = gt_t.squeeze(-1)

    # Calcolo vettorizzato (Batch)
    errors = torch.norm(pred_t - gt_t, dim=1)

    return errors.mean() * 100.0


def print_evaluation_results_table(metrics_per_class, save_table=False, table_path="evaluation_results.csv"):

    df = pd.DataFrame(metrics_per_class)
    
    # Sort by class_id (keep 'MEAN' at the end)
    df_all = df[df['class_id'] == 'MEAN']
    df_classes = df[df['class_id'] != 'MEAN'].sort_values('class_id')
    df = pd.concat([df_classes, df_all], ignore_index=True)
    
    df['Object Name'] = df['class_id'].map(LINEMOD_OBJECT_NAMES)
    df = df.drop(columns=['class_id'])
    
    # Converti solo ADD da metri a cm (trans_mean è già in cm da compute_translation_error)
    df['add_mean'] = df['add_mean'] * 100
    
    df = df.rename(
        columns={
            'object_name': 'Object Name',
            'num_samples': '#Samples',
            'rot_mean': 'Rotation Error (deg)',
            'trans_mean': 'Translation Error (cm)',
            'add_mean': 'ADD(-S) (cm)',
            'accuracy_10p': 'Accuracy @10% (%)'
        }
    )

    df = df[
        [
            'Object Name',
            '#Samples',
            'Rotation Error (deg)',
            'Translation Error (cm)',
            'ADD(-S) (cm)',
            'Accuracy @10% (%)'
        ]
    ]

    df = df.round(2)

    if save_table:
        df.to_csv(table_path, index=False)
        print(f"Saved CSV to {table_path}")
    return df
