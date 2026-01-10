import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import trimesh

IMG_WIDTH = 640
IMG_HEIGHT = 480

SYMMETRIC_OBJECTS = [10, 11]
N_POINTS_TO_LOAD = 2000

# Traduzione da YOLO (0,1,2...) a LINEMOD (1,2,4...)
YOLO_TO_LINEMOD_MAP = {
    0: 1,  
    1: 2,  
    2: 4,  
    3: 5,  
    4: 6,  
    5: 8,  
    6: 9,  
    7: 10, 
    8: 11, 
    9: 12, 
    10: 13,
    11: 14,
    12: 15 
}

LINEMOD_OBJECT_NAMES = {
    1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
    12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone",
    "MEAN": "MEAN"
}

# ------------------ FUNZIONI DI UTILS GENERICHE -------------------
#region
def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]


def compute_translation_from_depth_crop(cropped_depth, pred_uv, cam_k):
    """
    Calcola la coordinata Z (depth) usando un ROBUSTO PERCENTILE vettorizzato.
    
    Bug #2 Fix: Invece di campionare solo il centro fisso (che fallisce con jitter),
    usa il 10° percentile dell'intero crop valido (assume l'oggetto sia la cosa più vicina).
    
    IMPORTANTE: Assume che cropped_depth sia GIÀ IN METRI (garantito dal dataset).
    """
    B, _, H, W = cropped_depth.shape 
    
    # --- 1. Nessuna conversione unità - Dataset garantisce metri ---
    # Bug #3 Fix: Rimossa euristica fragile (if median > 10 -> divide)
    # Il dataset DEVE fornire depth in metri, altrimenti correggi il dataset.
    depth_m = cropped_depth.clone().detach()
    
    # --- 2. Sampling Robusto su TUTTO il Crop (non solo centro) ---
    # Bug #2 Fix: Con il jitter del bbox, l'oggetto può essere decentrato.
    # Soluzione: Prendi il PERCENTILE dei valori più vicini (assume oggetto > background)
    
    # Flatten dell'intera depth map per ogni sample
    flat_depth = depth_m[:, 0, :, :].reshape(B, -1)  # (B, H*W)
    
    # --- 3. Filtro Background e Outlier (Vettorizzato con NaN) ---
    # Creiamo una maschera dei valori validi
    valid_mask = (flat_depth > 0.05) & (flat_depth < 4.0)
    
    # Sostituiamo i valori invalidi con NaN (Not a Number)
    depth_with_nans = flat_depth.clone()
    depth_with_nans[~valid_mask] = float('nan')
    
    # --- 4. Calcolo Percentile Robusto (10% dei valori più vicini) ---
    # Strategia: L'oggetto è tipicamente la cosa più vicina nel crop.
    # Prendiamo il 10° percentile (ignora outlier come background lontano o pixel nulli)
    
    # Ordina i valori ignorando i NaN
    sorted_depths = torch.sort(depth_with_nans, dim=1).values  # (B, H*W)
    
    # Calcola l'indice del 10° percentile
    # Conta quanti valori validi ci sono per ogni sample
    valid_counts = torch.sum(~torch.isnan(depth_with_nans), dim=1)  # (B,)
    percentile_idx = (valid_counts * 0.10).long().clamp(min=0, max=H*W-1)  # 10° percentile
    
    # Estrai il valore del percentile per ogni batch
    z_finals = sorted_depths[torch.arange(B), percentile_idx]
    
    # --- 5. Fallback per righe completamente invalide ---
    invalid_batch_mask = torch.isnan(z_finals)
    
    if invalid_batch_mask.any():
        z_finals[invalid_batch_mask] = 0.5 

    z_final = z_finals.unsqueeze(1) # (B, 1)

    # --- 6. Back-Projection ---
    fx, fy = cam_k[:, 0], cam_k[:, 1]
    cx, cy = cam_k[:, 2], cam_k[:, 3]
    
    tx = (pred_uv[:, 0] - cx) * z_final / fx
    ty = (pred_uv[:, 1] - cy) * z_final / fy
    
    return torch.stack([tx, ty, z_final.squeeze(1)], dim=1)



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
                # Inizializzazione
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

#endregion


# --------------------- FUNZIONI UTILS PER IL TRAINING --------------

def compute_quaternion_loss(q1, q2):
        """
        Geodesic Distance tra quaternions (gestisce ambiguità q = -q).
        Loss: 1 - |q1 · q2|
        
        Usa torch.abs per gestire double cover (q e -q sono la stessa rotazione).
        """
        # Normalize con epsilon sicuro
        q1 = F.normalize(q1, p=2, dim=1, eps=1e-6)
        q2 = F.normalize(q2, p=2, dim=1, eps=1e-6)
        # Dot product con abs per gestire q = -q
        dot = torch.abs(torch.sum(q1 * q2, dim=1))
        # Clamp per sicurezza (non dovrebbe servire con abs)
        dot = torch.clamp(dot, 0.0, 1.0)
        return torch.mean(1.0 - dot)
    
def compute_geodesic_loss(pred_quat, gt_quat):
    """
    Geodesic Distance su SO(3) manifold usando rotation matrices.
    
    Questa è LA loss corretta per oggetti simmetrici:
    - Nessuna ambiguità (vs quaternion: q = -q)
    - Distanza nativa sul gruppo delle rotazioni SO(3)
    - Smooth e differenziabile per gradient descent
    - Gestisce simmetrie naturalmente
    
    Formula: arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    
    IMPORTANTE: Loss normalizzata in [0, 1] dividendo per π per compatibilità
    con Quaternion Loss nell'Hybrid mode.
    
    Questa loss è usata in SOTA papers (PoseCNN, DenseFusion, PVNet).
    """
    
    # Converti quaternions a rotation matrices
    pred_R = quaternion_to_rotation_matrix(pred_quat)  # (B, 3, 3)
    gt_R = quaternion_to_rotation_matrix(gt_quat)      # (B, 3, 3)
    
    # Calcola R_diff = R_pred^T @ R_gt
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)  # (B, 3, 3)
    
    # Trace di R_diff
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # (B,)
    
    cos_angle = (trace - 1.0) / 2.0
    
    # Clamp aggressivo per evitare sqrt di negativi
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    sin_half = torch.sqrt((1.0 - cos_angle) / 2.0 + 1e-7)
    
    return torch.mean(sin_half)

def compute_batch_rotation_error(pred_quat, gt_quat, class_ids, symmetry_lookup, model_points):
    """
    Calcola l'errore di rotazione medio in GRADI per l'intero batch,
    gestendo correttamente le simmetrie per evitare falsi positivi.
    """
    with torch.no_grad():
        B = pred_quat.shape[0]
        is_sym = symmetry_lookup[class_ids.long()] # Maschera Booleana (B,)
        errors = torch.zeros(B, device=pred_quat.device)
        
        # --- 1. ASIMMETRICI: Calcolo Geodetico Standard ---
        if (~is_sym).any():
            p_q = F.normalize(pred_quat[~is_sym], p=2, dim=1)
            g_q = F.normalize(gt_quat[~is_sym], p=2, dim=1)
            dot = torch.abs(torch.sum(p_q * g_q, dim=1))
            dot = torch.clamp(dot, -1.0, 1.0)
            errors[~is_sym] = torch.rad2deg(2 * torch.acos(dot))
            
        # --- 2. SIMMETRICI: Calcolo Symmetry-Aware (ADD-S) ---
        if is_sym.any():
            # Convertiamo in matrici per trasformare i punti
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

def compute_add_rot_loss(pred_R, gt_R, points):
    """
    Versione PyTorch Batch per il training.
    pred_R, gt_R: (B, 3, 3)
    points: (B, N, 3)
    """
    # Trasponiamo i punti per farli diventare (B, 3, N)
    # serve per fare (3x3) @ (3xN)
    points_t = points.transpose(1, 2)
    
    # Applica rotazione: (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    p_pred = torch.bmm(pred_R, points_t)
    p_gt = torch.bmm(gt_R, points_t)
    
    # Calcola distanza: norm su coordinate (dim 1), mean su punti (dim 2)
    dist = torch.norm(p_pred - p_gt, dim=1) 
    return dist.mean(dim=1) # Ritorna (B,)

def compute_adds_rot_loss(pred_R, gt_R, points):
    """
    Versione PyTorch Batch per simmetrici.
    """
    points_t = points.transpose(1, 2)
    
    # 1. Ruota i punti
    p_pred = torch.bmm(pred_R, points_t).transpose(1, 2) # Torna (B, N, 3)
    p_gt = torch.bmm(gt_R, points_t).transpose(1, 2)    # Torna (B, N, 3)
    
    # 2. Distanza punto-al-più-vicino (Nearest Neighbor)
    # torch.cdist fa esattamente quello che facevi tu con [:, None, :] ma è 100x più veloce
    dist_matrix = torch.cdist(p_pred, p_gt, p=2) # (B, N, N)
    
    # Per ogni punto predetto, prendi la distanza minima dal GT
    min_dist, _ = torch.min(dist_matrix, dim=2) # (B, N)
    
    return min_dist.mean(dim=1) # Ritorna (B,)



#----------- FUNZIONI PER EVALUATON  -----------

def batch_compute_add_metric(pred_R, pred_t, gt_R, gt_t, points):
    """
    Calcola ADD (Asimmetrico) per tutto il batch in PyTorch.
    pred_t, gt_t devono essere (B, 3, 1)
    """
    points_t = points.transpose(1, 2) # (B, 3, N)
    p_pred = torch.bmm(pred_R, points_t) + pred_t
    p_gt = torch.bmm(gt_R, points_t) + gt_t
    
    # Distanza punto-punto: mean su N punti (dim 2)
    return torch.norm(p_pred - p_gt, dim=1).mean(dim=1) # Ritorna (B,)


def batch_compute_add_s_metric(pred_R, pred_t, gt_R, gt_t, points):
    """
    Calcola ADD-S (Simmetrico) per tutto il batch in PyTorch.
    """
    points_t = points.transpose(1, 2)
    p_pred = (torch.bmm(pred_R, points_t) + pred_t).transpose(1, 2) # (B, N, 3)
    p_gt = (torch.bmm(gt_R, points_t) + gt_t).transpose(1, 2)       # (B, N, 3)
    
    # Distanza punto-al-più-vicino
    dist_matrix = torch.cdist(p_pred, p_gt, p=2) 
    min_dist, _ = torch.min(dist_matrix, dim=2)
    return min_dist.mean(dim=1) # Ritorna (B,)


def compute_add_rotation_only(pred_R, gt_R, model_points):
    """
    ADD metric SOLO sulla rotazione (ADD-R).
    Utile per capire se l'errore viene dalla rotazione ignorando la traslazione.
    """
    # Applica solo la rotazione (senza traslazione)
    pred_points = (pred_R @ model_points.T).T
    gt_points = (gt_R @ model_points.T).T

    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances)


def compute_add_s_rotation_only(pred_R, gt_R, model_points):
    """
    ADD-S metric SOLO sulla rotazione per simmetrici.
    """
    pred_points = (pred_R @ model_points.T).T 
    gt_points = (gt_R @ model_points.T).T 
    
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)


def compute_rotation_error(pred_R, gt_R, class_id, model_points):
    """
    Funzione UNIFICATA (NumPy) per l'Evaluation Loop.
    Sostituisce sia compute_rotation_error che compute_rotation_error_symmetric.
    
    Logica:
    1. Controlla se class_id è nella lista globale SYMMETRIC_OBJECTS.
    2. Se SÌ -> Usa calcolo geometrico (ADD-S logic) sui punti.
    3. Se NO -> Usa calcolo algebrico standard.
    """
    
    # 1. Controllo Simmetria (Usa la lista globale definita in cima al file)
    is_symmetric = (class_id in SYMMETRIC_OBJECTS)

    # --- CASO SIMMETRICO (Eggbox, Glue) ---
    if is_symmetric:
        
        mean_dist = compute_add_s_rotation_only(pred_R, gt_R, model_points)
        avg_radius = np.mean(np.linalg.norm(model_points, axis=1))

        ratio = mean_dist / avg_radius
        angle_rad = 2 * np.arcsin(np.clip(ratio / 2.0, -1.0, 1.0))

        return np.degrees(angle_rad)
    

    # --- CASO ASIMMETRICO (Ape, Cat, ecc.) ---
    else:
        # Calcolo standard Geodetico
        R_diff = pred_R.T @ gt_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))


def compute_translation_error(pred_t, gt_t):
    """Errore di translation in CM (già convertito per compatibilità con codice esistente)."""
    return np.linalg.norm(pred_t - gt_t) * 100  # metri -> cm


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


