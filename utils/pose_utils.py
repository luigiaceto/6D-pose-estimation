import torch
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
import trimesh
import cv2

IMG_WIDTH = 640
IMG_HEIGHT = 480

SYMMETRIC_OBJECTS = [10, 11]

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

def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]


def solve_translation_geometric_high_precision(cropped_depth, pred_uv, cam_k, bbox_center, bbox_dims, z_net=None, use_bbox_center_only=True):
    """
    Versione ROBUSTA + CENTER MODE.
    use_bbox_center_only: Se True, ignora l'offset predetto dalla rete e assume che 
                          il centro dell'oggetto sia il centro geometrico del BBox.
    """
    B, _, H, W = cropped_depth.shape 
    IMG_W, IMG_H = 640.0, 480.0
    
    # 1. Calcolo Scala (Zoom Factor)
    w_px = bbox_dims[:, 0] * IMG_W
    h_px = bbox_dims[:, 1] * IMG_H
    max_dim = torch.max(w_px, h_px)
    scale_factor = W / torch.clamp(max_dim, min=1.0)
    
    # --- LOGICA DI PUNTAMENTO ---
    if use_bbox_center_only:
        # FEDERICO MODE: Ci fidiamo del BBox. 
        # Il centro dell'oggetto è il centro del crop (W/2, H/2).
        u_local = torch.full((B,), W/2, device=cropped_depth.device)
        v_local = torch.full((B,), H/2, device=cropped_depth.device)
        
        # Per calcolare X e Y finali, usiamo le coordinate globali del centro BBox
        pred_uv_final = bbox_center
    else:
        # OFFSET MODE (Tua Rete): Usiamo l'offset predetto
        delta_uv_global = pred_uv - bbox_center 
        delta_uv_crop = delta_uv_global * scale_factor.unsqueeze(1)
        u_local = delta_uv_crop[:, 0] + (W / 2)
        v_local = delta_uv_crop[:, 1] + (H / 2)
        pred_uv_final = pred_uv

    # 2. Campionamento Adattivo (Kernel 5x5)
    # Kernel ridotto per evitare di pescare lo sfondo su oggetti sottili
    k_size = 5 
    pad = k_size // 2
    
    u_center = torch.clamp(u_local.long(), pad, W - 1 - pad)
    v_center = torch.clamp(v_local.long(), pad, H - 1 - pad)
    
    # Creazione griglia di campionamento vettorizzata
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-pad, pad + 1, device=cropped_depth.device),
        torch.arange(-pad, pad + 1, device=cropped_depth.device),
        indexing='ij'
    )
    off_x = grid_x.flatten()
    off_y = grid_y.flatten()
    
    sample_u = torch.clamp(u_center.unsqueeze(1) + off_x.unsqueeze(0), 0, W - 1)
    sample_v = torch.clamp(v_center.unsqueeze(1) + off_y.unsqueeze(0), 0, H - 1)
    
    flat_indices = (sample_v * W + sample_u).long()
    z_patches = torch.gather(cropped_depth.view(B, -1), 1, flat_indices)
    
    # 3. Filtraggio Ibrido (Mediana Geometrica + Fallback Rete)
    sorted_z, _ = torch.sort(z_patches, dim=1)
    z_geom = sorted_z[:, z_patches.shape[1] // 2] # Mediana
    
    # Unità mm -> m
    z_geom = torch.where(z_geom > 100.0, z_geom / 1000.0, z_geom)
    
    # Fallback sulla rete se disponibile
    if z_net is not None:
        z_net = z_net.view(-1)
        z_net = torch.where(z_net > 100.0, z_net / 1000.0, z_net)
    else:
        z_net = torch.ones_like(z_geom) * 0.5 # Dummy fallback

    # Usa la geometria se il valore è sensato (tra 10cm e 5m), altrimenti usa la rete
    valid_mask = (z_geom > 0.1) & (z_geom < 5.0)
    z_final = torch.where(valid_mask, z_geom, z_net)
    
    # 4. Back-Projection (Pinhole)
    fx, fy, cx, cy = cam_k[:, 0], cam_k[:, 1], cam_k[:, 2], cam_k[:, 3]
    
    tx = (pred_uv_final[:, 0] - cx) * z_final / fx
    ty = (pred_uv_final[:, 1] - cy) * z_final / fy
    
    return torch.stack([tx, ty, z_final], dim=1)
    
    return torch.stack([tx, ty, z_meters], dim=1)

def solve_translation_direct_from_file(depth_paths, pred_coords, cam_k, obj_ids, object_diameters):
    """
    Legge la Z dal file PNG originale alle coordinate PREDETTE DALLA RETE.
    Applica la correzione Skin-to-Heart: il sensore misura la superficie,
    ma la GT è riferita al centroide. Aggiungiamo radius = diameter / 2.
    
    Args:
        depth_paths: list[str] paths ai file depth
        pred_coords: Tensor (B, 2) coordinate globali [u, v] predette dalla rete
        cam_k: Tensor (B, 4) intrinseci
        obj_ids: Tensor (B,) ID degli oggetti nel batch
        object_diameters: dict {obj_id: diameter_mm} diametri in mm
    
    Returns:
        (B, 3) Traslazione [Tx, Ty, Tz] in metri
    """
    device = pred_coords.device
    B = len(depth_paths)
    z_values = []
    
    # Portiamo su CPU per usare con cv2/numpy
    coords_np = pred_coords.detach().cpu().numpy()
    obj_ids_np = obj_ids.cpu().numpy()
    
    for i in range(B):
        path = depth_paths[i]
        obj_id = int(obj_ids_np[i])
        
        # Usiamo le coordinate predette dalla rete!
        cx, cy = int(coords_np[i, 0]), int(coords_np[i, 1])
        
        # 1. Carica Depth Originale (16-bit mm)
        # Usa cv2.IMREAD_UNCHANGED per mantenere i uint16
        depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            z_values.append(0.5)  # Fallback dummy
            continue
            
        img_h, img_w = depth_img.shape
        
        # 2. Robust Center Calc (clip to image bounds)
        cx = min(max(cx, 0), img_w - 1)
        cy = min(max(cy, 0), img_h - 1)
        
        # 3. Depth value at center (mm) - SINGLE PIXEL READ
        z_surface_mm = float(depth_img[cy, cx])
        
        # 4. CORREZIONE SKIN-TO-HEART
        # Il sensore misura la superficie, ma la GT è al centroide
        # Aggiungiamo il raggio (metà diametro) per compensare
        diameter_mm = object_diameters.get(obj_id, 0.0)
        radius_mm = diameter_mm / 2.0
        z_corrected_mm = z_surface_mm + radius_mm
        
        # 5. Converti mm -> metri
        z_meters = z_corrected_mm / 1000.0
        z_values.append(z_meters)
    
    z_tensor = torch.tensor(z_values, device=device, dtype=torch.float32)
    
    # Fallback: Se la rete ha puntato nel vuoto (z=0), purtroppo non possiamo farci nulla 
    # se non usare un valore medio o fidarci della z_head (se passata). 
    # Per ora usiamo un clamp minimo.
    z_tensor = torch.where(z_tensor > 0.01, z_tensor, torch.tensor(0.5, device=device))
    
    # 4. Back-Projection usando le coordinate PREDETTE
    fx, fy, cx_k, cy_k = cam_k[:, 0], cam_k[:, 1], cam_k[:, 2], cam_k[:, 3]
    tx = (pred_coords[:, 0] - cx_k) * z_tensor / fx
    ty = (pred_coords[:, 1] - cy_k) * z_tensor / fy
    
    return torch.stack([tx, ty, z_tensor], dim=1)

def solve_translation_geometric(cropped_depth, bbox_center, cam_k):
    """
    Calcola Tx, Ty, Tz usando Pinhole Inverse Projection dalla depth map.
    La Z viene letta direttamente dal centro della depth map (mediana 5x5 per robustezza).
    
    Args:
        cropped_depth: (B, 1, H, W) - Depth map croppata dall'oggetto
        bbox_center: (B, 2) - Centro bbox in coordinate pixel [u, v]
        cam_k: (B, 4) - Parametri intrinseci camera [fx, fy, cx, cy]
    
    Returns:
        (B, 3) - Traslazione [Tx, Ty, Tz] in metri
    """
    device = cropped_depth.device
    B, _, H, W = cropped_depth.shape
    cy, cx = H // 2, W // 2
    
    # Prendi mediana 5x5 al centro per robustezza contro outlier
    z_crop = cropped_depth[:, 0, cy-2:cy+3, cx-2:cx+3]
    tz = z_crop.reshape(B, -1).median(dim=1).values
    
    # Gestione unità: se depth è in mm (valori > 100), converti in metri
    mask_mm = (tz > 100.0)
    tz[mask_mm] = tz[mask_mm] / 1000.0
    
    # Clamp per sicurezza (evita z=0 che causerebbe divisione per zero)
    tz = torch.clamp(tz, min=0.1)
    
    # Back-projection: da pixel (u, v) + depth Z -> 3D (X, Y, Z)
    fx, fy, cx_cam, cy_cam = cam_k[:, 0], cam_k[:, 1], cam_k[:, 2], cam_k[:, 3]
    u, v = bbox_center[:, 0], bbox_center[:, 1]
    
    tx = (u - cx_cam) * tz / fx
    ty = (v - cy_cam) * tz / fy
    
    return torch.stack([tx, ty, tz], dim=1)


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
    
def compute_matrix_geodesic_loss(pred_quat, gt_quat):
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


def compute_add_metric(pred_R, pred_t, gt_R, gt_t, model_points):
    """
    ADD metric: Average Distance of Model Points.
    Standard per oggetti asimmetrici.
    
    Args:
        pred_R, gt_R: (3, 3) matrici di rotazione
        pred_t, gt_t: (3,) vettori di traslazione
        model_points: (N, 3) nuvola di punti dell'oggetto
    Returns:
        float: errore medio in metri
    """
    # Trasforma i punti modello nello spazio camera predetto e ground truth
    pred_points = (pred_R @ model_points.T).T + pred_t
    gt_points = (gt_R @ model_points.T).T + gt_t
    
    # Distanza euclidea punto-a-punto
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances)


def compute_add_s_metric(pred_R, pred_t, gt_R, gt_t, model_points):
    """
    ADD-S metric: Average Closest Point Distance.
    Standard per oggetti simmetrici.
    """
    pred_points = (pred_R @ model_points.T).T + pred_t
    gt_points = (gt_R @ model_points.T).T + gt_t
    
    # Calcola matrice distanze (N, N) tra tutti i punti predetti e veri
    # pred_points[:, None, :] -> (N, 1, 3) broadcast
    # gt_points[None, :, :]   -> (1, N, 3) broadcast
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    
    # Per ogni punto predetto, trova la distanza dal punto vero più vicino
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)


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


def load_model_points(dataset_root, obj_id):
    """Carica corner points 3D del modello."""
    models_info_path = str(dataset_root / "models" / "models_info.yml")
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.CLoader)
    
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners del bounding box
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32) / 1000.0  # mm -> m
    
    return corners


def compute_rotation_error(pred_R, gt_R):
    """Errore di rotazione in gradi (standard per oggetti asimmetrici)."""
    R_diff = pred_R.T @ gt_R
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_rotation_error_symmetric(pred_R, gt_R, points):
    """
    Errore di rotazione SYMMETRY-AWARE per oggetti simmetrici.
    
    Calcola l'errore minimo considerando tutte le possibili rotazioni simmetriche.
    Per oggetti come Eggbox (180° di simmetria), non penalizza rotazioni equivalenti.
    
    Usa ADD-S centrato: applica rotazioni ai punti centrati e trova matching minimo.
    
    Args:
        pred_R: (3, 3) matrice di rotazione predetta
        gt_R: (3, 3) matrice di rotazione ground truth
        points: (N, 3) punti del modello 3D
    
    Returns:
        float: errore di rotazione in gradi (minimo considerando simmetrie)
    """
    # Centra i punti nell'origine
    points_centered = points - points.mean(axis=0, keepdims=True)
    
    # Applica rotazioni
    pred_pts = (pred_R @ points_centered.T).T  # (N, 3)
    gt_pts = (gt_R @ points_centered.T).T      # (N, 3)
    
    # Calcola distanza minima punto-a-punto (ADD-S centrato)
    # Per ogni punto predetto, trova il punto GT più vicino
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(pred_pts, gt_pts, metric='euclidean')  # (N, N)
    min_dists = dist_matrix.min(axis=1)  # (N,)
    
    # Errore medio come proxy dell'errore angolare
    # Converti distanza euclidea in angolo approssimato
    mean_dist = min_dists.mean()
    
    # Per sfere unitarie: dist ≈ 2*sin(angle/2)
    # Inverti per ottenere angle ≈ 2*arcsin(dist/2)
    angle_rad = 2 * np.arcsin(np.clip(mean_dist / 2.0, 0.0, 1.0))
    
    return np.degrees(angle_rad)


def compute_translation_error(pred_t, gt_t):
    """Errore di translation in cm."""
    return np.linalg.norm(pred_t - gt_t) * 100  # m -> cm


def print_evaluation_results_table(metrics_per_class, save_table=False, table_path="evaluation_results.csv"):

    df = pd.DataFrame(metrics_per_class)
    
    # Sort by class_id (keep 'MEAN' at the end)
    df_all = df[df['class_id'] == 'MEAN']
    df_classes = df[df['class_id'] != 'MEAN'].sort_values('class_id')
    df = pd.concat([df_classes, df_all], ignore_index=True)
    
    df['Object Name'] = df['class_id'].map(LINEMOD_OBJECT_NAMES)
    df = df.drop(columns=['class_id'])
    df = df.rename(
        columns={
            'object_name': 'Object Name',
            'num_samples': '#Samples',
            'rot_mean': 'Rotation Error (deg)',
            'trans_mean': 'Translation Error (cm)',
            'z_mean': 'Z-Error (cm)',  # 🎯 NUOVA COLONNA
            'add_mean': 'ADD(-S) (cm)',
            'accuracy_10p': 'Accuracy @10% (%)'
        }
    )

    # Riordina colonne per mostrare Z-Error dopo Translation
    df = df[
        [
            'Object Name',
            '#Samples',
            'Rotation Error (deg)',
            'Translation Error (cm)',
            'Z-Error (cm)',  # 🎯 Profondita separata
            'ADD(-S) (cm)',
            'Accuracy @10% (%)'
        ]
    ]

    df = df.round(2)

    if save_table:
        df.to_csv(table_path, index=False)
        print(f"Saved CSV to {table_path}")
    return df

# USATE DALLA ADD-Loss
def batch_add_loss(pred_R, pred_t, gt_R, gt_t, points):
    """
    Calcola ADD loss (Asymmetric) per un batch.
    
    Args:
        pred_R, gt_R: (B, 3, 3)
        pred_t, gt_t: (B, 3, 1)
        points: (B, N, 3) - I punti del modello specifici per ogni oggetto nel batch
    """
    # Trasponiamo i punti per la moltiplicazione matriciale: (B, 3, N)
    points_t = points.transpose(1, 2)
    
    # Applicazione trasformazione: R * p + t
    # Broadcasting automatico di t su N punti
    pred_pts = torch.bmm(pred_R, points_t) + pred_t # (B, 3, N)
    gt_pts = torch.bmm(gt_R, points_t) + gt_t       # (B, 3, N)
    
    # Calcolo distanza Euclidea media per ogni oggetto nel batch
    # norm su dim=1 (x,y,z), mean su dim=2 (punti)
    dist = torch.norm(pred_pts - gt_pts, dim=1) # (B, N)
    return torch.mean(dist, dim=1) # (B,) Loss per ogni elemento del batch

def batch_adds_loss(pred_R, pred_t, gt_R, gt_t, points):
    """
    Calcola ADD-S loss (Symmetric) usando Nearest Neighbor.
    """
    points_t = points.transpose(1, 2)
    pred_pts = torch.bmm(pred_R, points_t) + pred_t # (B, 3, N)
    gt_pts = torch.bmm(gt_R, points_t) + gt_t       # (B, 3, N)
    
    # Preparazione per cdist: servono shape (B, N, 3)
    pred_pts = pred_pts.permute(0, 2, 1)
    gt_pts = gt_pts.permute(0, 2, 1)
    
    # Matrice distanze pairwise (B, N, N)
    # Calcola la distanza tra OGNI punto predetto e OGNI punto GT
    dist_matrix = torch.cdist(pred_pts, gt_pts, p=2) 
    
    # Per ogni punto predetto, troviamo il minimo nel GT (Nearest Neighbor)
    min_dists, _ = torch.min(dist_matrix, dim=2) # (B, N)
    
    return torch.mean(min_dists, dim=1) # (B,)

def load_all_models_points(dataset_root, num_points=1000):
    """
    Carica i modelli 3D dal disco, li campiona e li restituisce in un dizionario.
    Normalizza da millimetri a METRI se necessario.
    """
    cache = {}
    models_dir = dataset_root / "models"
    
    obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    
    print(f"⏳ Preloading 3D models from {models_dir}...")
    
    for obj_id in obj_ids:
        ply_path = models_dir / f"obj_{obj_id:02d}.ply"
        
        if ply_path.exists():
            mesh = trimesh.load(str(ply_path))
            vertices = np.array(mesh.vertices)
            
            # Farthest Point Sampling o Random
            if len(vertices) >= num_points:
                idxs = np.random.choice(len(vertices), num_points, replace=False)
            else:
                idxs = np.random.choice(len(vertices), num_points, replace=True)
            
            sampled_points = vertices[idxs]
            
            # LineMOD .ply sono in mm. Converto quindi in m
            tensor_points = torch.tensor(sampled_points, dtype=torch.float32) / 1000.0
            
            cache[obj_id] = tensor_points
        else:
            print(f"⚠️ Warning: Model {ply_path} not found.")
            
    print(f"✅ Loaded {len(cache)} models.")
    return cache

def compute_batch_rotation_error_all(pred_quat, gt_quat):
    """
    Calcola l'errore medio di rotazione in gradi per TUTTO il batch.
    NON filtra oggetti simmetrici - utile per logging e monitoraggio.
    
    Usa la formula: angular_distance = 2 * arccos(|q1 · q2|)
    L'abs gestisce la double cover dei quaternioni (q = -q).
    
    Args:
        pred_quat: (B, 4) quaternioni predetti
        gt_quat: (B, 4) quaternioni ground truth
    
    Returns:
        Tensor scalare: errore angolare medio in gradi
    """
    with torch.no_grad():
        # Normalizza quaternioni
        pred_q = F.normalize(pred_quat, p=2, dim=1, eps=1e-8)
        gt_q = F.normalize(gt_quat, p=2, dim=1, eps=1e-8)
        
        # Dot product assoluto per gestire q = -q
        dot_prod = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
        
        # Distanza angolare in radianti
        angular_dist_rad = 2 * torch.acos(dot_prod)
        
        # Converti in gradi e restituisci media
        return torch.rad2deg(angular_dist_rad).mean()

def compute_batch_rotation_error_asymm(pred_quat, gt_quat, class_ids, symmetry_lookup):
    """
    Calcola l'errore medio di rotazione in gradi solo per oggetti ASIMMETRICI.
    Filtra gli oggetti simmetrici usando symmetry_lookup.
    
    DEPRECATO: Usa compute_batch_rotation_error_all per logging completo.
    """
    
    is_sym = symmetry_lookup[class_ids.long()]
    mask = ~is_sym
    pred_quat = pred_quat[mask]
    gt_quat = gt_quat[mask]

    if pred_quat.numel() == 0:
        return torch.tensor(0.0, device=pred_quat.device)

    with torch.no_grad():
        pred_q = F.normalize(pred_quat, p=2, dim=1)
        gt_q = F.normalize(gt_quat, p=2, dim=1)
        
        # Dot product assoluto (q == -q)
        dot_prod = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
        
        angular_dist_rad = 2 * torch.acos(dot_prod)
        
        # Ritorniamo il tensore (NO .item())
        return torch.rad2deg(angular_dist_rad).mean()
    