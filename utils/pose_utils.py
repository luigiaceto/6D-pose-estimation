import torch
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
import trimesh

IMG_WIDTH = 640
IMG_HEIGHT = 480

SYMMETRIC_OBJECTS = [10, 11]

# 1. Nomi per visualizzazione umana
LINEMOD_OBJECT_NAMES = {
    1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
    12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone",
    "MEAN": "MEAN"
}

# 2. Traduzione da YOLO (0,1,2...) a LINEMOD (1,2,4...)
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
    """Errore di rotazione in gradi."""
    R_diff = pred_R.T @ gt_R
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
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
            'accuracy_10p': 'Accuracy @10% (%)',
            'add_r_accuracy_10p': 'ADD-R Accuracy @10% (%)',
            'rot_mean': 'Rotation Error (deg)',
            'trans_mean': 'Translation Error (cm)',
            'add_mean': 'ADD / ADD-S (cm)',
            'add_rot_only_mean': 'ADD-R (cm)'
        }
    )

    df = df[
        [
            'Object Name',
            '#Samples',
            'Accuracy @10% (%)',
            'ADD-R Accuracy @10% (%)',
            'Rotation Error (deg)',
            'Translation Error (cm)',
            'ADD / ADD-S (cm)',
            'ADD-R (cm)'
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

def compute_batch_rotation_error(pred_quat, gt_quat):
    """
    Calcola l'errore medio di rotazione in gradi per un batch di quaternioni.
    Gestisce l'ambiguità q = -q e la stabilità numerica.
    
    Args:
        pred_quat: (B, 4) tensor
        gt_quat: (B, 4) tensor
        
    Returns:
        float: Errore medio in gradi
    """
    with torch.no_grad():
        # Assicuriamoci che siano normalizzati (sicurezza extra)
        pred_q = F.normalize(pred_quat, p=2, dim=1)
        gt_q = F.normalize(gt_quat, p=2, dim=1)
        
        # Dot product assoluto per gestire la doppia copertura (q e -q sono uguali)
        dot_prod = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        
        # Clamping per evitare NaN dovuti a errori numerici (es. 1.0000001)
        dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
        
        # Formula: theta = 2 * acos(|<q1, q2>|)
        angular_dist_rad = 2 * torch.acos(dot_prod)
        
        # Conversione in gradi e media sul batch
        return torch.rad2deg(angular_dist_rad).mean().item()
    