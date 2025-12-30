import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNetPose import quaternion_to_rotation_matrix
from data.CustomDatasetPose import SYMMETRIC_OBJECTS

class PoseLoss(nn.Module):
    """
    Loss per 6D pose estimation con supporto per oggetti simmetrici.
    
    IMPORTANTE: ResNet predice SOLO quaternion (rotazione).
    La translation viene calcolata geometricamente da bbox + diametro.
    
    Loss functions:
    - Quaternion Geodesic: per oggetti standard
    - Rotation Matrix Geodesic: per oggetti simmetrici (teoricamente corretta!)
    - Hybrid Mode: Quaternion per standard, Rotation Matrix per simmetrici (BEST!)
    
    """
    
    def __init__(self, lambda_rotation=1.0, lambda_translation=0.0):
        super(PoseLoss, self).__init__()
        self.lambda_rotation = lambda_rotation
        self.lambda_translation = lambda_translation
        self.symmetric_objects = SYMMETRIC_OBJECTS
    
    def quaternion_angular_distance(self, q1, q2):
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
    
    def rotation_matrix_geodesic_loss(self, pred_quat, gt_quat):
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
    
    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, class_ids=None):
        """
        Args:
            pred_quat: (B, 4) predicted quaternion
            pred_trans: (B, 3) predicted translation
            gt_quat: (B, 4) ground truth quaternion
            gt_trans: (B, 3) ground truth translation
            class_ids: (B,) tensor con class ID per ogni sample (per hybrid mode)
            
        Returns:
            dict con total_loss, metriche
        """
        # Hybrid Mode: scegli loss in base all'oggetto
        if  class_ids is not None:
            batch_size = pred_quat.shape[0]
            rot_losses = []
            
            for i in range(batch_size):
                class_id = class_ids[i].item()
                
                # Oggetto simmetrico -> Rotation Matrix Geodesic
                if class_id in self.symmetric_objects:
                    loss = self.rotation_matrix_geodesic_loss(
                        pred_quat[i:i+1], gt_quat[i:i+1]
                    )
                # Oggetto standard -> Quaternion Geodesic
                else:
                    loss = self.quaternion_angular_distance(
                        pred_quat[i:i+1], gt_quat[i:i+1]
                    )
                rot_losses.append(loss)
            
            rot_loss = torch.mean(torch.stack(rot_losses))
        
        # Translation error: SOLO per monitoraggio, NO backprop
        with torch.no_grad():
            trans_error_m = torch.mean(torch.norm(pred_trans - gt_trans, dim=1))
            trans_error_cm = trans_error_m * 100  # metri -> cm
        
        # Total loss: SOLO rotazione (translation è geometrica)
        total_loss = self.lambda_rotation * rot_loss
        
        return {
            'total_loss': total_loss,
            'rotation_loss': rot_loss.item(),
            'translation_error_cm': trans_error_cm.item()
        }


def compute_add_metric(pred_R, pred_t, gt_R, gt_t, model_points):
    """
    ADD metric: Average Distance of Model Points.
    
    Args:
        pred_R: (3, 3) numpy array
        pred_t: (3,) numpy array
        gt_R: (3, 3) numpy array
        gt_t: (3,) numpy array
        model_points: (N, 3) numpy array - punti 3D del modello
        
    Returns:
        float: ADD in metri
    """
    import numpy as np
    
    pred_points = (pred_R @ model_points.T).T + pred_t
    gt_points = (gt_R @ model_points.T).T + gt_t
    
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances)


def compute_add_s_metric(pred_R, pred_t, gt_R, gt_t, model_points):
    """
    ADD-S metric per oggetti simmetrici.
    
    Args:
        Same as compute_add_metric
        
    Returns:
        float: ADD-S in metri
    """
    import numpy as np
    
    pred_points = (pred_R @ model_points.T).T + pred_t
    gt_points = (gt_R @ model_points.T).T + gt_t
    
    # Distanza minima per ogni punto predetto
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)


def compute_add_rotation_only(pred_R, gt_R, model_points):
    """
    ADD metric SOLO sulla rotazione (ADD-R).

    Args:
        pred_R: (3, 3) numpy array
        gt_R: (3, 3) numpy array
        model_points: (N, 3) numpy array

    Returns:
        float: ADD-R in metri
    """
    import numpy as np

    pred_points = (pred_R @ model_points.T).T
    gt_points = (gt_R @ model_points.T).T

    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances)


def compute_add_s_rotation_only(pred_R,  gt_R, model_points):
    """
    ADD-S metric per oggetti simmetrici.
    
    Args:
        Same as compute_add_metric
        
    Returns:
        float: ADD-S in metri
    """
    import numpy as np
    
    pred_points = (pred_R @ model_points.T).T 
    gt_points = (gt_R @ model_points.T).T 
    
    # Distanza minima per ogni punto predetto
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)