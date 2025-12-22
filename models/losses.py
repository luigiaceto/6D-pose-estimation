import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    """
    Loss per 6D pose estimation.
    
    IMPORTANTE: ResNet predice SOLO quaternion (rotazione).
    La translation viene calcolata geometricamente da bbox + diametro.
    
    Quindi la loss ottimizza SOLO il quaternion!
    La translation viene usata solo per metriche di monitoraggio.
    """
    
    def __init__(self, lambda_rotation=1.0, lambda_translation=0.0):
        super(PoseLoss, self).__init__()
        self.lambda_rotation = lambda_rotation
        self.lambda_translation = lambda_translation  # Sempre 0! Translation non ha gradienti
    
    def quaternion_angular_distance(self, q1, q2):
        """
        Angular distance tra quaternions.
        Loss: 1 - |q1 · q2|
        
        Più il dot product è vicino a 1, più i quaternion sono simili.
        """
        # Normalize
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)
        
        # Dot product
        dot = torch.abs(torch.sum(q1 * q2, dim=1))
        dot = torch.clamp(dot, 0, 1)
        
        return torch.mean(1.0 - dot)
    
    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans):
        """
        Args:
            pred_quat: (B, 4) predicted quaternion (DA RESNET - HA GRADIENTI)
            pred_trans: (B, 3) predicted translation (CALCOLATA GEOMETRICAMENTE - NO GRADIENTI)
            gt_quat: (B, 4) ground truth quaternion
            gt_trans: (B, 3) ground truth translation
            
        Returns:
            dict con total_loss (solo rotation), metriche
        """
        # Loss SOLO su rotazione (questa backpropaga a ResNet)
        rot_loss = self.quaternion_angular_distance(pred_quat, gt_quat)
        
        # Translation error: SOLO per monitoraggio, NO backprop
        with torch.no_grad():
            trans_error_m = torch.mean(torch.norm(pred_trans - gt_trans, dim=1))
            trans_error_cm = trans_error_m * 100  # metri -> cm
        
        # Total loss: SOLO rotazione!
        total_loss = self.lambda_rotation * rot_loss
        # NON aggiungiamo translation perché non c'è da ottimizzare
        
        return {
            'total_loss': total_loss,
            'rotation_loss': rot_loss.item(),
            'translation_error_cm': trans_error_cm.item()  # Solo metrica
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