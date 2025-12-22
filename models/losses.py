import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    """
    Loss per 6D pose estimation con supporto per oggetti simmetrici.
    
    IMPORTANTE: ResNet predice SOLO quaternion (rotazione).
    La translation viene calcolata geometricamente da bbox + diametro.
    
    Per oggetti simmetrici, usa ADD-based loss invece di quaternion distance.
    """
    
    def __init__(self, lambda_rotation=1.0, lambda_translation=0.0, use_add_loss=False, model_points=None):
        super(PoseLoss, self).__init__()
        self.lambda_rotation = lambda_rotation
        self.lambda_translation = lambda_translation
        self.use_add_loss = use_add_loss  # True per oggetti simmetrici
        self.model_points = model_points  # Punti 3D del modello (per ADD loss)
    
    def quaternion_angular_distance(self, q1, q2):
        """
        Geodesic Distance tra quaternions (gestisce ambiguità q = -q).
        Loss: 1 - |q1 · q2|
        
        Usa torch.abs per gestire double cover (q e -q sono la stessa rotazione).
        """
        # Normalize con epsilon per stabilità numerica
        q1 = F.normalize(q1, p=2, dim=1, eps=1e-8)
        q2 = F.normalize(q2, p=2, dim=1, eps=1e-8)
        
        # Dot product con abs per gestire q = -q
        dot = torch.abs(torch.sum(q1 * q2, dim=1))
        dot = torch.clamp(dot, 0.0, 1.0)
        
        # Geodesic distance: 1 - |<q1, q2>|
        return torch.mean(1.0 - dot)
    
    def add_loss(self, pred_quat, pred_trans, gt_quat, gt_trans, model_points):
        """
        ADD-based loss per oggetti simmetrici.
        Calcola point-to-point distance tra modello trasformato.
        
        Args:
            pred_quat: (B, 4) predicted quaternion
            pred_trans: (B, 3) predicted translation
            gt_quat: (B, 4) ground truth quaternion
            gt_trans: (B, 3) ground truth translation
            model_points: (N, 3) punti 3D del modello
        
        Returns:
            ADD loss in metri
        """
        from models.ResNetPose import quaternion_to_rotation_matrix
        
        # Converti quaternioni a matrici di rotazione
        pred_R = quaternion_to_rotation_matrix(pred_quat)  # (B, 3, 3)
        gt_R = quaternion_to_rotation_matrix(gt_quat)      # (B, 3, 3)
        
        # Trasforma punti del modello: R @ p + t
        # model_points: (N, 3) → (1, N, 3)
        points = model_points.unsqueeze(0)  # (1, N, 3)
        
        # pred_R: (B, 3, 3), points.T: (1, 3, N) → (B, 3, N)
        pred_points = torch.bmm(pred_R, points.transpose(1, 2))  # (B, 3, N)
        pred_points = pred_points.transpose(1, 2)  # (B, N, 3)
        pred_points = pred_points + pred_trans.unsqueeze(1)  # (B, N, 3)
        
        gt_points = torch.bmm(gt_R, points.transpose(1, 2))
        gt_points = gt_points.transpose(1, 2)
        gt_points = gt_points + gt_trans.unsqueeze(1)
        
        # Distanza point-to-point
        distances = torch.norm(pred_points - gt_points, dim=2)  # (B, N)
        add_error = torch.mean(distances)
        
        return add_error
    
    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, model_points=None):
        """
        Args:
            pred_quat: (B, 4) predicted quaternion (DA RESNET - HA GRADIENTI)
            pred_trans: (B, 3) predicted translation (CALCOLATA GEOMETRICAMENTE - NO GRADIENTI)
            gt_quat: (B, 4) ground truth quaternion
            gt_trans: (B, 3) ground truth translation
            model_points: (N, 3) punti 3D modello (opzionale, per ADD loss)
            
        Returns:
            dict con total_loss, metriche
        """
        # Scegli loss in base al tipo di oggetto
        if self.use_add_loss and model_points is not None:
            # ADD loss per oggetti simmetrici (point-to-point distance)
            rot_loss = self.add_loss(pred_quat, pred_trans, gt_quat, gt_trans, model_points)
        else:
            # Quaternion geodesic loss per oggetti non simmetrici
            rot_loss = self.quaternion_angular_distance(pred_quat, gt_quat)
        
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
