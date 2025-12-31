import torch
import torch.nn as nn
from utils.pose_utils import compute_matrix_geodesic_loss, compute_quaternion_loss,  SYMMETRIC_OBJECTS

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
    
    
    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, class_ids=None):
        """
        Args:
            pred_quat: (B, 4) predicted quaternion
            pred_trans: (B, 3) predicted translation
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
                if class_id in SYMMETRIC_OBJECTS:
                    loss = compute_matrix_geodesic_loss(
                        pred_quat[i:i+1], gt_quat[i:i+1]
                    )
                # Oggetto standard -> Quaternion Geodesic
                else:
                    loss = compute_quaternion_loss(
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