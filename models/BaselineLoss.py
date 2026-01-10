import torch
import torch.nn as nn
from utils.pose_utils import (
    compute_ADD,
    compute_ADDS,
    quaternion_to_rotation_matrix,
    compute_rotation_error,
    SYMMETRIC_OBJECTS
)


class BaselineLoss(nn.Module):
    """
    Loss per 6D pose estimation con supporto per oggetti simmetrici.
    
    IMPORTANTE: ResNet predice SOLO quaternion (rotazione).
    La translation viene calcolata geometricamente da bbox + diametro.
    
    Loss functions:
    - Centered ADD/ADD-S: isola la rotazione sui punti centrati (come ExtensionLoss)
    """
    
    def __init__(self, model_points_dict):
        super(BaselineLoss, self).__init__()
        
        # bank for mesh points (needed for ADDSR)
        max_id = max(model_points_dict.keys())
        n_pts = list(model_points_dict.values())[0].shape[0]
        bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32)
        for oid, pts in model_points_dict.items():
            bank[oid] = pts
        self.register_buffer('model_points_bank', bank)
            
        # symmetry lookup table (True for symmetric objects)
        symmetry_mask = torch.zeros(max_id + 1, dtype=torch.bool)
        for obj_id in SYMMETRIC_OBJECTS:
            if obj_id <= max_id:
                symmetry_mask[obj_id] = True
        self.register_buffer('symmetry_lookup', symmetry_mask)
    
    
    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, class_ids=None):
        """
        Args:
            pred_quat: (B, 4) predicted quaternion
            pred_trans: (B, 3) predicted translation
            gt_quat: (B, 4) ground truth quaternion
            gt_trans: (B, 3) ground truth translation
            class_ids: (B,) tensor con class ID per ogni sample
            
        Returns:
            dict con total_loss, metriche
        """
        device = pred_quat.device
        
        # Indicizzazione su CPU, poi sposta su GPU
        batch_points = self.model_points_bank[class_ids.cpu().long()].to(device)  # (B, N, 3)
        pred_R = quaternion_to_rotation_matrix(pred_quat)  # (B, 3, 3)
        gt_R = quaternion_to_rotation_matrix(gt_quat)      # (B, 3, 3)
            
        # symmetry mask (indicizzazione su CPU, risultato già bool)
        is_symmetric = self.symmetry_lookup[class_ids.cpu().long()].to(device)  # (B,)
            
        loss_add = compute_ADD(pred_R, gt_R, batch_points)    # (B,) asymmetric
        loss_adds = compute_ADDS(pred_R, gt_R, batch_points)  # (B,) symmetric
            
        # Select correct loss per sample using boolean mask
        rot_loss = torch.where(is_symmetric, loss_adds, loss_add).mean()
        
        # Translation error SOLO per monitoraggio
        with torch.no_grad():
            trans_error_m = torch.mean(torch.norm(pred_trans - gt_trans, dim=1))
            trans_error_cm = trans_error_m * 100  # metri -> cm
            rot_err_deg = compute_rotation_error(
                pred_quat, 
                gt_quat, 
                class_ids.cpu(), 
                self.symmetry_lookup, 
                self.model_points_bank
            )
        
        return {
            'total_loss': rot_loss,
            'rotation_loss': rot_loss.item(),
            'translation_error_cm': trans_error_cm.item(),
            'rot_err_deg': rot_err_deg
        }
    