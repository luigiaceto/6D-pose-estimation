import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PinholeCamera import PinholeCamera
from utils.pose_utils import (
    compute_add_rot_loss,
    compute_adds_rot_loss,
    quaternion_to_rotation_matrix,
    compute_batch_rotation_error,
    SYMMETRIC_OBJECTS
)

class ExtensionLoss(nn.Module):
    """
    Loss GEOMETRICA 3D + 2D per 6D Pose Estimation.
    
    Componenti:
    - L_rot: Centered ADD/ADD-S (isola rotazione sui punti centrati)
    - L_trans: Pure Translation L1 (distanza euclidea in metri)
    - L_proj: 2D Projection (regolarizzazione, guida ottimizzazione)
    
    Total Loss = λ_rot * L_rot + λ_trans * L_trans + λ_proj * L_proj
    
    NOTA: La ricostruzione 3D (back-projection) è ora nel modello.
    """
    def __init__(self, rot_weight, trans_weight, proj_weight, cam_k, model_points_dict):
        super().__init__()
        
        # Initialize PinholeCamera for 3D↔2D conversions
        self.pinhole = PinholeCamera(cam_k)
        
        # Build model points bank (object_id → 3D points)
        max_id = max(model_points_dict.keys())
        n_pts = list(model_points_dict.values())[0].shape[0]
        bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32)
        for oid, pts in model_points_dict.items():
            bank[oid] = pts
        self.register_buffer('model_points_bank', bank)
    
        # Build symmetry lookup table (True for symmetric objects)
        symmetry_mask = torch.zeros(max_id + 1, dtype=torch.bool)
        for obj_id in SYMMETRIC_OBJECTS:
            if obj_id <= max_id:
                symmetry_mask[obj_id] = True
        self.register_buffer('symmetry_lookup', symmetry_mask)

        # Loss weights
        self.w_rot = rot_weight
        self.w_trans = trans_weight
        self.w_proj = proj_weight

    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, class_ids):
        """
        Loss geometrica 3D + 2D: confronta rotation, translation 3D e proiezione 2D.
        
        Args:
            pred_quat: (B, 4) - Predicted quaternion (normalizzato)
            pred_trans: (B, 3) - Predicted translation in metri [x, y, z]
            gt_quat: (B, 4) - Ground truth quaternion
            gt_trans: (B, 3) - Ground truth translation in metri [x, y, z]
            class_ids: (B,) - Object class IDs
        
        Returns:
            dict: {
                'total_loss': weighted sum,
                'rot_loss': centered ADD/ADD-S,
                'trans_loss': translation L1,
                'proj_loss': 2D projection error,
                'trans_err_cm': translation error in cm,
                'proj_err_px': projection error in pixels,
                'rot_err_deg': rotation error in degrees
            }
        """
        device = pred_quat.device
        
        # STEP 1: Compute 2D projections 
        # Clamp depth per evitare divisioni per zero
        pred_trans_safe = pred_trans.clone()
        pred_trans_safe[:, 2] = torch.clamp(pred_trans[:, 2], min=0.01)
        
        gt_trans_safe = gt_trans.clone()
        gt_trans_safe[:, 2] = torch.clamp(gt_trans[:, 2], min=0.01)
        
        # Proiezione 3D → 2D
        pred_2d = self.pinhole.project_3d_to_2d(pred_trans_safe)  # (B, 2)
        gt_2d = self.pinhole.project_3d_to_2d(gt_trans_safe)      # (B, 2)
        
        # STEP 2: LOSS COMPUTATION
        
        # --- L_rot: Centered ADD/ADD-S (rotation-only, optimized) ---
        batch_points = self.model_points_bank[class_ids.long()]  # (B, N, 3)
        pred_R = quaternion_to_rotation_matrix(pred_quat)  # (B, 3, 3)
        gt_R = quaternion_to_rotation_matrix(gt_quat)      # (B, 3, 3)
        
        is_symmetric = self.symmetry_lookup[class_ids.long()]  # (B,)
        loss_rot_values = torch.zeros(len(class_ids), device=device)
        
        # Compute ADD for asymmetric objects only
        if (~is_symmetric).any():
            asym_mask = ~is_symmetric
            loss_add = compute_add_rot_loss(
                pred_R[asym_mask], 
                gt_R[asym_mask], 
                batch_points[asym_mask]
            )
            loss_rot_values[asym_mask] = loss_add
        
        # Compute ADD-S for symmetric objects only
        if is_symmetric.any():
            sym_mask = is_symmetric
            loss_adds = compute_adds_rot_loss(
                pred_R[sym_mask], 
                gt_R[sym_mask], 
                batch_points[sym_mask]
            )
            loss_rot_values[sym_mask] = loss_adds
        
        loss_rot = loss_rot_values.mean()
        
        # --- L_trans: Pure Translation L1 (in METERS) ---
        loss_trans = F.l1_loss(pred_trans, gt_trans)
        
        # --- L_proj: 2D Projection (regolarizzazione) ---
        loss_proj = F.smooth_l1_loss(pred_2d, gt_2d, beta=1.0)

        # TOTAL LOSS
        total_loss = (
            self.w_rot * loss_rot + 
            self.w_trans * loss_trans + 
            self.w_proj * loss_proj
        )

        # METRICS FOR LOGGING
        with torch.no_grad():
            trans_err_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            proj_err_px = torch.norm(pred_2d - gt_2d, p=2, dim=1).mean()
            rot_err_deg = compute_batch_rotation_error(
                pred_quat, gt_quat, class_ids, 
                self.symmetry_lookup, self.model_points_bank
            )

        return {
            'total_loss': total_loss,
            'rot_loss': loss_rot.detach(),
            'trans_loss': loss_trans.detach(),
            'proj_loss': loss_proj.detach(),
            'trans_err_cm': trans_err_cm,
            'proj_err_px': proj_err_px,
            'rot_err_deg': rot_err_deg
        }