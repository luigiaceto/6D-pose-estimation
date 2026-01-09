import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pose_utils import (
    compute_add_rotation_only,
    compute_add_s_rotation_only,
    quaternion_to_rotation_matrix,
    compute_batch_rotation_error_all,
    SYMMETRIC_OBJECTS
)

class ExtensionLoss(nn.Module):
    """
    Loss PURAMENTE GEOMETRICA per 6D Pose Estimation.
    ELIMINA completamente Geodesic Loss e Quaternion Loss algebriche.
    
    Componenti:
    - L_rot: Centered ADD/ADD-S (isola rotazione sui punti centrati, traslazione = 0)
    - L_trans: Pure Translation L1 (distanza euclidea sui vettori [x,y,z])
    - L_proj: 2D Projection (distanza pixel, opzionale per regolarizzazione)
    
    Total Loss = λ_rot * L_rot + λ_trans * L_trans + λ_proj * L_proj
    """
    def __init__(self, rot_weight, trans_weight, proj_weight, cam_k, model_points_dict):
        super().__init__()

        # Register camera intrinsics [fx, fy, cx, cy]
        self.register_buffer(
            'cam_k',
            torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]]).view(1, 4)
        )
        
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

        # Loss weights (can be updated dynamically during training)
        self.w_rot = rot_weight
        self.w_trans = trans_weight
        self.w_proj = proj_weight

    def forward(self, pred_quat, pred_delta_z, gt_quat, gt_trans, pred_2d, class_ids, z_geometric):
        """
        Args:
            pred_quat: (B, 4) - Predicted quaternion
            pred_delta_z: (B, 1) - Predicted depth refinement
            gt_quat: (B, 4) - Ground truth quaternion
            gt_trans: (B, 3) - Ground truth translation [x, y, z]
            pred_2d: (B, 2) - Predicted 2D pixel offset [u, v]
            class_ids: (B,) - Object class IDs for each sample
            z_geometric: (B, 1) - Geometric depth estimate (from solve_translation)
        
        Returns:
            dict: {
                'total_loss': weighted sum of all losses,
                'rot_loss': centered ADD/ADD-S (detached for logging),
                'trans_loss': pure translation L1 (detached),
                'proj_loss': 2D projection (detached),
                'trans_err_cm': translation error in cm,
                'proj_err_px': projection error in pixels,
                'rot_err_asymm_deg': rotation error in degrees
            }
        """
        device = pred_quat.device
        
        # ============================================
        # STEP 1: Reconstruct 3D translation from 2D + depth
        # ============================================
        z_final = z_geometric.detach() + pred_delta_z
        z_safe = torch.clamp(z_final, min=0.01)
        
        fx, fy = self.cam_k[:, 0:1], self.cam_k[:, 1:2]
        cx, cy = self.cam_k[:, 2:3], self.cam_k[:, 3:4]
        
        pred_x = (pred_2d[:, 0:1] - cx) * z_safe / fx
        pred_y = (pred_2d[:, 1:2] - cy) * z_safe / fy
        pred_trans = torch.cat([pred_x, pred_y, z_safe], dim=1)  # (B, 3)

        # ============================================
        # STEP 2: Compute ground truth 2D projection (for L_proj)
        # ============================================
        gt_z_safe = torch.clamp(gt_trans[:, 2:3], min=0.001)
        gt_u = (gt_trans[:, 0:1] * fx / gt_z_safe) + cx
        gt_v = (gt_trans[:, 1:2] * fy / gt_z_safe) + cy
        gt_2d_target = torch.cat([gt_u, gt_v], dim=1)  # (B, 2)

        # ============================================
        # LOSS COMPUTATION: PURE GEOMETRY
        # ============================================
        
        # --- L_rot: Centered ADD/ADD-S (rotation-only geometric loss) ---
        batch_points = self.model_points_bank[class_ids.long()]  # (B, N, 3)
        pred_R = quaternion_to_rotation_matrix(pred_quat)  # (B, 3, 3)
        gt_R = quaternion_to_rotation_matrix(gt_quat)      # (B, 3, 3)
        
        # Compute centered ADD loss for each sample
        # For symmetric objects, use ADD-S; for asymmetric, use ADD
        rot_losses = []
        for i, cid in enumerate(class_ids):
            if self.symmetry_lookup[cid.long()]:
                # Symmetric: use Nearest Neighbor matching
                l = compute_add_s_rotation_only(
                    pred_R[i:i+1], 
                    gt_R[i:i+1], 
                    batch_points[i:i+1]
                )
            else:
                # Asymmetric: point-to-point matching
                l = compute_add_rotation_only(
                    pred_R[i:i+1], 
                    gt_R[i:i+1], 
                    batch_points[i:i+1]
                )
            rot_losses.append(l)
        loss_rot = torch.mean(torch.cat(rot_losses))  # Average over batch
        
        # --- L_trans: Pure Translation L1 ---
        loss_trans = F.l1_loss(pred_trans, gt_trans)  # Direct L1 on [x, y, z]
        
        # --- L_proj: 2D Projection (optional regularization) ---
        loss_proj = F.smooth_l1_loss(pred_2d, gt_2d_target, beta=1.0)

        # ============================================
        # TOTAL LOSS: Weighted sum
        # ============================================
        total_loss = (
            self.w_rot * loss_rot + 
            self.w_trans * loss_trans + 
            self.w_proj * loss_proj
        )

        # ============================================
        # METRICS FOR LOGGING (no gradients)
        # ============================================
        with torch.no_grad():
            trans_err_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            proj_err_px = torch.norm(pred_2d - gt_2d_target, p=2, dim=1).mean()
            # 🎯 DISACCOPPIAMENTO TOTALE: Calcola errore su TUTTO il batch
            # Nessun filtro per simmetrici - così i log mostrano sempre valori reali
            rot_err_deg = compute_batch_rotation_error_all(pred_quat, gt_quat)

        return {
            'total_loss': total_loss,
            'rot_loss': loss_rot.detach(),
            'trans_loss': loss_trans.detach(),
            'proj_loss': loss_proj.detach(),
            'trans_err_cm': trans_err_cm,
            'proj_err_px': proj_err_px,
            'rot_err_deg': rot_err_deg  # 🎯 Rinominato (no 'asymm')
        }