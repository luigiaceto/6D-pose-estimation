import torch
import torch.nn as nn
from utils.pose_utils import (
    batch_add_loss, 
    batch_adds_loss, 
    quaternion_to_rotation_matrix,
    compute_batch_rotation_error_asymm,
    compute_quaternion_loss,
    compute_matrix_geodesic_loss,
    SYMMETRIC_OBJECTS
)

class ExtensionLoss(nn.Module):
    def __init__(self, add_weight, proj_weight, cam_k, model_points_dict, rot_weight=0.0, trans_weight=0.0):
        super().__init__()

        # Register buffers (non-trainable params)
        self.register_buffer(
            'cam_k',
            (torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]])).view(1, 4)
        )
        
        max_id = max(model_points_dict.keys())
        n_pts = list(model_points_dict.values())[0].shape[0]
        bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32)
        for oid, pts in model_points_dict.items():
            bank[oid] = pts
        self.register_buffer('model_points_bank', bank)
    
        symmetry_mask = torch.zeros(max_id + 1, dtype=torch.bool)
        for obj_id in SYMMETRIC_OBJECTS:
            if obj_id <= max_id:
                symmetry_mask[obj_id] = True
        self.register_buffer('symmetry_lookup', symmetry_mask)

        # Loss functions base
        self.proj_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.trans_loss_fn = nn.SmoothL1Loss(beta=1.0)

        # Pesi iniziali (modificabili runtime)
        self.w_add = add_weight   
        self.w_proj = proj_weight
        self.w_rot = rot_weight
        self.w_trans = trans_weight

    def forward(self, pred_quat, pred_delta_z, gt_quat, gt_trans, pred_2d, class_ids, z_geometric):
        # 1. Ricostruzione 3D (Hybrid Depth)
        z_final = z_geometric.detach() + pred_delta_z
        z_safe = torch.clamp(z_final, min=0.01)
        
        fx, fy = self.cam_k[:, 0:1], self.cam_k[:, 1:2]
        cx, cy = self.cam_k[:, 2:3], self.cam_k[:, 3:4]
        
        pred_x = (pred_2d[:, 0:1] - cx) * z_safe / fx
        pred_y = (pred_2d[:, 1:2] - cy) * z_safe / fy
        pred_trans = torch.cat([pred_x, pred_y, z_safe], dim=1)

        # 2. Target 2D (per Projection Loss)
        gt_z_safe = torch.clamp(gt_trans[:, 2:3], min=0.001)
        gt_u = (gt_trans[:, 0:1] * fx / gt_z_safe) + cx
        gt_v = (gt_trans[:, 1:2] * fy / gt_z_safe) + cy
        gt_2d_target = torch.cat([gt_u, gt_v], dim=1)

        # --- CALCOLO LOSSES ---

        # A. Rotation Loss (Sempre calcolata per log, applicata se w_rot > 0)
        rot_losses = []
        for i, cid in enumerate(class_ids):
            if cid.item() in SYMMETRIC_OBJECTS:
                # Geodesic loss per simmetrici (più robusta)
                l = compute_matrix_geodesic_loss(pred_quat[i:i+1], gt_quat[i:i+1])
            else:
                # Quaternion loss standard
                l = compute_quaternion_loss(pred_quat[i:i+1], gt_quat[i:i+1])
            rot_losses.append(l)
        loss_rot = torch.mean(torch.stack(rot_losses))

        # B. Translation & Projection Losses
        loss_trans_pure = self.trans_loss_fn(pred_trans, gt_trans)
        loss_proj = self.proj_loss_fn(pred_2d, gt_2d_target)

        # C. ADD Loss (Calcolata solo se serve o per logging leggero)
        # Nota: ADD è computazionalmente pesante, la calcoliamo se ha peso > 0
        loss_add = torch.tensor(0.0, device=pred_quat.device)
        
        # Se w_add > 0 OPPURE siamo in validation (vogliamo vederla nei log)
        # Per training veloce, calcola solo se w_add > 0
        if self.w_add > 0 or not self.training:
            batch_points = self.model_points_bank[class_ids.long()]
            pred_R = quaternion_to_rotation_matrix(pred_quat)
            gt_R = quaternion_to_rotation_matrix(gt_quat)
            pred_t = pred_trans.unsqueeze(2)
            gt_t = gt_trans.unsqueeze(2)
            
            losses = batch_add_loss(pred_R, pred_t, gt_R, gt_t, batch_points)
            
            # Gestione Simmetria ADD-S
            for i, cid in enumerate(class_ids):
                if cid.item() in SYMMETRIC_OBJECTS:
                    l_adds = batch_adds_loss(
                        pred_R[i:i+1], pred_t[i:i+1], 
                        gt_R[i:i+1], gt_t[i:i+1], 
                        batch_points[i:i+1]
                    )
                    losses[i] = l_adds
            loss_add = torch.mean(losses)

        # --- SOMMA PESATA ---
        total_loss = (
            self.w_rot * loss_rot + 
            self.w_trans * loss_trans_pure + 
            self.w_proj * loss_proj + 
            self.w_add * loss_add
        )

        # --- METRICHE PER LOGGING ---
        with torch.no_grad():
            trans_err_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            proj_err_px = torch.norm(pred_2d - gt_2d_target, p=2, dim=1).mean()
            rot_err_deg = compute_batch_rotation_error_asymm(pred_quat, gt_quat, class_ids, self.symmetry_lookup)

        return {
            'total_loss': total_loss,
            'add_loss': loss_add.detach(),
            'proj_loss': loss_proj.detach(),
            'rot_loss': loss_rot.detach(),
            'trans_loss': loss_trans_pure.detach(),
            'trans_err_cm': trans_err_cm.detach(),
            'proj_err_px': proj_err_px.detach(),
            'rot_err_asymm_deg': rot_err_deg
        }