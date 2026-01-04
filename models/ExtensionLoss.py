import torch
import torch.nn as nn
from utils.pose_utils import (
    batch_add_loss, 
    batch_adds_loss, 
    quaternion_to_rotation_matrix,
    compute_batch_rotation_error, 
    SYMMETRIC_OBJECTS
)


class RGBDPoseLoss(nn.Module):
    """
    Loss ottimizzata per RGB-D Pose Estimation (Extension).
    
    Caratteristiche:
    1. Hybrid Rotation Loss: Quaternion Loss per asimmetrici, Matrix Geodesic per simmetrici.
    2. Learnable Uncertainties: Pesi s_rot, s_trans, s_proj appresi automaticamente.
    3. Pinhole Constraint: Loss sulla riproiezione 2D (u, v) per legare geometria e visione.
    """
    
    def __init__(self, cam_k, model_points_dict):
        super(RGBDPoseLoss, self).__init__()

        # Buffer per parametri camera
        self.register_buffer(
            'cam_k',
            (torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]])).view(1, 4)
        )
        
        # Creiamo un unico tensore che contiene i punti di TUTTI gli oggetti.
        # Shape: (MAX_ID + 1, Num_Points, 3)
        max_id = max(model_points_dict.keys())
        # Assumiamo tutti abbiano lo stesso numero di punti (es. 1000)
        n_pts = list(model_points_dict.values())[0].shape[0]
            
        bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32)
            
        for oid, pts in model_points_dict.items():
            bank[oid] = pts
            
        # register_buffer sposta automaticamente questo tensore su GPU insieme al modello
        self.register_buffer('model_points_bank', bank)

        self.proj_loss_fn = nn.MSELoss()

        self.w_add = 10.0   
        self.w_proj = 1.0


    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, pred_2d, class_ids):
        """
        Calcola la loss totale pesata.
        """
        
        batch_points = self.model_points_bank[class_ids.long()]

        pred_R = quaternion_to_rotation_matrix(pred_quat) # (B, 3, 3)
        gt_R = quaternion_to_rotation_matrix(gt_quat)     # (B, 3, 3)

        pred_t = pred_trans.unsqueeze(2) # (B, 3, 1) per broadcasting
        gt_t = gt_trans.unsqueeze(2)     # (B, 3, 1)

        losses = batch_add_loss(pred_R, pred_t, gt_R, gt_t, batch_points)
        # Gestione Simmetrie (Sovrascrittura selettiva)
        # Se ci sono oggetti simmetrici nel batch, ricalcoliamo la loro loss con ADD-S
        # (Questo loop è breve perché itera sul batch size, non sui punti)
        for i, cid in enumerate(class_ids):
            if cid.item() in SYMMETRIC_OBJECTS:
                l_adds = batch_adds_loss(
                    pred_R[i:i+1], pred_t[i:i+1], 
                    gt_R[i:i+1], gt_t[i:i+1], 
                    batch_points[i:i+1]
                )
                losses[i] = l_adds

        loss_add = torch.mean(losses)
        
        fx, fy = self.cam_k[:, 0:1], self.cam_k[:, 1:2]
        cx, cy = self.cam_k[:, 2:3], self.cam_k[:, 3:4]
        gt_z_safe = torch.clamp(gt_trans[:, 2:3], min=0.001)
        gt_u = (gt_trans[:, 0:1] * fx / gt_z_safe) + cx
        gt_v = (gt_trans[:, 1:2] * fy / gt_z_safe) + cy
        gt_2d_target = torch.cat([gt_u, gt_v], dim=1)

        loss_proj = self.proj_loss_fn(pred_2d, gt_2d_target)

        total_loss = self.w_add * loss_add + self.w_proj * loss_proj
        
        with torch.no_grad(): 
            trans_err_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            proj_err_px = torch.norm(pred_2d - gt_2d_target, p=2, dim=1).mean()
            rot_err_deg = compute_batch_rotation_error(pred_quat, gt_quat)
        
        return {
            # loss
            'total_loss': total_loss,
            'add_loss': loss_add.detach(),
            'proj_loss': loss_proj.detach(),
            # errori
            'trans_err_cm': trans_err_cm.detach(),
            'proj_err_px': proj_err_px.detach(),
            'rot_err_deg': torch.tensor(rot_err_deg)
        }
