import torch
import torch.nn as nn
from utils.pose_utils import (
    batch_add_loss, 
    batch_adds_loss, 
    quaternion_to_rotation_matrix,
    compute_batch_rotation_error,
    compute_quaternion_loss,
    compute_matrix_geodesic_loss,
    SYMMETRIC_OBJECTS
)


class RGBDPoseLoss(nn.Module):
    """
    Loss ottimizzata per RGB-D Pose Estimation (Extension).
    
    Caratteristiche:
    1. Hybrid Rotation Loss: Quaternion Loss per asimmetrici, Matrix Geodesic per simmetrici.
    2. Learnable Uncertainties: Pesi s_rot, s_trans, s_proj appresi automaticamente.
    3. Pinhole Constraint: Loss sulla riproiezione 2D (u, v) per legare geometria e visione.
    
    Args:
        add_weight: Peso per ADD loss (geometrica)
        proj_weight: Peso per projection loss (2D)
        rot_weight: Peso per rotation loss diretta (solo con loss_mode='rotation')
        cam_k: Parametri intrinseci camera
        model_points_dict: Dizionario con punti 3D dei modelli
        loss_mode: 'add' (default, usa ADD+proj) o 'rotation' (usa rotation loss diretta)
    """
    
    def __init__(self, add_weight, proj_weight, cam_k, model_points_dict, rot_weight=0.0, loss_mode='add'):
        super(RGBDPoseLoss, self).__init__()

        # Buffer per parametri camera
        self.register_buffer(
            'cam_k',
            (torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]])).view(1, 4)
        )
        
        # --- VETTORE MAPPA DI MODELLI 3D ---
        max_id = max(model_points_dict.keys())
        # tutti agli oggetti hanno lo stesso numero di punti (es. 1000)
        n_pts = list(model_points_dict.values())[0].shape[0]
        bank = torch.zeros((max_id + 1, n_pts, 3), dtype=torch.float32) # vettore dim 16 (da 0 a 15)
        for oid, pts in model_points_dict.items(): # riempio solo gli indici corrispondenti ad ID (1, 2, 4, 5, ...)
            bank[oid] = pts
        self.register_buffer('model_points_bank', bank)
    
        # --- VETTORE MASCHERA DI OGGETTI SIMMETRICI ---
        max_id = max(model_points_dict.keys())
        symmetry_mask = torch.zeros(max_id + 1, dtype=torch.bool)
        for obj_id in SYMMETRIC_OBJECTS:
            if obj_id <= max_id:
                symmetry_mask[obj_id] = True
        self.register_buffer('symmetry_lookup', symmetry_mask)

        self.proj_loss_fn = nn.MSELoss()

        self.w_add = add_weight   
        self.w_proj = proj_weight
        self.w_rot = rot_weight
        self.loss_mode = loss_mode
        
        if loss_mode not in ['add', 'rotation']:
            raise ValueError(f"loss_mode deve essere 'add' o 'rotation', ricevuto: {loss_mode}")


    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, pred_2d, class_ids):
        """
        Calcola la loss totale pesata.
        """
        
        if self.loss_mode == 'add':
            # ========== MODALITÀ ADD (Default) ==========
            batch_points = self.model_points_bank[class_ids.long()]

            pred_R = quaternion_to_rotation_matrix(pred_quat) # (B, 3, 3)
            gt_R = quaternion_to_rotation_matrix(gt_quat)     # (B, 3, 3)

            pred_t = pred_trans.unsqueeze(2) # (B, 3, 1) per broadcasting
            gt_t = gt_trans.unsqueeze(2)     # (B, 3, 1)

            losses = batch_add_loss(pred_R, pred_t, gt_R, gt_t, batch_points)
            # Gestione Simmetrie (Sovrascrittura selettiva)
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
            loss_rot = torch.tensor(0.0, device=pred_quat.device)  # Placeholder

            total_loss = self.w_add * loss_add + self.w_proj * loss_proj
        
        else:  # loss_mode == 'rotation'
            # ========== MODALITÀ ROTATION (Chirurgia Rotazione) ==========
            # Calcola ADD solo per metriche (peso basso o zero)
            batch_points = self.model_points_bank[class_ids.long()]
            pred_R = quaternion_to_rotation_matrix(pred_quat)
            gt_R = quaternion_to_rotation_matrix(gt_quat)
            pred_t = pred_trans.unsqueeze(2)
            gt_t = gt_trans.unsqueeze(2)
            
            losses = batch_add_loss(pred_R, pred_t, gt_R, gt_t, batch_points)
            for i, cid in enumerate(class_ids):
                if cid.item() in SYMMETRIC_OBJECTS:
                    l_adds = batch_adds_loss(
                        pred_R[i:i+1], pred_t[i:i+1], 
                        gt_R[i:i+1], gt_t[i:i+1], 
                        batch_points[i:i+1]
                    )
                    losses[i] = l_adds
            loss_add = torch.mean(losses)
            
            # Calcola ROTATION LOSS DIRETTA (come baseline)
            rot_losses = []
            for i, cid in enumerate(class_ids):
                if cid.item() in SYMMETRIC_OBJECTS:
                    # Per i simmetrici usiamo la geodesic loss sulle matrici
                    l = compute_matrix_geodesic_loss(pred_quat[i:i+1], gt_quat[i:i+1])
                else:
                    # Per asimmetrici (Ape!) usiamo la loss diretta sui quaternioni
                    l = compute_quaternion_loss(pred_quat[i:i+1], gt_quat[i:i+1])
                rot_losses.append(l)
            loss_rot = torch.mean(torch.stack(rot_losses))
            
            loss_proj = torch.tensor(0.0, device=pred_quat.device)  # Non usata in questa modalità
            
            # PRIORITÀ TOTALE SULLA ROTAZIONE
            total_loss = self.w_add * loss_add + self.w_rot * loss_rot
        
        with torch.no_grad(): 
            trans_err_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            proj_err_px = torch.norm(pred_2d - gt_2d_target, p=2, dim=1).mean()
            rot_err_deg = compute_batch_rotation_error_asymm(pred_quat, gt_quat, class_ids, self.symmetry_lookup)
        
        return {
            # loss
            'total_loss': total_loss,
            'add_loss': loss_add.detach(),
            'proj_loss': loss_proj.detach(),
            'rot_loss': loss_rot.detach(),
            # errori
            'trans_err_cm': trans_err_cm.detach(),
            'proj_err_px': proj_err_px.detach(),
            'rot_err_asymm_deg': rot_err_deg
        }
