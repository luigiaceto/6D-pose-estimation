import torch
import torch.nn as nn
from utils.pose_utils import compute_matrix_geodesic_loss, compute_quaternion_loss,  SYMMETRIC_OBJECTS


class RGBDPoseLoss(nn.Module):
    """
    Loss ottimizzata per RGB-D Pose Estimation (Extension).
    
    Caratteristiche:
    1. Hybrid Rotation Loss: Quaternion Loss per asimmetrici, Matrix Geodesic per simmetrici.
    2. Learnable Uncertainties: Pesi s_rot, s_trans, s_proj appresi automaticamente.
    3. Pinhole Constraint: Loss sulla riproiezione 2D (u, v) per legare geometria e visione.
    """
    
    def __init__(self, cam_k):
        super(RGBDPoseLoss, self).__init__()

        # Buffer per parametri camera
        self.register_buffer(
            'cam_k',
            (torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]])).view(1, 4)
        )
        
        # Loss per traslazione e proiezione
        self.trans_loss_fn = nn.SmoothL1Loss(beta=1.0) # Robusta agli outlier
        self.proj_loss_fn = nn.MSELoss()

        # --- PARAMETRI LEARNABLE (Kendall's Multi-Task Loss) ---
        # Eventualmente inizializzare a valori negativi (es. -2.0) per dare
        # un peso iniziale alto
        #self.s_rot = nn.Parameter(torch.tensor(0.0), requires_grad=True) # peso iniziale ~1.0
        #self.s_trans = nn.Parameter(torch.tensor(-4.6), requires_grad=True) # peso iniziale ~100.0
        #self.s_proj = nn.Parameter(torch.tensor(2.3), requires_grad=True) # peso iniziale ~0.1


    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, pred_2d, class_ids=None):
        """
        Calcola la loss totale pesata.
        """
        
        # 1. Calcolo target 2D per la Projection Loss
        fx, fy = self.cam_k[:, 0:1], self.cam_k[:, 1:2]
        cx, cy = self.cam_k[:, 2:3], self.cam_k[:, 3:4]
        gt_z = gt_trans[:, 2:3] # Z deve essere > 0
        
        # Evitiamo divisioni per zero se GT ha errori (clamp a 1mm)
        gt_z = torch.clamp(gt_z, min=0.001) 
        
        gt_u = (gt_trans[:, 0:1] * fx / gt_z) + cx
        gt_v = (gt_trans[:, 1:2] * fy / gt_z) + cy
        gt_2d = torch.cat([gt_u, gt_v], dim=1) # (B, 2)

        # --- LOSS ROTAZIONE IBRIDA ---
        if class_ids is not None:
            rot_losses = []
            for i in range(len(class_ids)):
                cid = int(class_ids[i].item())
                # Se simmetrico -> Matrix Loss
                if cid in SYMMETRIC_OBJECTS:
                    l = compute_matrix_geodesic_loss(pred_quat[i:i+1], gt_quat[i:i+1])
                # Se asimmetrico -> Quaternion Loss
                else:
                    l = compute_quaternion_loss(pred_quat[i:i+1], gt_quat[i:i+1])
                rot_losses.append(l)
            loss_r = torch.mean(torch.stack(rot_losses))
        else:
            # Fallback (tutti asimmetrici)
            loss_r = compute_quaternion_loss(pred_quat, gt_quat)
        
        # --- LOSS TRASLAZIONE & PROIEZIONE ---
        loss_t = self.trans_loss_fn(pred_trans, gt_trans)
        loss_p = self.proj_loss_fn(pred_2d, gt_2d)

        # --- LOSS TOTALE PESATA (Learnable) ---
        # ogni termine è del tipo exp(-s) * Loss + s, con s learnable
        #weighted_loss_r = torch.exp(-self.s_rot) * loss_r + self.s_rot
        #weighted_loss_t = torch.exp(-self.s_trans) * loss_t + self.s_trans
        #weighted_loss_p = torch.exp(-self.s_proj) * loss_p + self.s_proj
        #total_loss = weighted_loss_r + weighted_loss_t + weighted_loss_p

        # ALTERNATIVA: loss con pesi fissati
        # Rotazione (range 0-1): Peso neutro
        w_rot = 2.0   
        # Traslazione (range 0.001 - 0.1): Peso ENORME
        # Dobbiamo portare quel 0.005 a valere quanto la rotazione/proiezione
        w_trans = 100.0  
        # Proiezione (range 10 - 100): Peso PICCOLO
        # Dobbiamo ridurre quei 25.0 per non farli dominare
        w_proj = 0.1

        total_loss = w_rot * loss_r + w_trans * loss_t + w_proj * loss_p

        # --- LOGGING ---
        with torch.no_grad():
            # calcolo errore come norma euclidea (magnituto vettore)
            error_cm = torch.norm(pred_trans - gt_trans, p=2, dim=1).mean() * 100
            error_px = torch.norm(pred_2d - gt_2d, p=2, dim=1).mean()
        
        return {
            'total_loss': total_loss,
            'rot_loss': loss_r.detach(),
            'trans_loss': loss_t.detach(),
            'trans_err_cm': error_cm.detach(),
            'proj_err_px': error_px.detach()
        }