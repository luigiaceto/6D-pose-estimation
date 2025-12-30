import torch
import torch.nn as nn


class RGBDPoseLoss(nn.Module):
    """
    Loss combinata per RGB-D Pose Estimation.
    Gestisce sia la Rotazione (Quaternioni) che la Traslazione (XYZ).
    NON eredita da PoseLoss per evitare conflitti sui gradienti.
    """
    
    def __init__(self, lambda_rot=20.0, lambda_trans=1.0):
        super(RGBDPoseLoss, self).__init__()
        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans
        
        # Loss L1 per la traslazione (forse meglio nn.SmoothL1Loss(beta=1.0) rispetto nn.L1Loss ???)
        self.trans_loss_fn = nn.SmoothL1Loss(beta=1.0)

        # PARAMETRI LEARNABLE
        # Inizializziamo a -2.0 o 0.0. Rappresentano log(sigma^2).
        # Un valore negativo iniziale dà un peso iniziale alto alle loss,
        # costringendo la rete a imparare velocemente all'inizio.
        self.s_rot = nn.Parameter(torch.tensor(-2.0), requires_grad=True)
        self.s_trans = nn.Parameter(torch.tensor(-2.0), requires_grad=True)

    # non risolve problemi per gli oggetti 'simmetrici', occorrerebbe usare la
    # ADD ma diventa molto pesante il training ???
    def compute_rot_loss(self, pred_q, gt_q):
        """
        Calcola la distanza angolare tra quaternioni.
        Formula: 1 - |<q1, q2>|
        """
        # Normalizzazione di sicurezza (per evitare instabilità numeriche).
        # Anche se la rete ha una normalizzazione finale, è meglio rifarla qui per sicurezza
        pred_q = pred_q / (torch.norm(pred_q, dim=1, keepdim=True) + 1e-8)
        gt_q = gt_q / (torch.norm(gt_q, dim=1, keepdim=True) + 1e-8)
        
        # Dot product.
        # q e -q rappresentano la stessa rotazione, quindi prendiamo il valore assoluto
        dot_product = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        
        # Clamp per evitare problemi numerici
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        
        # Loss: Vogliamo massimizzare il dot product (avvicinarlo a 1)
        # Quindi minimizziamo 1 - dot
        return torch.mean(1.0 - dot_product)

    def forward(self, pred_quat, pred_trans, gt_quat, gt_trans, class_ids=None):
        """
        Calcola la loss totale.
        TUTTI gli input devono essere su GPU e far parte del grafo computazionale.
        """
        
        # --- Loss Rotazione ---
        # Restituisce un Tensor con gradiente attivo
        loss_r = self.compute_rot_loss(pred_quat, gt_quat)
        
        # --- Loss Traslazione ---
        # pred_trans viene dal Pinhole Layer differenziabile -> ha gradiente attivo
        loss_t = self.trans_loss_fn(pred_trans, gt_trans)
        
        # --- Loss Totale Pesata ---
        # Somma pesata delle due componenti
        weighted_loss_r = torch.exp(-self.s_rot) * loss_r + self.s_rot
        weighted_loss_t = torch.exp(-self.s_trans) * loss_t + self.s_trans
        total_loss = weighted_loss_r + weighted_loss_t
        
        # --- Metriche per Logging (Senza gradienti) ---
        # Calcoliamo l'errore in cm solo per stamparlo a video
        with torch.no_grad():
            # Distanza euclidea media tra i vettori
            diff = pred_trans - gt_trans
            dist_m = torch.norm(diff, p=2, dim=1).mean()
            error_cm = dist_m * 100 # Converti in cm
        
        return {
            'total_loss': total_loss,           # Tensor (per .backward())
            'rot_loss': loss_r.detach(),        # Float (per print/log)
            'trans_loss': loss_t.detach(),      # Float (per print/log)
            'trans_err_cm': error_cm.detach()   # Float (per capire quanto sbaglia in cm)
        }