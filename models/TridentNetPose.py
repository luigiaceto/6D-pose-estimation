import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.pose_utils import IMG_HEIGHT, IMG_WIDTH, compute_z_from_depth_crop
from models.PinholeCamera import PinholeCamera

class DepthEncoder(nn.Module):
    """
    CNN semplice per estrarre features dalla depth map (1 canale).
    Input: (B, 1, 224, 224)
    Output: Feature vector
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(

            nn.BatchNorm2d(1),

            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),               # -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # -> 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),            # -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),    # -> 7x7
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                                        # -> 1x1
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1) # Flatten -> (B, feature_dim)


class TridentNetPose(nn.Module):
    def __init__(self, cam_k):
        super().__init__()

        # serve per poter spostare la variabile statica 'cam_k'
        # su GPU insieme al modello
        self.register_buffer(
            'cam_k',
            (
                torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]]) # [fx, fy, cx, cy]
            ).view(1, 4)
        )

        # ========== ResNet50 processa RGB ==========
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: 2048 dim
        self.rgb_dim = 2048
        
        # ========== CNN processa depth ==========
        self.depth_dim = 512
        self.depth_backbone = DepthEncoder(feature_dim=self.depth_dim)
        
        # ========== Fusione ==========
        # Feature totali dopo la concatenazione
        fusion_dim = self.rgb_dim + self.depth_dim + 2 # 2048 + 512 + 2 = 2562
        
        # Layer di fusione, comune alle teste finali
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024), # nel layer di fusione è meglio rispetto la BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ========== 3 teste finali ==========
        # --- Head Rotazione ---
        self.rot_head = nn.Linear(1024, 4) # Output: 4 quaternioni
        
        # --- Head Depth (Z) 
        # Invece di predire Z assoluta, predice un DELTA (correzione skin-to-heart)
        # Inizializzazione vicino a 0 così all'inizio si affida alla geometria
        self.z_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output: Delta_Z (metri) - correzione da applicare a Z_geometric
        )

        # --- Head Offset u & v ---
        self.offset_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Output: δu, δv (correzioni a u e v ottenuti dal BBOX)
        )

        self._init_weights()

    def _init_weights(self):
        # Inizializzazione GENERICA per i moduli "standard"
        for m in [self.fusion_fc, self.z_head, self.offset_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        # Inizializzazione SPECIFICA per la Rotazione (Rot Head)
        # Gain 0.01 serve a far partire la rotazione come "quasi uniforme" ma deterministica
        nn.init.xavier_uniform_(self.rot_head.weight, gain=0.01)
        
        # Bias: Quaternione identità [1, 0, 0, 0] (w, x, y, z)
        # Importante: così all'inizio la rete predice "nessuna rotazione" invece di rotazione random
        with torch.no_grad():
            self.rot_head.bias.fill_(0)
            self.rot_head.bias[0] = 1.0

        # Sovrascrittura SPECIFICA per l'ultimo layer di Z (Residual).
        # I layer precedenti di z_head sono stati inizializzati nel loop sopra (gain 1.0).
        # Qui sovrascriviamo SOLO l'ultimo per farlo partire da ~0.
        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.01)
        nn.init.constant_(self.z_head[-1].bias, 0.0)

    def forward(self, rgb, depth, bbox_center_pixel, bbox_dims):
        """
        Forward pass con back-projection interna: calcola direttamente la translation 3D.
        
        Args:
            rgb: (B, 3, 224, 224) - RGB crop
            depth: (B, 1, 224, 224) - Depth crop
            bbox_center_pixel: (B, 2) - Centro bbox [u, v]
            bbox_dims: (B, 2) - Dimensioni bbox normalizzate [w%, h%]
        
        Returns:
            pred_quat: (B, 4) - Quaternione normalizzato
            pred_trans: (B, 3) - Translation assoluta in metri [x, y, z]
        """
        
        # --- Calcolo stima geometrica Z iniziale ---
        z_geometric = compute_z_from_depth_crop(cropped_depth=depth)  # (B, 1)
        
        # --- Feature Extraction ---
        rgb_feat = self.rgb_backbone(rgb).view(rgb.size(0), -1)     # (B, 2048)
        depth_centered = depth - z_geometric.view(-1, 1, 1, 1)
        depth_feat = self.depth_backbone(depth_centered)             # (B, 512)
        
        # --- Fusion ---
        fused = torch.cat([rgb_feat, depth_feat, bbox_dims], dim=1) # (B, 2562)
        fused = self.fusion_fc(fused)                               # (B, 1024)
        
        # --- Prediction Heads ---
        # Rotazione 
        pred_quat = self.rot_head(fused)
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        
        # Depth Z - RESIDUAL DELTA
        delta_z = self.z_head(fused)  # (B, 1) - predice delta solo dalle feature
        pred_z = z_geometric + delta_z  # (B, 1)
        pred_z = torch.clamp(pred_z, min=0.01)  # Evita depth negativa
        
        # Offset 2D (percentuale bbox)
        delta_pct = self.offset_head(fused)  # (B, 2)
        offset_px_x = delta_pct[:, 0:1] * bbox_dims[:, 0:1] * IMG_WIDTH     # (B, 1)
        offset_px_y = delta_pct[:, 1:2] * bbox_dims[:, 1:2] * IMG_HEIGHT    # (B, 1)
        offset_px = torch.cat([offset_px_x, offset_px_y], dim=1)            # (B, 2)
        
        # Coordinate pixel finali
        pred_uv = bbox_center_pixel + offset_px  # (B, 2)
        
        # BACK-PROJECTION: Da (u, v, z) a (x, y, z)
        pred_trans = PinholeCamera.apply_unprojection(
            points_2d=pred_uv,     # (B, 2)
            depth=pred_z,          # (B, 1)
            intrinsics=self.cam_k  # (1, 4)
        )  # Restituisce (B, 3)
        
        return pred_quat, pred_trans, pred_uv

    def freeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_rgb(self):
        """
        Unfreezes only layer4 (last residual block) of RGB backbone.
        
        Partial unfreeze strategy:
        - Mantiene frozen i layer bassi (conv1-layer3) per preservare feature ImageNet
        - Sblocca solo layer4 (ultimo blocco) per adattamento al task specifico
        - Migliore stabilità con dataset piccoli come LineMOD
        """
        for param in self.rgb_backbone[-2].parameters():
            param.requires_grad = True
