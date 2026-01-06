import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                   # -> 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # -> 28x28
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
            nn.Dropout(0.5)
        )
        
        # ========== 3 teste finali ========== idee nomi modello: ChimeraPose, TridentNet, HydraPose, Hecate6D, DeltaPose, Fusio3, CerberusNet
        # --- Head Rotazione ---
        self.rot_head = nn.Linear(1024, 4) # Output: 4 quaternioni
        
        # --- Head Depth (Z) - RESIDUAL LEARNING ---
        # Invece di predire Z assoluta, predice un DELTA (correzione skin-to-heart)
        # Inizializzazione vicino a 0 così all'inizio si affida alla geometria
        self.z_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output: Delta_Z (metri) - correzione da applicare a Z_geometric
        )

        # --- Head Offset u & v ---
        # N.B. si sarebbe potuto usare un'unica testa per predirre
        # Z, δu e δv ma Z è in metri e gli altri due pixel. Si potrebbero
        # avere problemi di scala e quindi training più instabile (?).
        self.offset_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Output: δu, δv (correzioni a u e v ottenuti dal BBOX)
        )

        self._init_weights()

    def _init_weights(self):
        # Inizializzazione Xavier per le head lineari
        for m in [self.fusion_fc, self.rot_head, self.z_head, self.offset_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        # Per la rotazione, inizializzazione specifica
        nn.init.xavier_uniform_(self.rot_head.weight, gain=0.01)
        
        # 🎯 RESIDUAL Z: Inizializza delta_z ~0 (piccolo gain per stabilità)
        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.01)
        nn.init.constant_(self.z_head[-1].bias, 0.0)

        with torch.no_grad():
            self.rot_head.bias.fill_(0)
            self.rot_head.bias[0] = 1.0

    def forward(self, rgb, depth, bbox_center_pixel, bbox_dims):
        """
        rgb: (B, 3, 224, 224)
        depth: (B, 1, 224, 224)
        bbox_center_pixel: (B, 2) -> [u, v]
        """
        
        # --- Feature Extraction ---
        rgb_feat = self.rgb_backbone(rgb).view(rgb.size(0), -1)     # (B, 2048)
        depth_feat = self.depth_backbone(depth)                     # (B, 512)
        
        # --- Fusion ---
        # concatenazione vettori
        fused = torch.cat([rgb_feat, depth_feat, bbox_dims], dim=1) # (B, 2562)
        fused = self.fusion_fc(fused)                               # (B, 1024)
        
        # --- Prediction Heads ---
        # ** Rotazione **
        quaternion = self.rot_head(fused)
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        # ** Depth Z - RESIDUAL DELTA **
        # La rete predice solo un DELTA (correzione), non un valore assoluto
        # Delta_Z sarà combinato con Z_geometric nella loss
        delta_z = self.z_head(fused)  # (B, 1) - delta in metri (può essere +/-)
        
        # Nota: Se z_pred diventa troppo piccolo/grande, gradient clipping lo gestisce

        # ** Offset per u & v **
        delta_uv = self.offset_head(fused) # (B, 2)
        uv = bbox_center_pixel + delta_uv
        delta_uv_percent = self.offset_head(fused) # (B, 2)
        # ALTERNATIVA
        #IMG_W = 640.0
        #IMG_H = 480.0
        #img_size = torch.tensor([IMG_W, IMG_H], device=bbox_dims.device).view(1, 2)
        #bbox_size_px = bbox_dims * img_size
        #delta_uv_px = delta_uv_percent * bbox_size_px
        #uv = bbox_center_pixel + delta_uv_px

        # 🎯 HYBRID APPROACH: Restituisci delta_z (non translation completa)
        # La translation finale sarà z_geometric + delta_z (calcolata nella loss)
        return quaternion, delta_z, uv  # [quaternion, delta_z (residual), centro 2D]
        # N.B. ⚠⚠⚠ CONVIENE COMUNQUE RESTITUIRE ANCHE TRANS PER UN FATTO DI COMODITA' AD INFERENCE TIME ⚠⚠⚠

    def freeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_rgb(self, partial=False):
        """Unfreezes RGB backbone.
        
        Args:
            partial:
                - If True, unfreezes only layer4 (last residual block).
                - If False, unfreezes all layers.
        """

        # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        if partial:
            # Sblocca solo layer4 (ultimo blocco convoluzionale)
            for param in self.rgb_backbone[-2].parameters():
                param.requires_grad = True
        else:
            # Sblocca tutta la ResNet
            for param in self.rgb_backbone.parameters():
                param.requires_grad = True
