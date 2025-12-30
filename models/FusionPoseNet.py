import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthEncoder(nn.Module):
    """
    CNN semplice per estrarre features dalla depth map (1 canale).
    Input: (B, 1, 224, 224)
    Output: Feature vector
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),               # -> 112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                   # -> 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # -> 28x28
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),            # -> 14x14
            nn.ReLU(),
            
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),    # -> 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                                        # -> 1x1
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1) # Flatten -> (B, feature_dim)


# Ha senso aggiungere un predizione di offset deltaX e deltaY così da
# correggere X e Y predetti col pinhole ??? Vedere in fondo al file le
# modifiche eventuali da applicare
class FusionPoseNet(nn.Module):
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

        # ========== ResNet50 per RGB ==========
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: 2048 dim
        self.rgb_dim = 2048
        
        # ========== CNN per depth ==========
        self.depth_dim = 512
        self.depth_backbone = DepthEncoder(feature_dim=self.depth_dim)
        
        # ========== Fusione e Teste ==========
        # Feature totali dopo la concatenazione
        fusion_dim = self.rgb_dim + self.depth_dim # 2048 + 512 = 2560
        
        # Layer di fusione, comune alla due teste
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024), # meglio rispetto la batchnorm nell'MLP
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Head Rotazione (Quaternioni)
        self.rot_head = nn.Linear(1024, 4)
        
        # Head Traslazione (Z only)
        # Predice solo la Z, X e Y sono calcolate matematicamente
        self.z_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output scalare Z
        )

    def forward(self, rgb, depth, bbox_center_pixel):
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
        fused = torch.cat([rgb_feat, depth_feat], dim=1)            # (B, 2560)
        fused = self.fusion_fc(fused)                               # (B, 1024)
        
        # --- Prediction Heads ---
        # Rotazione
        quaternion = self.rot_head(fused)
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        # Traslazione (solo Z)
        z_pred = self.z_head(fused) # (B, 1)
        # faccio si che Z possa essere solo >0 (altrimenti la fotocamera
        # non lo vedrebbe) e che sia ad almeno 10cm dalla fotocamera. Questo
        # riduce lo spazio di ricerca ed evita anche errori numerici.
        z_pred = F.softplus(z_pred) + 0.1
        
        # --- Differentiable Pinhole Layer ---
        # Usiamo il buffer registrato
        fx = self.cam_k[:, 0:1]
        fy = self.cam_k[:, 1:2]
        cx = self.cam_k[:, 2:3]
        cy = self.cam_k[:, 3:4]
        
        u = bbox_center_pixel[:, 0:1]
        v = bbox_center_pixel[:, 1:2]
        
        # il gradiente fluirà attraverso questa formula fino a z_pred
        x_pred = (u - cx) * z_pred / fx
        y_pred = (v - cy) * z_pred / fy
        
        # Concateniamo per ottenere output (B, 3)
        translation = torch.cat([x_pred, y_pred, z_pred], dim=1)
        
        return quaternion, translation

    def freeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True


# PER AGGIUNGERE deltaX e deltaY

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthEncoder(nn.Module):
    
    CNN semplice per estrarre features dalla depth map (1 canale).
    Input: (B, 1, 224, 224)
    Output: Feature vector

    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


class FusionPoseNet(nn.Module):
    def __init__(self, cam_k):
        super().__init__()

        self.register_buffer(
            'cam_k',
            (torch.tensor([cam_k[0], cam_k[4], cam_k[2], cam_k[5]])).view(1, 4)
        )

        # ========== ResNet50 per RGB ==========
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.rgb_dim = 2048
        
        # ========== CNN per depth ==========
        self.depth_dim = 512
        self.depth_backbone = DepthEncoder(feature_dim=self.depth_dim)
        
        # ========== Fusione e Teste ==========
        fusion_dim = self.rgb_dim + self.depth_dim 
        
        # Layer di fusione, comune alle due teste
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024), 
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Head Rotazione
        self.rot_head = nn.Linear(1024, 4)
        
        # MODIFICA: Head Traslazione Estesa (z, delta_u, delta_v)
        # Predice Z, ma anche l'offset in pixel rispetto al centro del bbox
        self.trans_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: [z_raw, delta_u, delta_v]
        )

    def forward(self, rgb, depth, bbox_center_pixel):
    
        rgb: (B, 3, 224, 224)
        depth: (B, 1, 224, 224)
        bbox_center_pixel: (B, 2) -> [u_bbox, v_bbox]
        
        
        # --- Feature Extraction ---
        rgb_feat = self.rgb_backbone(rgb).view(rgb.size(0), -1)
        depth_feat = self.depth_backbone(depth)
        
        # --- Fusion ---
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        fused = self.fusion_fc(fused)
        
        # --- Prediction Heads ---
        # Rotazione
        quaternion = self.rot_head(fused)
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        # Traslazione (Z + Offsets)
        trans_raw = self.trans_head(fused) # (B, 3)
        
        # 1. Estrazione Z (deve essere positiva)
        z_pred = F.softplus(trans_raw[:, 0:1]) + 0.1
        
        # 2. Estrazione Offsets (possono essere negativi)
        delta_uv = trans_raw[:, 1:3]
        
        # --- Differentiable Pinhole Layer con OFFSET ---
        fx = self.cam_k[:, 0:1]
        fy = self.cam_k[:, 1:2]
        cx = self.cam_k[:, 2:3]
        cy = self.cam_k[:, 3:4]
        
        # Applichiamo l'offset predetto al centro del bbox originale
        u_final = bbox_center_pixel[:, 0:1] + delta_uv[:, 0:1]
        v_final = bbox_center_pixel[:, 1:2] + delta_uv[:, 1:2]
        
        # Pinhole projection inversa usando le coordinate corrette
        x_pred = (u_final - cx) * z_pred / fx
        y_pred = (v_final - cy) * z_pred / fy
        
        translation = torch.cat([x_pred, y_pred, z_pred], dim=1)
        
        return quaternion, translation

    def freeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_rgb(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True
"""

# In extension_train.py (o forse anche altri files) fare
# da {'params': model.z_head.parameters(), 'lr': lr},
# a {'params': model.trans_head.parameters(), 'lr': lr},
# per aggiornare il nome