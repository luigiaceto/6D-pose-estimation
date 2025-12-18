import torch
import torch.nn as nn
import torchvision.models as models


class ResNetPose(nn.Module):
    """
    Modello per 6D pose estimation.
    
    Input: Immagine RGB croppata (dalla bounding box YOLO)
    Output:
        - quaternion (w, x, y, z): rotazione ONLY
    
    La translation 3D viene calcolata geometricamente:
    1. Bbox YOLO -> centro 2D (u,v) e dimensione in pixels
    2. Diametro oggetto da models_info.yml
    3. Z = (diametro_reale * focal) / diametro_pixels (pinhole formula)
    4. (u,v,Z) -> (X,Y,Z) con pinhole unprojection
    """
    
    def __init__(self, pretrained=True, dropout=0.3):
        super(ResNetPose, self).__init__()
        
        # ResNet18 backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        
        feature_dim = 2048
        
        # Rotation head: predice SOLO quaternion (w, x, y, z)
        self.quaternion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # quaternion (w, x, y, z)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - batch di immagini RGB cropped
            
        Returns:
            quaternion: (B, 4) - quaternion normalizzato (w, x, y, z)
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Predict quaternion and normalize
        quaternion = self.quaternion_head(features)
        quaternion = quaternion / (torch.norm(quaternion, dim=1, keepdim=True) + 1e-8)
        
        return quaternion
    
    def freeze_backbone(self):
        """Freeze ResNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def quaternion_to_rotation_matrix(quaternion):
    """
    Converte quaternion (w, x, y, z) a rotation matrix (3x3).
    
    Args:
        quaternion: (B, 4) tensor
        
    Returns:
        rotation_matrix: (B, 3, 3) tensor
    """
    batch_size = quaternion.shape[0]
    
    # Normalize
    quaternion = quaternion / (torch.norm(quaternion, dim=1, keepdim=True) + 1e-8)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Rotation matrix elements
    R = torch.zeros(batch_size, 3, 3, device=quaternion.device, dtype=quaternion.dtype)
    
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R
