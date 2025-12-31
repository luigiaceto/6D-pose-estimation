import torch
import numpy as np

class PinholeCamera:
    """
    Pinhole camera model per conversione tra coordinate 2D e 3D.
    Gestisce sia Tensor PyTorch (differenziabili) che Numpy Array.
    """
    
    def __init__(self, cam_k):
        self.fx, self.fy, self.cx, self.cy = cam_k[0], cam_k[4], cam_k[2], cam_k[5]
        self.K = cam_k
        
        # Salviamo K anche come tensore (1, 4) [fx, fy, cx, cy] per uso interno con PyTorch
        intrinsics_params = [self.fx, self.fy, self.cx, self.cy]
        if torch.is_tensor(cam_k):
            # Se cam_k è già un tensore, prendiamo i valori scalari e creiamo (1, 4)
            self.K_tensor = torch.tensor(intrinsics_params, dtype=torch.float32, device=cam_k.device).view(1, 4)
        else:
            # Se cam_k è una lista o numpy, lo convertiamo in float32
            self.K_tensor = torch.tensor(intrinsics_params, dtype=torch.float32).view(1, 4)
    
    @staticmethod
    def apply_unprojection(points_2d, depth, intrinsics):
        """
        CORE STATICO: Metodo 'funzionale' per unprojecting differenziabile.
        Questa è l'unica parte dove la formula matematica risiede per PyTorch.
        
        Args:
            points_2d: (B, 2) tensor [u, v]
            depth: (B,) or (B, 1) tensor [Z]
            intrinsics: (1, 4) or (B, 4) tensor [fx, fy, cx, cy]
        
        Returns:
            points_3d: (B, 3) tensor [X, Y, Z]
        """
        batch_size = points_2d.shape[0]
        
        # Assicuriamoci che depth abbia shape (B, 1)
        if depth.dim() == 1:
            depth = depth.unsqueeze(1)
        
        # Se intrinsics ha batch_size=1, espandiamolo al batch_size corretto
        if intrinsics.shape[0] == 1 and batch_size > 1:
            intrinsics = intrinsics.expand(batch_size, -1)
        
        # Estrazione parametri dal tensore batch (B, 4)
        fx = intrinsics[:, 0:1]
        fy = intrinsics[:, 1:2]
        cx = intrinsics[:, 2:3]
        cy = intrinsics[:, 3:4]
        
        u = points_2d[:, 0:1]
        v = points_2d[:, 1:2]
        
        # Formula Pinhole Inversa (unica fonte di verità)
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        
        return torch.cat([X, Y, depth], dim=1)

    def unproject_2d_to_3d(self, points_2d, depth):
        """
        Metodo d'istanza (usato dalla Baseline).
        Se l'input è PyTorch, DELEGA al metodo statico apply_unprojection.
        """
        is_torch = torch.is_tensor(points_2d)
        
        if is_torch:
            # Assicuriamoci che i parametri intrinseci siano sullo stesso device dei dati
            K_tensor_device = self.K_tensor.to(points_2d.device)
            
            return PinholeCamera.apply_unprojection(points_2d, depth, K_tensor_device)
            
        else:
            u, v = points_2d[:, 0], points_2d[:, 1]
            Z = depth
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            points_3d = np.stack([X, Y, Z], axis=1)
            return points_3d

    def project_3d_to_2d(self, points_3d):
        """
        Proietta punti 3D in coordinate camera a 2D image.
        """
        is_torch = torch.is_tensor(points_3d)
        
        if is_torch:
            X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
            u = self.fx * X / Z + self.cx
            v = self.fy * Y / Z + self.cy
            points_2d = torch.stack([u, v], dim=1)
        else:
            X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
            u = self.fx * X / Z + self.cx
            v = self.fy * Y / Z + self.cy
            points_2d = np.stack([u, v], axis=1)
        
        return points_2d
    
    def compute_depth_from_bbox(self, bbox, object_diameter):
        """
        Calcola depth Z usando il diametro dell'oggetto e la dimensione del bbox.
        """
        is_torch = torch.is_tensor(bbox)
        
        if is_torch:
            bbox_width = bbox[:, 2] - bbox[:, 0]
            bbox_height = bbox[:, 3] - bbox[:, 1]
            bbox_size = torch.maximum(bbox_width, bbox_height)
            
            # Conversione mm -> metri e formula pinhole
            depth = (object_diameter / 1000.0) * self.fx / bbox_size
        else:
            bbox_width = bbox[:, 2] - bbox[:, 0]
            bbox_height = bbox[:, 3] - bbox[:, 1]
            bbox_size = np.maximum(bbox_width, bbox_height)
            
            depth = (object_diameter / 1000.0) * self.fx / bbox_size
        
        return depth
    
    def get_intrinsics_matrix(self):
        return self.K