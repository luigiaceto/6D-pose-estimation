import torch
import numpy as np


class PinholeCamera:
    """
    Pinhole camera model per conversione tra coordinate 2D e 3D.
    
    Il modello pinhole proietta un punto 3D in coordinate camera (X, Y, Z)
    a coordinate 2D image (u, v) usando:
        u = fx * X/Z + cx
        v = fy * Y/Z + cy
    
    E inversamente, dato (u, v, Z) possiamo calcolare (X, Y, Z):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    
    def __init__(self, cam_k):
        """
        
        """
        self.fx, self.fy, self.cx, self.cy = cam_k[0], cam_k[4], cam_k[2], cam_k[5]
        
        # Camera intrinsics matrix K
        self.K = cam_k
    
    def project_3d_to_2d(self, points_3d):
        """
        Proietta punti 3D in coordinate camera a 2D image.
        
        Args:
            points_3d: (N, 3) array o tensor [X, Y, Z] in metri
            
        Returns:
            points_2d: (N, 2) array [u, v] in pixels
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
    
    def unproject_2d_to_3d(self, points_2d, depth):
        """
        Unproject punti 2D + depth a 3D in coordinate camera.
        
        Args:
            points_2d: (N, 2) array o tensor [u, v] in pixels
            depth: (N,) array o tensor [Z] in metri
            
        Returns:
            points_3d: (N, 3) array [X, Y, Z] in metri
        """
        is_torch = torch.is_tensor(points_2d)
        
        if is_torch:
            u, v = points_2d[:, 0], points_2d[:, 1]
            Z = depth
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            points_3d = torch.stack([X, Y, Z], dim=1)
        else:
            u, v = points_2d[:, 0], points_2d[:, 1]
            Z = depth
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            points_3d = np.stack([X, Y, Z], axis=1)
        
        return points_3d
    
    def compute_depth_from_bbox(self, bbox, object_diameter):
        """
        Calcola depth Z usando il diametro dell'oggetto e la dimensione del bbox.
        
        Formula pinhole: Z = (object_diameter * focal_length) / bbox_size_pixels
        
        Args:
            bbox: (N, 4) array o tensor [x1, y1, x2, y2] in pixels
            object_diameter: (N,) array o tensor diametro oggetto in mm
            
        Returns:
            depth: (N,) array depth Z in metri
        """
        is_torch = torch.is_tensor(bbox)
        
        if is_torch:
            # Calcola dimensione bbox (usiamo il max tra width e height come proxy del diametro)
            bbox_width = bbox[:, 2] - bbox[:, 0]
            bbox_height = bbox[:, 3] - bbox[:, 1]
            bbox_size = torch.maximum(bbox_width, bbox_height)  # max dimension
            
            # Z = (diameter_real * focal) / diameter_pixels
            # object_diameter è in mm, convertiamo in metri /1000
            # usiamo focal media fx
            depth = (object_diameter / 1000.0) * self.fx / bbox_size
        else:
            bbox_width = bbox[:, 2] - bbox[:, 0]
            bbox_height = bbox[:, 3] - bbox[:, 1]
            bbox_size = np.maximum(bbox_width, bbox_height)
            
            depth = (object_diameter / 1000.0) * self.fx / bbox_size
        
        return depth
    
    def get_intrinsics_matrix(self):
        """Ritorna la camera intrinsics matrix K."""
        return self.K
