import numpy as np
import cv2

# =============================================================================
# FUNZIONI DI DISEGNO
# =============================================================================

def draw_2d_bbox(img, box, color, label, thickness=3):
    """Draw bounding box on image with label."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def draw_3d_bbox_colored(img, R, t, K, obj_id, models_info, color=(255, 255, 0)):
    """Disegna il bounding box 3D proiettato sull'immagine 2D."""
    # Recupera info modello 3D
    info = models_info[obj_id]
    min_x, min_y, min_z = info['min_x'], info['min_y'], info['min_z']
    size_x, size_y, size_z = info['size_x'], info['size_y'], info['size_z']
    
    # 8 corners in mm
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z]
    ], dtype=np.float32)
    
    # Proiezione: t è in metri, convertiamo in mm per coerenza con corners_3d (o viceversa)
    # Qui convertiamo t in mm: t * 1000
    corners_cam = (R @ corners_3d.T).T + t * 1000.0
    
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    corners_2d = []
    for p in corners_cam:
        # p[2] è Z in mm. Deve essere > 0
        if p[2] > 0:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            corners_2d.append((u, v))
        else:
            return img # Bbox dietro la camera
    
    # Disegna linee
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # verticali
    ]
    for e in edges:
        if e[0] < len(corners_2d) and e[1] < len(corners_2d):
            cv2.line(img, corners_2d[e[0]], corners_2d[e[1]], color, 2)
    
    return img

def draw_axis_colored(img, R, t, K, scale=0.05, colors=None):
    """Disegna gli assi XYZ."""
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB default
    
    # Scale in metri (0.05 = 5cm)
    points_3d = np.array([
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ], dtype=np.float32)
    
    # t è già in metri
    points_cam = (R @ points_3d.T).T + t
    
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    points_2d = []
    for p in points_cam:
        if p[2] > 0:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            points_2d.append((u, v))
        else:
            points_2d.append(None)
    
    if all(p is not None for p in points_2d):
        origin = points_2d[0]
        cv2.line(img, origin, points_2d[1], colors[0], 3) # X
        cv2.line(img, origin, points_2d[2], colors[1], 3) # Y
        cv2.line(img, origin, points_2d[3], colors[2], 3) # Z
    
    return img