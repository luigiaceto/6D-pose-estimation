"""
Collate functions che prendono la lista di samples e lo trasforma
in un unico tensore pronto per il modello.
Non possiamo usare quella di default di DataLoader dato che i ritagli
hanno dimensioni diverse.
"""

import torch
import torch.nn.functional as F
from utils.init import set_device


def rgb_collate_fn(batch):
    """  
    Trova le dimensioni massime nel batch per fare padding.
    Utilizzata sia per YOLO che per il modello baseline di posa 6D.
    """

    max_H = max(item['cropped_img'].shape[1] for item in batch)
    max_W = max(item['cropped_img'].shape[2] for item in batch)
    
    padded_cropped_imgs = []
    paddings = []

    device = set_device()
    
    for item in batch:
        # --- symmetric padding for image ---
        img = item['cropped_img']
        _, H, W = img.shape
        pad_H = max_H - H
        pad_W = max_W - W

        # compute symmetric padding: (left, right, top, bottom)
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        # pad images by replicating the border pixels
        padded_img = F.pad(img, padding, mode='replicate')
        padded_cropped_imgs.append(padded_img)
        padding = torch.tensor([pad_left, pad_right, pad_top, pad_bottom])
        paddings.append(padding)

    batch_dict = {
        # sample
        "sample_id": torch.stack([item['sample_id'] for item in batch]).to(device),
        "rgb": torch.stack([item['rgb'] for item in batch]).to(device),
        "cropped_img": torch.stack(padded_cropped_imgs).to(device),
        "paddings":torch.stack(paddings).to(device),
        # label/ground truth
        "translation": torch.stack([item['translation'] for item in batch]).to(device),
        "rotation": torch.stack([item['rotation'] for item in batch]).to(device),
        "quaternion": torch.stack([item['quaternion'] for item in batch]).to(device),
        "bbox_base": torch.stack([item['bbox_base'] for item in batch]).to(device),
        "bbox_YOLO": torch.stack([item['bbox_YOLO'] for item in batch]).to(device), # utilizzata per YOLO
        "obj_id": torch.stack([item['obj_id'] for item in batch]).to(device),
    }

    return batch_dict

def rgbd_collate_fn(batch):
    """
    Usato solo per il modello RGB+depth per la stima di posa 6D.
    """
    max_H = max(item['cropped_img'].shape[1] for item in batch)
    max_W = max(item['cropped_img'].shape[2] for item in batch)     
    padded_cropped_imgs = []
    padded_depths = []  # Nuova lista per le depth map
    paddings = []     
    device = set_device()

    for item in batch:
        img = item['cropped_img']
        depth = item['depth']  # Recuperiamo la depth map (che ora ci aspettiamo sia un tensore [1, H, W])
    
        _, H, W = img.shape
        pad_H = max_H - H
        pad_W = max_W - W
    
        # Calcolo padding simmetrico
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
    
        padding = (pad_left, pad_right, pad_top, pad_bottom)
            
        # Padding Immagine RGB (replicate per evitare bordi netti)
        padded_img = F.pad(img, padding, mode='replicate')
        padded_cropped_imgs.append(padded_img)
    
        # Padding Depth Map (costante 0 per indicare "nessun dato/sfondo")
        # Assumiamo che 'depth' sia [1, H, W]
        padded_depth = F.pad(depth, padding, mode='constant', value=0)
        padded_depths.append(padded_depth)
    
        # Salviamo i valori di padding per poter eventualmente ricostruire l'originale
        padding_tensor = torch.tensor([pad_left, pad_right, pad_top, pad_bottom])
        paddings.append(padding_tensor)
 
    batch_dict = {
        # sample
        # "rgb": ... (non serve per il training)
        "sample_id": torch.stack([item['sample_id'] for item in batch]).to(device),
        "paddings": torch.stack(paddings).to(device),
        "cropped_img": torch.stack(padded_cropped_imgs).to(device),
        "depth": torch.stack(padded_depths).to(device),  # Stackiamo le depth map
        "camera_intrinsics": torch.stack([item['camera_intrinsics'] for item in batch]).to(device),
        # label/ground truth
        "translation": torch.stack([item['translation'] for item in batch]).to(device),
        "rotation": torch.stack([item['rotation'] for item in batch]).to(device),
        "quaternion": torch.stack([item['quaternion'] for item in batch]).to(device),
        "bbox_base": torch.stack([item['bbox_base'] for item in batch]).to(device),
        "obj_id": torch.stack([item['obj_id'] for item in batch]).to(device),
    }
     
    return batch_dict
