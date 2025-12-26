import torch
import numpy as np
from PIL import Image
from data.CustomDatasetPose import CustomDatasetPose


class RGBDDatasetPose(CustomDatasetPose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trasformazione specifica per depth: Normalizzazione semplice
        # Non usiamo ImageNet mean/std per la depth perchè non ha senso fisico qui
    
    # normalizzare in range [0, 1] o altro ???
    def load_cropped_depth(self, folder_id, sample_id, bbox):
        """
        Carica immagine depth, la croppa e la normalizza.
        """
        # path della depth image 'dataset_root/data/01/depth/0000.png'
        depth_path = self.dataset_root / "data" / f"{folder_id:02d}" / "depth" / f"{sample_id:04d}.png"
        
        # Carica immagine a 16-bit (valori in millimetri)
        depth_img = Image.open(str(depth_path)) 
        
        # Crop usando lo stesso bbox dell'RGB
        x, y, w, h = bbox
        cropped_depth = depth_img.crop((x, y, x+w, y+h))
        
        # Padding e Resize a 224x224 (come fatto per RGB in CustomDatasetPose)
        w_crop, h_crop = cropped_depth.size
        max_dim = max(w_crop, h_crop)
        square_depth = Image.new('I', (max_dim, max_dim), 0) # 'I' per 16-bit integer
        
        offset_x = (max_dim - w_crop) // 2
        offset_y = (max_dim - h_crop) // 2
        square_depth.paste(cropped_depth, (offset_x, offset_y))
        
        # Resize (Nearest neighbor è meglio per la depth per non interpolare valori falsi, 
        # ma Bilinear va bene per CNN feature extraction)
        square_depth = square_depth.resize((224, 224), Image.BILINEAR)
        
        # Conversione in Tensor: da (H, W) a (1, H, W) e in metri
        depth_tensor = torch.tensor(np.array(square_depth), dtype=torch.float32)
        depth_tensor = depth_tensor / 1000.0 # Converti mm -> metri

        # in training applico data augmentation alla depth
        if self.split == 'train':
            noise = torch.randn_like(depth_tensor) * 0.003 # +/- 3mm di rumore
            mask = torch.rand_like(depth_tensor) > 0.02 # 2% dei pixel persi
            depth_tensor = (depth_tensor + noise) * mask

        depth_tensor = depth_tensor.unsqueeze(0) # Aggiungi canale: (1, 224, 224)
        
        return depth_tensor

    def __getitem__(self, idx):
        # dati base dalla classe padre
        data = super().__getitem__(idx)
        
        folder_id, sample_id = self.samples[idx]
        
        # Recupera il bbox ground truth per fare il crop.
        # Nota: bbox_base è [x_min, y_min, w, h]
        bbox_base = data['bbox_base'].numpy()
        
        # Carica depth processata
        depth_tensor = self.load_cropped_depth(folder_id, sample_id, bbox_base)
        data['cropped_depth'] = depth_tensor
        
        # IMPORTANTE: Aggiungiamo i centri del bbox in pixel per la formula Pinhole nel modello.
        # bbox_base[0] = x_min, bbox_base[2] = width
        cx_pixel = bbox_base[0] + bbox_base[2] / 2.0
        cy_pixel = bbox_base[1] + bbox_base[3] / 2.0
        data['bbox_center_pixel'] = torch.tensor([cx_pixel, cy_pixel], dtype=torch.float32)
        
        return data