import torch
import numpy as np
from PIL import Image
from data.CustomDatasetPose import CustomDatasetPose


class RGBDDatasetPose(CustomDatasetPose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trasformazione specifica per depth: Normalizzazione semplice
        # Non usiamo ImageNet mean/std per la depth perchè non ha senso fisico qui
    
    def load_cropped_depth(self, folder_id, sample_id, bbox):
        """
        Carica immagine depth, la croppa e la normalizza.
        CRITICO: Usa lo stesso bbox (già jitterato se in training) per mantenere allineamento con RGB.
        
        IMPORTANTE: Usa Image.NEAREST per preservare valori di profondità esatti sui bordi.
        """
        # path della depth image 'dataset_root/data/01/depth/0000.png'
        depth_path = self.dataset_root / "data" / f"{folder_id:02d}" / "depth" / f"{sample_id:04d}.png"
        
        # Carica immagine a 16-bit (valori in millimetri)
        depth_img = Image.open(str(depth_path))
        
        # Usa BILINEAR anche per la depth per aiutare la CNN
        square_depth = self._crop_and_pad_image(depth_img, bbox, resample=Image.BILINEAR)
        
        # Conversione in Tensor: da (H, W) a (1, H, W) e in metri
        depth_tensor = torch.tensor(np.array(square_depth), dtype=torch.float32)
        depth_tensor = depth_tensor / 1000.0 # Converti mm -> metri

        # in training applico data augmentation alla depth
        if self.split == 'train':
            noise = torch.randn_like(depth_tensor) * 0.005 # +/- 5mm di rumore
            mask = torch.rand_like(depth_tensor) > 0.10 # 10% dei pixel persi
            depth_tensor = (depth_tensor + noise) * mask

        depth_tensor = depth_tensor.unsqueeze(0) # Aggiungi canale: (1, 224, 224)
        
        return depth_tensor

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        
        # 1. Carica bbox ground truth
        bbox_base = np.array(self.ground_truths[folder_id][sample_id]['obj_bb'], dtype=np.float32)
        
        # 2. CRITICO: Applica jitter UNA SOLA VOLTA - stesso bbox per RGB e Depth
        img_path = str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        bbox_jittered = self.apply_bbox_jitter(tuple(bbox_base), img_w, img_h)
        
        # 3. Crop RGB con bbox jitterato (usa metodo PURO della classe padre)
        square_img = self._crop_and_pad_image(img, bbox_jittered)
        cropped_img = self.transform_crop(square_img)
        
        # 4. Crop Depth con STESSO bbox jitterato (allineamento garantito!)
        depth_tensor = self.load_cropped_depth(folder_id, sample_id, bbox_jittered)
        
        # 5. Carica ground truth
        pose = self.ground_truths[folder_id][sample_id]
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32) / 1000.0
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        quaternion = np.array(pose['quaternion'], dtype=np.float32)
        obj_id = np.array(pose['obj_id'], dtype=np.float32)
        
        # 6. Carica RGB full
        img_tensor = self.transform_img(img)
        
        # 7. Calcola bbox YOLO format (usa metodo centralizzato del padre)
        bbox_YOLO = self.compute_yolo_bbox(bbox_base)
        
        # 8. Bbox center in pixels (per pinhole)
        # CRITICO: Usa bbox_jittered per coerenza geometrica con il crop!
        # Il network vede un'immagine croppata con bbox_jittered, quindi il 
        # reference point deve essere il centro di QUELLO, non di bbox_base.
        # Durante inference (val/test) non c'è jitter quindi bbox_jittered == bbox_base.
        cx_pixel = bbox_jittered[0] + bbox_jittered[2] / 2.0
        cy_pixel = bbox_jittered[1] + bbox_jittered[3] / 2.0
        
        return {
            "sample_id": torch.tensor([folder_id, sample_id]),
            "cropped_img": cropped_img,
            "cropped_depth": depth_tensor,
            "rgb": img_tensor,
            "obj_id": torch.tensor(obj_id),
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "quaternion": torch.tensor(quaternion),
            "bbox_base": torch.tensor(bbox_base, dtype=torch.float32),
            "bbox_YOLO": torch.tensor(bbox_YOLO),
            "bbox_center_pixel": torch.tensor([cx_pixel, cy_pixel], dtype=torch.float32)
        }