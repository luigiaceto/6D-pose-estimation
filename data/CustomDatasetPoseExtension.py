import torch
import numpy as np
from PIL import Image
from data.CustomDatasetPose import CustomDatasetPose


class RGBDDatasetPose(CustomDatasetPose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_cropped_depth(self, folder_id, sample_id, bbox):
        """
        Carica immagine depth, la croppa e la normalizza.
        CRITICO: Usa lo stesso bbox (già jitterato se in training) per mantenere allineamento con RGB.
        
        IMPORTANTE: Usa letterbox padding (come RGB) per preservare aspect ratio dell'oggetto.
        Resize con NEAREST per preservare valori di profondità esatti.
        """
        # path della depth image 'dataset_root/data/01/depth/0000.png'
        depth_path = self.dataset_root / "data" / f"{folder_id:02d}" / "depth" / f"{sample_id:04d}.png"
        
        # Carica immagine a 16-bit (valori in millimetri)
        depth_img = Image.open(str(depth_path))
        
        # Usa letterbox padding (come RGB) + resize con NEAREST per preservare valori metrici
        square_depth = self._crop_and_pad_image(depth_img, bbox, resample=Image.NEAREST)
        
        # Conversione in Tensor: da (H, W) a (1, H, W) e in metri
        depth_tensor = torch.tensor(np.array(square_depth), dtype=torch.float32)
        depth_tensor = depth_tensor / 1000.0 # Converti mm -> metri

        # in training applico data augmentation alla depth
        if self.split == 'train':
            # Simula errori di calibrazione minimi e previene overfitting sui valori esatti
            scale_factor = torch.empty(1).uniform_(0.99, 1.01).item()
            depth_tensor = depth_tensor * scale_factor

            noise = torch.randn_like(depth_tensor) * 0.003 # +/- 5mm di rumore
            mask = torch.rand_like(depth_tensor) > 0.03 # 10% dei pixel persi
            depth_tensor = (depth_tensor + noise) * mask

        depth_tensor = depth_tensor.unsqueeze(0) # Aggiungi canale: (1, 224, 224)
        
        return depth_tensor

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        
        # carica bbox ground truth
        bbox_base = np.array(self.ground_truths[folder_id][sample_id]['obj_bb'], dtype=np.float32)
        
        # CRITICO: Applica jitter UNA SOLA VOLTA - stesso bbox per RGB e Depth
        img_path = str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        bbox_jittered = self.apply_bbox_jitter(tuple(bbox_base), img_w, img_h)
        
        # crop RGB con bbox jitterato
        square_img = self._crop_and_pad_image(img, bbox_jittered)
        cropped_img = self.transform_crop(square_img)
        
        # crop Depth con STESSO bbox jitterato usato per croppare l'immagine RGB
        depth_tensor = self.load_cropped_depth(folder_id, sample_id, bbox_jittered)
        
        # carica ground truth
        pose = self.ground_truths[folder_id][sample_id]
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32) / 1000.0
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        quaternion = np.array(pose['quaternion'], dtype=np.float32)
        obj_id = np.array(pose['obj_id'], dtype=np.float32)
        
        # carica immagine RGB completa
        img_tensor = self.transform_img(img)
        
        # calcola bbox in fomatio YOLO
        bbox_YOLO = self.compute_yolo_bbox(bbox_base)

        # quanto il crop (bbox) occupa dell'immagine originale
        bbox_dims = bbox_YOLO[2:4]
        
        # DA METTERE ANCHE NELLA BASELINE ???
        # bbox center in pixels (per pinhole).
        # CRITICO: Usa bbox_jittered per coerenza geometrica con il crop!
        # Il network vede un'immagine croppata con bbox_jittered, quindi il 
        # reference point deve essere il centro di QUELLO, non di bbox_base.
        # Durante inference (val/test) non c'è jitter quindi bbox_jittered == bbox_base.
        cx_pixel = bbox_jittered[0] + bbox_jittered[2] / 2.0
        cy_pixel = bbox_jittered[1] + bbox_jittered[3] / 2.0
        
        return {
            # sample  
            "sample_id": torch.tensor([folder_id, sample_id]),
            "cropped_img": cropped_img,
            "cropped_depth": depth_tensor,
            "rgb": img_tensor,

            # label/ground truth
            "obj_id": torch.tensor(obj_id),
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "quaternion": torch.tensor(quaternion),
            "bbox_base": torch.tensor(bbox_base, dtype=torch.float32),
            "bbox_YOLO": torch.tensor(bbox_YOLO),
            "bbox_dims": torch.tensor(bbox_dims, dtype=torch.float32),
            "bbox_center_pixel": torch.tensor([cx_pixel, cy_pixel], dtype=torch.float32)
        }