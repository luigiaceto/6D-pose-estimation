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

            noise = torch.randn_like(depth_tensor) * 0.003  # +/- 3mm di rumore
            mask = torch.rand_like(depth_tensor) > 0.03     # 3% dei pixel persi
            depth_tensor = (depth_tensor + noise) * mask

        depth_tensor = depth_tensor.unsqueeze(0) # Aggiungi canale: (1, 224, 224)
        
        return depth_tensor

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        translation, rotation, quaternion, bbox_gt, obj_id, bbox_gt_YOLO = self.load_6d_pose(folder_id, sample_id)
        
        img, img_w, img_h = self.load_image(
            str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        )

        bbox_jittered = self.apply_bbox_jitter(tuple(bbox_gt), img_w, img_h)
        bbox_jittered_YOLO = self.compute_yolo_bbox(bbox_jittered)
        bbox_jittered_dims = bbox_jittered_YOLO[2:4] # quanto il crop jitterato occupa dell'immagine originale
        
        cropped_img = self._crop_and_pad_image(img, bbox_jittered)
        cropped_img = self.transform_crop(cropped_img)
        
        cropped_depth = self.load_cropped_depth(folder_id, sample_id, bbox_jittered)
        
        # centro del bbox jitterato (verrà regredito al centro reale dell'oggetto
        # grazie all'utilizzo di δu e δv predetti dalla rete).
        # N.B. a test time non c'è jitter, quindi bbox_jittered == bbox_base.
        bbox_jittered_cx_pixel = bbox_jittered[0] + bbox_jittered[2] / 2.0
        bbox_jittered_cy_pixel = bbox_jittered[1] + bbox_jittered[3] / 2.0
        
        # Path assoluto depth map originale per Direct Read
        depth_path_str = str(self.dataset_root / "data" / f"{folder_id:02d}" / "depth" / f"{sample_id:04d}.png")
        
        return {
            # sample  
            "sample_id": torch.tensor([folder_id, sample_id]),
            "cropped_img": cropped_img,
            "cropped_depth": cropped_depth,
            "rgb": self.transform_img(img),                                         # usato solo in visualizzazione
            "bbox_base": torch.tensor(bbox_jittered, dtype=torch.float32),
            "bbox_YOLO": torch.tensor(bbox_gt_YOLO),                                # usato solo in visualizzazione
            "bbox_dims": torch.tensor(bbox_jittered_dims, dtype=torch.float32),
            "bbox_center_pixel": torch.tensor(
                [
                    bbox_jittered_cx_pixel,
                    bbox_jittered_cy_pixel
                ],
                dtype=torch.float32
            ),

            # label/ground truth
            "obj_id": torch.tensor(obj_id),
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "quaternion": torch.tensor(quaternion)
        }