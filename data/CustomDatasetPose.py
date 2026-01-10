import os
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.pose_utils import IMG_HEIGHT, IMG_WIDTH


class CustomDatasetPose(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train' or 'test'.
            train_ratio (float): Percentage of data used for training (default 0.8 = 80%).
            seed (int): Random seed for reproducibility.
            camera intrinsics:
            image mean:
            image standard deviation:

        Carica e preprocessa i dati.
        Serve al modello di 6D pose estimation baseline (che usa solo immagini RGB).
        
        NOTE: Split 80/20 - Durante training si usa test set anche per validation.
        """
        from sklearn.model_selection import train_test_split
        
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # Get list of all samples as (folder_id, sample_id)
        self.samples, self.folder_names = self.get_all_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {str(self.dataset_root)}. Check the dataset path and structure.")

        # Split dataset into [training set] and [test set]
        labels = [elem[0] for elem in self.samples]
        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed, stratify=labels
        )

        # Select the appropriate split
        if split == "train":
            self.samples = self.train_samples
        else:
            self.samples = self.test_samples

        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

        # Define image transformations for the baseline
        if self.split == 'train':
            self.transform_img = transforms.ToTensor()

            # AUGMENTATION OTTIMIZZATA PER LINEMOD (Dataset Piccolo)
            self.transform_crop = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.image_mean,
                    std=self.image_std
                )
            ])
        else:
            # Validation/Test: Nessuna augmentation, solo resize e normalize
            self.transform_img = transforms.ToTensor()

            self.transform_crop = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.image_mean,
                    std=self.image_std
                )
            ])

        # store everything instead of opening each time file, this can speed up computation
        self.ground_truths = self.extract_ground_truth()
        
        # Load object info and extract diameters
        self.objects_info = self.load_obj_info()
        

    def get_samples_id(self):
        """
        Ritorna lista di tuple (numero cartella, numero immagine)
        """
        return self.samples

    def get_all_samples(self):
        """
        Retrieve the list of all available sample indices from all folders.
        """
        folder_names = []
        samples = []
        for folder_id in range(1, 16):  # Assuming folders are named 01 02 ... 15
            folder_path = str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb")
            if os.path.exists(folder_path):
                # get id of the images
                folder_names.append(folder_id)
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])  # store (folder_id, sample_id)

        return samples, folder_names
    
    def extract_ground_truth(self):
        ground_truth = {}
        for elem in self.folder_names:

            pose_file = str(self.dataset_root / f"{elem:02d}_gt.yml")

            with open(pose_file, 'r') as f:
                pose_data = yaml.load(f, Loader=yaml.CLoader)

            keys_to_extract = ['cam_t_m2c', 'cam_R_m2c', 'quaternion', 'obj_bb', 'obj_id']
            extracted_data = {}
            
            for key, value in pose_data.items():
                entry = value[0] # get first object of image, entry is a dictionary

                # extract desidered key
                extracted = {k: entry[k] for k in keys_to_extract if k in entry}

                # store in extracted_data
                extracted_data[key] = extracted # store image_id (int) and extracted value
            
            # store for each class all the extracted data
            ground_truth[elem] = extracted_data

        return ground_truth
    
    def load_obj_info(self):
        """
        Load YAML configuration files for object info for a specific folder.
        """

        objects_info_path = str(self.dataset_root / "models" / "models_info.yml")

        with open(objects_info_path, 'r') as f:
            objects_info = yaml.load(f, Loader=yaml.CLoader)

        return objects_info
    
    def get_object_diameters(self): 
        """
        Get the object diameters dictionary.
        """
        return {obj_id: info['diameter'] for obj_id, info in self.objects_info.items()}

    def load_image(self, img_path):
        """
        Load an RGB image.
        """
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        return img, img_w, img_h
    
    def compute_yolo_bbox(self, bbox_base):
        """
        Calcola bbox in formato YOLO normalizzato con gestione dei bordi.
        
        Args:
            bbox_base: (x_min, y_min, width, height) in pixels
        
        Returns:
            bbox_YOLO: (x_center_norm, y_center_norm, width_norm, height_norm) normalizzato [0,1]
        """
        x_min, y_min, width, height = bbox_base
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Gestione bbox parzialmente fuori dall'immagine
        # Se il centro è fuori, lo clippiamo e aggiustiamo width/height di conseguenza
        if x_center < 0:
            width += 2 * x_center  # x_center è negativo, sottraiamo
            x_center = 0
        elif x_center > IMG_WIDTH:
            width -= 2 * (x_center - IMG_WIDTH)
            x_center = IMG_WIDTH

        if y_center < 0:
            height += 2 * y_center
            y_center = 0
        elif y_center > IMG_HEIGHT:
            height -= 2 * (y_center - IMG_HEIGHT)
            y_center = IMG_HEIGHT

        # Assicurati che width/height non siano negativi
        width = max(0, width)
        height = max(0, height)
        
        # Normalizza rispetto alle dimensioni immagine
        bbox_YOLO = np.array([
            x_center / IMG_WIDTH,
            y_center / IMG_HEIGHT,
            width / IMG_WIDTH,
            height / IMG_HEIGHT
        ], dtype=np.float32)
        
        return bbox_YOLO
    
    def apply_bbox_jitter(self, bbox, img_width, img_height):
        """
        Applica random jitter al bbox per Data Augmentation.
        CRITICO: Questo metodo viene usato sia per RGB che per Depth per mantenere l'allineamento.
        
        Args:
            bbox: (x, y, w, h) - top-left corner and size
            img_width: larghezza immagine
            img_height: altezza immagine
        
        Returns:
            bbox_jittered: (x, y, w, h) dopo jittering
        """
        x, y, w, h = bbox
        
        if self.split != 'train':
            return bbox  # Nessun jitter in val/test
        
        if np.random.rand() < 0.2:
            return bbox

        # random Scale (zoom in/out del +/- 10%)
        scale_factor = np.random.uniform(0.9, 1.1)
        w_new = w * scale_factor
        h_new = h * scale_factor
        
        # random Shift (spostamento centro +/- 10%)
        center_x = x + w / 2
        center_y = y + h / 2
        
        shift_x = (np.random.rand() - 0.5) * 0.2 * w
        shift_y = (np.random.rand() - 0.5) * 0.2 * h
        
        center_x_new = center_x + shift_x
        center_y_new = center_y + shift_y
        
        # ricalcolo top-left corner
        x_new = center_x_new - w_new / 2
        y_new = center_y_new - h_new / 2
        
        # Se il box esce dai bordi, lo shiftiamo all'interno
        if x_new < 0:
            x_new = 0
        elif x_new + w_new > img_width:
            x_new = max(0, img_width - w_new)
            
        if y_new < 0:
            y_new = 0
        elif y_new + h_new > img_height:
            y_new = max(0, img_height - h_new)
        
        # Valida dimensioni minime (se il box è troppo grande per l'immagine, usa l'originale)
        if w_new > img_width or h_new > img_height or w_new < 1 or h_new < 1:
            return bbox  # fallback al bbox originale
            
        return (x_new, y_new, w_new, h_new)

    def _crop_and_pad_image(self, img, bbox, resample=Image.BILINEAR):
        """
        Metodo PURO per crop + letterbox padding + resize.
        Non applica jitter - usa il bbox fornito così com'è.
        
        Args:
            img: PIL Image
            bbox: (x, y, w, h)
                    - può essere già jitterato o no
            resample: Metodo di resampling per resize (default: BILINEAR)
                    - Image.BILINEAR: Per RGB (smooth interpolation)
                    - Image.NEAREST: Per Depth (preserva valori esatti)
        
        Returns:
            PIL Image preprocessata (224, 224)
        """
        x, y, w, h = bbox
        
        # Convertiamo in interi per PIL e facciamo il crop
        crop_rect = (
            int(x), 
            int(y), 
            int(x) + max(1, int(w)), 
            int(y) + max(1, int(h))
        )
        
        cropped_img = img.crop(crop_rect)

        # --- Letterbox Padding: Padding Quadrato + Resize ---
        w_crop, h_crop = cropped_img.size
        max_dim = max(w_crop, h_crop)

        # Creiamo immagine quadrata nera (RGB o grayscale dipende dall'input)
        mode = img.mode
        fill_value = 0
        square_img = Image.new(mode, (max_dim, max_dim), fill_value)

        # Incolliamo il crop al centro
        offset_x = (max_dim - w_crop) // 2
        offset_y = (max_dim - h_crop) // 2
        square_img.paste(cropped_img, (offset_x, offset_y))
        
        # Resize finale a 224x224 con metodo di resampling specificato
        square_img = square_img.resize((224, 224), resample)

        return square_img

    def load_cropped_image(self, img_path, bbox):
        """
        Load an RGB image, crop it, resize/pad and return it.
            
        MODIFICA: Applica random jitter al bbox durante il training per simulare 
        l'imperfezione di YOLO e rendere la rete più robusta.
        """
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
            
        # Applica jittering (metodo centralizzato)
        bbox_jittered = self.apply_bbox_jitter(bbox, img_w, img_h)
            
        # Crop usando il metodo puro (condiviso con la classe figlia)
        square_img = self._crop_and_pad_image(img, bbox_jittered)

        return self.transform_crop(square_img)

    def load_6d_pose(self, folder_id: int = None, sample_id: int = None):
        """
        Load the 6D pose (translation and rotation) for the object in this sample.
        """
        pose = self.ground_truths[folder_id][sample_id]
        
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32) / 1000.0    # (x,y,z) in metri ---> dim 3
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # rotation matrix ---> dim 3x3
        quaternion = np.array(pose['quaternion'], dtype=np.float32)             # quaternion ---> dim 4
        bbox_gt = np.array(pose['obj_bb'], dtype=np.float32)                    # x_min, y_min, width, height ---> dim 4
        obj_id = np.array(pose['obj_id'], dtype=np.float32)                     # label ---> dim 1

        bbox_gt_YOLO = self.compute_yolo_bbox(bbox_gt)

        # Calcola bbox YOLO usando il metodo centralizzato
        bbox_gt_YOLO = self.compute_yolo_bbox(bbox_gt)

        return translation, rotation, quaternion, bbox_gt, obj_id, bbox_gt_YOLO

    def __len__(self):
        """
        Return the total number of samples in the selected split.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a dataset sample.
        """
        folder_id, sample_id = self.samples[idx]
        translation, rotation, quaternion, bbox_gt, obj_id, bbox_gt_YOLO = self.load_6d_pose(folder_id, sample_id)

        img, img_w, img_h = self.load_image(
            str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")
        )

        bbox_jittered = self.apply_bbox_jitter(tuple(bbox_gt), img_w, img_h)

        cropped_img = self._crop_and_pad_image(img, bbox_jittered)
        cropped_img = self.transform_crop(cropped_img)
        
        return {
            # sample
            "sample_id": torch.tensor(self.samples[idx]),
            "cropped_img": cropped_img,                     # input della ResNet50
            "rgb": self.transform_img(img),                 # usato solo in visualizzazione
            "bbox_base": torch.tensor(bbox_jittered),
            "bbox_YOLO": torch.tensor(bbox_gt_YOLO),        # usato solo in visualizzazione e da YOLO per costruire il suo dataset

            # label/ground truth
            "obj_id": torch.tensor(obj_id),
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "quaternion": torch.tensor(quaternion)
        }
    