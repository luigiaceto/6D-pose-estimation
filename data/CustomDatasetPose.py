import os
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

IMG_WIDTH = 640
IMG_HEIGHT = 480

class CustomDatasetPose(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42, cam_K=None):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train', 'validation' or 'test'.
            train_ratio (float): Percentage of data used for training.
            seed (int): Random seed for reproducibility.
            camera intrinsics:
            image mean:
            image standard deviation:

        Carica e preprocessa i dati.
        Serve al modello di 6D pose estimation baseline (che usa solo immagini RGB).
        """
        from sklearn.model_selection import train_test_split
        
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.camera_intrinsics = [cam_K[0], cam_K[4], cam_K[2], cam_K[5]] # ci serve ???

        # Get list of all samples as (folder_id, sample_id)
        self.samples, self.folder_names = self.get_all_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {str(self.dataset_root)}. Check the dataset path and structure.")

        # Split dataset into [training set] and [validation set + test set]
        labels = [elem[0] for elem in self.samples]
        self.train_samples, self.val_test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed, stratify=labels
        )

        # split [validation set + test set] into [validation set] and [test set]
        labels = [elem[0] for elem in self.val_test_samples]
        self.val_samples, self.test_samples = train_test_split(
            self.val_test_samples, train_size=0.5, random_state=self.seed, stratify=labels
        )

        # Select the appropriate split
        if split == "train":
            self.samples = self.train_samples
        elif split == "validation":
            self.samples = self.val_samples
        else:
            self.samples = self.test_samples

        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

        # Define image transformations for the baseline
        if self.split == 'train':
            self.transform_img = transforms.ToTensor()

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
    
    def get_image_mean_std(self):
        return self.image_mean, self.image_std
    
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
        return self.transform_img(img)

    # APPLICARE ANCHE RANDOM JITTER AL BBOX GROUND TRUTH ???
    def load_cropped_image(self, img_path, bbox):
        """
        Load an RGB image, crop it, resize/pad and return it.
        """
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = bbox

        # crop iniziale dell'immagine secondo il ground truth bounding box
        cropped_img = img.crop((x, y, x+w, y+h)) # give as input the coordinates for left, top, right, bottom

        w_crop, h_crop = cropped_img.size
        max_dim = max(w_crop, h_crop)

        # creiamo una nuova immagine quadrata nera (o media dataset).
        # Background nero (0,0,0) va bene se normalizzi dopo
        square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

        # incolliamo l'immagine al centro (o in alto a sinistra, basta essere coerenti)
        offset_x = (max_dim - w_crop) // 2
        offset_y = (max_dim - h_crop) // 2
        square_img.paste(cropped_img, (offset_x, offset_y))
        
        # resize alla dimensione di input della ResNet (224x224).
        # Importante: ora che è quadrata, il resize non deforma l'oggetto!
        square_img = square_img.resize((224, 224), Image.BILINEAR)

        return self.transform_crop(square_img)

    def load_6d_pose(self, folder_id: int = None, sample_id: int = None):
        """
        Load the 6D pose (translation and rotation) for the object in this sample.
        """
        pose = self.ground_truths[folder_id][sample_id]
        
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)/1000.0  # [3] ---> (x,y,z) in meters
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # [3x3] ---> rotation matrix
        quaternion = np.array(pose['quaternion'], dtype=np.float32)  # [4] ---> quaternion
        bbox_base = np.array(pose['obj_bb'], dtype=np.float32) # [4] ---> x_min, y_min, width, height
        # bbox is top left corner and width and height info, YOLO needs center coordinates and width and height
        obj_id = np.array(pose['obj_id'], dtype=np.float32) # [1] ---> label
        
        cropped_img = self.load_cropped_image(str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png"), bbox_base)

        # compute initial center
        x_min, y_min, width, height = np.array(pose['obj_bb'], dtype=np.float32)
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # slip center to image bounds and adjust width/height accordingly
        if x_center < 0:
            width += 2 * x_center  # x_center is negative, subtract its absolute value * 2 from width
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

        # ensure width and height are not negative.
        # This is when bounding box is completely outside image (it should never happen)
        width = max(0, width)
        height = max(0, height)
        # store coordinates of the center and width and height of the bounding box normalized to the
        # image width=640 pixels and height=480 pixels
        bbox_YOLO = np.array([x_center/IMG_WIDTH, y_center/IMG_HEIGHT, width/IMG_WIDTH, height/IMG_HEIGHT], dtype=np.float32)

        return cropped_img, translation, rotation, quaternion, bbox_base, obj_id, bbox_YOLO

    def __len__(self):
        """
        Return the total number of samples in the selected split.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a dataset sample.
        """
        folder_id, sample_id = self.samples[idx] # both are integer

        img_path = str(self.dataset_root / "data" / f"{folder_id:02d}" / "rgb" / f"{sample_id:04d}.png")

        img = self.load_image(img_path)

        cropped_img, translation, rotation, quaternion, bbox_base, obj_id, bbox_YOLO = self.load_6d_pose(folder_id, sample_id)

        return {
            # sample
            "sample_id": torch.tensor(self.samples[idx]),
            "cropped_img": cropped_img, # input della ResNet50 della baseline
            "rgb": img,

            # label/ground truth
            "obj_id": torch.tensor(obj_id),
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "quaternion": torch.tensor(quaternion),
            "bbox_base": torch.tensor(bbox_base),
            "bbox_YOLO": torch.tensor(bbox_YOLO), # bounding box ground truth in formato yolo
        }
    