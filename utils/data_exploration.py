import os
import yaml
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
from data.CustomDatasetPose import IMG_WIDTH, IMG_HEIGHT

def load_image(label: int, object: int):
    """
    Starting from 6DPose_Estimation plot image given label and objectId
    """
    img_path = f"./datasets/linemod/DenseFusion/Linemod_preprocessed/data/{label:02d}/rgb/{object:04d}.png"
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.show()

def load_depth_image(path: str = None):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()
    return img

def get_camera_intrinsics(dataset_root):
    """
    Estrae la matrice dei parametri intrinseci (K) dal primo file gt.yml disponibile.
    Si assume che la camera sia la stessa per tutto il dataset.
    """
    # Proviamo a leggere il file gt della prima cartella (01)
    # Linemod di solito ha file tipo '01_gt.yml' nella root
    target_file = os.path.join(dataset_root, "01_gt.yml")    
    if not os.path.exists(target_file):
        # Fallback: prova a cercare dentro data/01/ se la struttura è diversa
        target_file = os.path.join(dataset_root, "data", "01", "gt.yml")
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Impossibile trovare un file di ground truth per estrarre cam_K in {dataset_root}")
   
    with open(target_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
           
        # Prende il primo frame (chiave '0') e cerca 'cam_K'
        # Nota: In Linemod 'cam_K' è spesso salvato come lista di 9 elementi
        first_frame_data = data[0][0] # Chiave 0 (frame id), primo oggetto della lista
        
        if 'cam_K' in first_frame_data:
            cam_K = np.array(first_frame_data['cam_K'], dtype=np.float32).reshape(3, 3)
            return cam_K
        else:
            raise KeyError("La chiave 'cam_K' non è presente nel file YAML analizzato.")
    
def get_class_names():
    """
    Gets class names.
    """
    folder_names = []
    for folder_id in range(1, 16):
        folder_path = os.path.join('./datasets/linemod/DenseFusion/Linemod_preprocessed/data', f"{folder_id:02d}", "rgb")
        if os.path.exists(folder_path):
            folder_names.append(folder_id)
    return folder_names

# ci serve ???
def compare_rgb_mask_in_data(root, extensions={'.png', '.jpg', '.jpeg', '.bmp'}):
    """
    For each class in "root", compare files in folders "rgb" and "mask".
    Print files that are only in one of the folders.
    Se manca la maschera, spesso quel campione è inutilizzabile per certi task.

    Args:
        root (str): main path (f.i. "data/")
        extensions (set): file format to consider
    """
    classes = sorted(os.listdir(root))
    only_rgb= 0
    only_mask=0
    print(classes)

    for class_name in classes:
        class_path = os.path.join(root, class_name)
        rgb_path = os.path.join(class_path, 'rgb')
        mask_path = os.path.join(class_path, 'mask')

        if not os.path.isdir(rgb_path) or not os.path.isdir(mask_path):
            print(f"No rgb or mask folder in '{class_name}'")
            continue

        rgb_files = {f for f in os.listdir(rgb_path) if os.path.splitext(f)[1].lower() in extensions}
        mask_files = {f for f in os.listdir(mask_path) if os.path.splitext(f)[1].lower() in extensions}

        only_in_rgb = sorted(rgb_files - mask_files)
        only_in_mask = sorted(mask_files - rgb_files)

        if only_in_rgb:
            print(f"\nClass '{class_name}' — only in rgb:")
            for f in only_in_rgb:
                print(f"  {f}")
                only_rgb+=1

        if only_in_mask:
            print(f"\nClass '{class_name}' — only in mask:")
            for f in only_in_mask:
                print(f"  {f}")
                only_mask+=1

    print(f"Total files only in rgb: {only_rgb}")
    print(f"Total files only in mask: {only_mask}")

def load_dataset_distribution(counter_df, index_dict, number_classes):
    """
    Plots distribution of labels in training, validation and test set.
    """

    fig, axes = plt.subplots(1,3,figsize=(15,6),sharey=True)
    for index, column in enumerate(counter_df.columns):
        axes[index].barh([str(el) for el in index_dict.keys()], counter_df[column],color="orange", edgecolor='gray')
        axes[index].set_title(column.capitalize())
        # add line that represents the uniform distribution of the labels
        axes[index].axvline(x=1/number_classes, color="blue")
        axes[index].text(x=1/number_classes,y=-0.5,s=f"{1/number_classes: .5f}", color="blue")

    fig.supxlabel("Frequency")
    fig.supylabel("Labels")
    plt.subplots_adjust(left=0.07, wspace=0.1)
    plt.suptitle("Labels Distribution over the Training, Validation and Test sets")
    plt.savefig("./images/YOLO_dataset_distribution.png")
    plt.show()

def load_depth_patch(path: str = None, folder: str = None, imageId: str = None, image=None):
    """
    Plots image with bounding box.    
    """

    # Load the ground truth poses from the gt.yml file
    with open(f"{path}/datasets/linemod/DenseFusion/Linemod_preprocessed/data/{folder}/gt.yml", 'r') as f:
        pose_data = yaml.load(f, Loader=yaml.FullLoader)
    pose = pose_data[int(imageId)][0] # access image imageId (start counting from 0) and get first element (in case of multiple objects)

    bbox = np.array(pose['obj_bb'], dtype=np.float32) #[4]
    obj_id = np.array(pose['obj_id'], dtype=np.float32) #[1]

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Create a rectangle patch
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),  # (x, y)
        bbox[2],             # width
        bbox[3],             # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Optionally add object ID label (write a bit above the top left corner)
    ax.text(bbox[0], bbox[1] - 10, f'ID: {int(obj_id)}', color='yellow', fontsize=12, backgroundcolor='black')

    plt.axis('off')
    plt.show()

def plot_batch_data(train_loader, val_loader, test_loader):
    """
    Visualizza un intero batch di dati cine esce dal DataLoader (dopo trasformazioni e padding).
    """

    # Get one batch from the train loader (there are batch_size images)
    batch = next(iter(train_loader)) # it uses load_6d_pose, so one pose per object

    # Extract relevant data
    rgb_images = batch["rgb"]         # (B, 3, H, W)
    bboxes = batch["bbox_YOLO"]       # (B, 4) in pixel coords: x_center, y_center, width, height
    obj_ids = batch["obj_id"]         # (B,)

    # Convert to numpy and rearrange channels
    rgb_images = rgb_images.permute(0, 2, 3, 1).to(torch.device("cpu")).numpy()  # (B, H, W, 3)
    bboxes = bboxes.to(torch.device("cpu")).numpy()
    obj_ids = obj_ids.to(torch.device("cpu")).numpy()

    # Plot settings
    batch_size = rgb_images.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(batch_size):
        ax = axes[i]
        img = rgb_images[i]
        # each element is [x_center/IMG_WIDTH, y_center/IMG_HEIGHT, width/IMG_WIDTH, height/IMG_HEIGHT]
        x_center, y_center, width, height = bboxes[i]
        # remove normalization
        x_center = x_center*IMG_WIDTH
        y_center = y_center*IMG_HEIGHT
        width = width*IMG_WIDTH
        height = height*IMG_HEIGHT
        x_min = x_center-(width/2)
        y_min = y_center-(height/2)
        obj_id = obj_ids[i]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sample {i}")

        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min),   # (x_min, y_min)
            width,            # width
            height,           # height
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add object ID as label
        ax.text(
            x_min,
            y_min - 10,
            f'ID: {int(obj_id)}',
            color='yellow',
            fontsize=10,
            backgroundcolor='black'
        )

    # Hide unused axes if batch_size < cols * rows
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()