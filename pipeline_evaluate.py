"""  
Funzioni di visualizzazione per 6D pose.
"""

from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
import yaml
import os
from ultralytics import YOLO

from models.ResNetPose import ResNetPose, quaternion_to_rotation_matrix
from models.PinholeCamera import PinholeCamera
import torchvision.transforms as transforms
from utils.data_exploration import get_class_names
from baseline_evaluate import compute_add_metric, compute_add_rotation_only, compute_add_s_metric, compute_add_s_rotation_only, compute_rotation_error, compute_translation_error
from torch.utils.data import DataLoader
from baseline_evaluate import load_model_points


def evaluate_pipeline_batch1(
    test_dataset,
    cam_k,
    yolo_checkpoint,
    pose_checkpoint,
    device="cuda",
    img_mean=[0.485, 0.456, 0.406],
    img_std=[0.229, 0.224, 0.225]
):
    # Load models
    yolo = YOLO(yolo_checkpoint)

    checkpoint = torch.load(
        pose_checkpoint,
        map_location=device,
        weights_only=False
        )
    pose_model = ResNetPose().to(device)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    # Pinhole camera
    pinhole = PinholeCamera(cam_k)

    # image transform
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=img_mean,
                std=img_std
            )
        ])

    object_diameters = test_dataset.get_object_diameters()
    symmetric_objects = [10, 11]

    # mapping between obj_id and yolo class
    linemod_ids = [1,2,4,5,6,8,9,10,11,12,13,14,15]

    objid_to_yolo = {obj_id: i for i, obj_id in enumerate(linemod_ids)}
    yolo_to_objid = {i: obj_id for i, obj_id in enumerate(linemod_ids)}

    test_loader= DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) 

    #Metrics 
    rot_err=[]
    trans_err=[]
    accuracy=[] 
    add_err=[]
    adds_err=[]
    add_rot_only=[]
    all_add=[]
    all_diameters=[]

    # collect metrics per classe
    per_class_metrics= defaultdict(list)

    for batch in tqdm(test_loader):
        
        rgb_tensor = batch["rgb"][0]  # tensor 3xHxW
        gt_R = batch["rotation"][0].cpu().numpy()
        gt_t = batch["translation"][0].cpu().numpy()
        gt_obj_id = int(batch["obj_id"][0])
        
        rgb= rgb_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
        H, W, _ = rgb.shape
        rgb_yolo = (rgb_tensor * 255).byte()   # ora uint8 3xHxW
        rgb_yolo = rgb_yolo.permute(1,2,0)     # HxWx3
        rgb_yolo = rgb_yolo.cpu().numpy()      # numpy array pronto per YOLO
        results = yolo(rgb_yolo, verbose=False)[0]
        boxes= results.boxes
        
        yolo_cls_gt = objid_to_yolo[gt_obj_id]
        valid_idx = np.where(cls == yolo_cls_gt)[0]
        
        #valid = boxes[0]  # oppure class_map[obj_id]
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        
        valid_idx = np.where(cls == yolo_cls_gt)[0]
        if len(valid_idx) == 0:
            print("YOLO non rileva l'oggetto GT")
            continue
        
        i = valid_idx[conf[valid_idx].argmax()]
        x_c, y_c, w, h = boxes.xywh[i].cpu().numpy()

        
        # =============================
        # 3. CROP
        # =============================
        x_min = int(x_c - w/2)
        y_min = int(y_c - h/2)
        x_max = int(x_c + w/2)
        y_max = int(y_c + h/2)
                    
        # Crop handling
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(rgb.shape[1], x_max)
        y_max = min(rgb.shape[0], y_max)
                    
        cropped = rgb[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            print("Empty crop")
            continue
                    
        
        cropped_pil = Image.fromarray(cropped)
        
        # =============================
        # 4. LETTERBOX + RESIZE (TRAINING-COMPATIBLE)
        # =============================
        # =================================================================
        w_crop, h_crop = cropped_pil.size
        max_dim = max(w_crop, h_crop)
        
        # Creiamo una nuova immagine quadrata nera
        square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        
        # Calcoliamo offset per centrare l'immagine
        offset_x = (max_dim - w_crop) // 2
        offset_y = (max_dim - h_crop) // 2
                    
        # Incolliamo l'immagine al centro
        square_img.paste(cropped_pil, (offset_x, offset_y))
                    
        # Resize alla dimensione di input della ResNet (224x224)
        # Questo è fondamentale perché la rete aspetta questa dimensione fissa
        final_input = square_img.resize((224, 224), Image.BILINEAR)
                    
        crop_tensor = transform(final_input).unsqueeze(0).to(device)
        
        
        # # =============================
        # # 5. RESNET (ROTATION)
        # # =============================
        with torch.no_grad():
            pred_q = pose_model(crop_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q)[0].cpu().numpy()
        
        # # =============================
        # # 6. PINHOLE (TRANSLATION)
        # # =============================
        diameter = object_diameters[gt_obj_id]
        
        bbox_xyxy = torch.tensor([[x_min,y_min,x_max,y_max]], device=device)
        center_2d = torch.tensor([[(x_min+x_max)/2, (y_min+y_max)/2]], device=device)
        depth = pinhole.compute_depth_from_bbox(
            bbox_xyxy,
            torch.tensor([diameter], device=device)
        )  
        pred_t = pinhole.unproject_2d_to_3d(center_2d, depth)[0].cpu().numpy()
        
        # =============================
        # 7. METRICS
        # =============================
        
        model_points = load_model_points(test_dataset.dataset_root, gt_obj_id)
        
        all_diameters.append(object_diameters[gt_obj_id])              
        #rotation error
        r_err=compute_rotation_error(pred_R, gt_R)
        rot_err.append(r_err)
        #traslation error
        t_err=compute_translation_error(pred_t, gt_t)
        trans_err.append(t_err)
        
        if gt_obj_id in symmetric_objects:
            add_s = compute_add_s_metric(
                    pred_R, pred_t, gt_R, gt_t, model_points
                    )

            # TOTAL ADD/ADD-S error
            all_add.append(add_s * 100)  # Per calcolo complessivo
            
            # ADD/ADD-s rot only
            add_s_rotation_only = compute_add_s_rotation_only(
                pred_R, gt_R, model_points
                )
            add_rot_only.append(add_s_rotation_only * 100)
            # PER CLASS
            per_class_metrics[gt_obj_id].append({ 'rotation': r_err, 'translation': t_err, 'add': add_s * 100, 'add_rotation_only': add_s_rotation_only * 100 })
        else:
            add = compute_add_metric(
                pred_R, pred_t, gt_R, gt_t, model_points
                )
        
            # TOTAL ADD/ADD-S error
            all_add.append(add * 100)  # m -> cm
        
            # ADD/ADD-s rot only
            add_rotation_only = compute_add_rotation_only(
                pred_R, gt_R, model_points
                )
            add_rot_only.append(add_rotation_only * 100)
            # PER CLASS
            per_class_metrics[gt_obj_id].append({ 'rotation': r_err, 'translation': t_err, 'add': add * 100, 'add_rotation_only': add_rotation_only * 100 })



    per_class_results=[]
    for class_id, metrics in per_class_metrics.items():
        if len(metrics) == 0:
            continue

        class_rot_errors = np.array([m['rotation'] for m in metrics])
        class_trans_errors = np.array([m['translation'] for m in metrics])
        class_add_errors = np.array([m['add'] for m in metrics])
        class_add_rotation_only_errors = np.array([m['add_rotation_only'] for m in metrics])
            
        # accuracy @ 10% diameter
        class_diameter_cm = object_diameters[class_id] / 10.0
        class_threshold = 0.1 * class_diameter_cm
        class_accuracy = np.mean(class_add_errors < class_threshold) * 100
            
        per_class_results.append({
            'class_id': class_id,
            'num_samples': len(metrics),
            'accuracy_10p': class_accuracy,
            'rot_mean': class_rot_errors.mean(),
            'trans_mean': class_trans_errors.mean(),
            'add_mean': class_add_errors.mean(),
            'add_rot_only_mean': class_add_rotation_only_errors.mean(),
        })

    # media di tutte le classi
    # Converti diametri da mm a cm per confronto
    all_add_np = np.array(all_add)
    all_diameters_np = np.array(all_diameters)

    all_diameters_cm = all_diameters_np / 10.0
    
    # Accuracy @ 10% diameter (metrica standard)
    threshold_10 = all_diameters_cm * 0.1
    accuracy.append( np.mean(all_add_np < threshold_10) * 100)

    per_class_results.append({
            'class_id': 'ALL',
            'num_samples': len(all_add),
            'accuracy_10p': np.mean(accuracy),
            'rot_mean': np.mean(rot_err),
            'trans_mean': np.mean(trans_err),
            'add_mean': np.mean(all_add),
            'add_rot_only_mean': np.mean(add_rot_only),
            })
    print_evaluation_results_table(per_class_results)  