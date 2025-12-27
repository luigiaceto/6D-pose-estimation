"""  
Funzioni di visualizzazione per 6D pose.
"""

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
    device="cuda"
):
    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------
    yolo = YOLO(yolo_checkpoint)

    #checkpoint = torch.load(pose_checkpoint, map_location=device)
    
    checkpoint = torch.load(
    pose_checkpoint,
    map_location=device,
    weights_only=False
    )

    pose_model = ResNetPose().to(device)
    pose_model.load_state_dict(checkpoint["model_state_dict"])
    pose_model.eval()

    pinhole = PinholeCamera(cam_k)

    object_diameters = test_dataset.get_object_diameters()
    symmetric_objects = [10, 11]

    # test dataloader with batch size=1
    test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    all_rot_err = []
    all_trans_err = []
    all_add = []

    # --------------------------------------------------
    # LOOP DATASET
    # --------------------------------------------------
    for batch in tqdm(test_loader):

        # =============================
        # 1. INPUT & GT 
        # =============================
        rgb_tensor = batch["rgb"][0]
        obj_id = int(batch["obj_id"][0])
        gt_R = batch["rotation"][0].cpu().numpy()
        gt_t = batch["translation"][0].cpu().numpy()

    

        # =============================
        # 2. YOLO
        # =============================
        rgb_yolo = (rgb_tensor * 255).byte()   # ora uint8 3xHxW
        rgb_yolo = rgb_yolo.permute(1,2,0)     # HxWx3
        rgb_yolo = rgb_yolo.cpu().numpy()      # numpy array pronto per YOLO
        results = yolo(rgb_yolo, verbose=False)[0]

        if len(results.boxes) == 0:
            print("YOLO non rileva boxes, No detection")
            continue
        
        rgb= rgb_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
        H, W, _ = rgb.shape

        # prendi bbox della classe di riferimento

        boxes = results.boxes
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        # probabilmente predice solo un box ed è sufficiente fare restults.boxes[0]
        valid = boxes[0]  # oppure class_map[obj_id]

        if len(valid) == 0:
            print("YOLO non rileva boxes della classe richiesta, No detection")
            continue

        
        x_c, y_c, w, h = valid.xywh[0].cpu().numpy()
        
        # =============================
        # 3. CROP
        # =============================
        x_min = int(x_c - w/2)
        y_min = int(y_c - h/2)
        x_max = int(x_c + w/2)
        y_max = int(y_c + h/2)

        x_min, y_min = max(0,x_min), max(0,y_min)
        x_max, y_max = min(W,x_max), min(H,y_max)


        crop = rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            print("Crop vuoto, skipping")
            continue

        # =============================
        # 4. LETTERBOX + RESIZE (TRAINING-COMPATIBLE)
        # =============================
        crop_pil = Image.fromarray(crop.astype(np.uint8))

        w0, h0 = crop_pil.size
        S = max(w0, h0)
        square = Image.new("RGB", (S,S))
        square.paste(crop_pil, ((S-w0)//2, (S-h0)//2))
        square = square.resize((224,224), Image.BILINEAR)

        crop_tensor = transforms.functional.to_tensor(square)
        crop_tensor = transforms.functional.normalize(
            crop_tensor,
            mean=checkpoint["image_mean"],
            std=checkpoint["image_std"]
        )
        crop_tensor = crop_tensor.unsqueeze(0).to(device)

        # =============================
        # 5. RESNET (ROTATION)
        # =============================
        with torch.no_grad():
            pred_q = pose_model(crop_tensor)
            pred_R = quaternion_to_rotation_matrix(pred_q)[0].cpu().numpy()

        # =============================
        # 6. PINHOLE (TRANSLATION)
        # =============================
        diameter = object_diameters[obj_id]

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
        rot_err = compute_rotation_error(pred_R, gt_R)
        trans_err = compute_translation_error(pred_t, gt_t)

        model_pts = load_model_points(test_dataset.dataset_root, obj_id)

        if obj_id in symmetric_objects:
            add = compute_add_s_metric(pred_R, pred_t, gt_R, gt_t, model_pts)
        else:
            add = compute_add_metric(pred_R, pred_t, gt_R, gt_t, model_pts)

        all_rot_err.append(rot_err)
        all_trans_err.append(trans_err)
        all_add.append(add * 100)

    # =============================
    # SUMMARY
    # =============================
    print(f"Mean Rot Error:  {np.mean(all_rot_err):.2f}°")
    print(f"Mean Trans Err: {np.mean(all_trans_err):.2f} cm")
    print(f"Mean ADD:       {np.mean(all_add):.2f} cm")
