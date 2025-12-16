"""
Training script per 6D Pose Estimation.

Pipeline:
1. YOLO (già trainato) -> bounding box
2. Crop immagine
3. ResNet -> quaternion + centro 2D + depth
4. Pinhole model -> translation 3D da (centro 2D, depth)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from models.losses import PoseLoss
from data.CustomDatasetPose import CustomDatasetPose
from data.DataLoaderCollating import rgb_collate_fn


def train(
    dataset_root="./datasets/linemod/DenseFusion/Linemod_preprocessed",
    epochs=50,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    freeze_epochs=5,
    checkpoint_dir='./checkpoints'
):
    """
    Training del modello di pose estimation.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Camera intrinsics LINEMOD
    #TODO: hardcoded, cambiare in modo che li prenda da CustomDataset
    fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
    cam_K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32)
    
    # Pinhole camera model
    pinhole = PinholeCamera(fx, fy, cx, cy)
    
    # Dataset
    print("Loading datasets...")
    train_dataset = CustomDatasetPose(
        dataset_root=dataset_root,
        split='train',
        train_ratio=0.7,
        seed=42,
        device='cpu',
        cam_K=cam_K
    )
    
    val_dataset = CustomDatasetPose(
        dataset_root=dataset_root,
        split='validation',
        train_ratio=0.7,
        seed=42,
        device='cpu',
        cam_K=cam_K,
        img_mean=train_dataset.image_mean,
        img_std=train_dataset.image_std
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=rgb_collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=rgb_collate_fn, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get object diameters from dataset
    object_diameters = train_dataset.get_object_diameters()
    
    # Model
    model = ResNetPose(pretrained=True, dropout=0.3).to(device)
    
    # Freeze backbone inizialmente
    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen per {freeze_epochs} epochs")
    
    # Loss e optimizer
    # lambda_translation=0.0 perché translation è calcolata geometricamente, non da ResNet!
    criterion = PoseLoss(lambda_rotation=1.0, lambda_translation=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    IMG_WIDTH, IMG_HEIGHT = 640, 480
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Unfreeze backbone dopo freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=lr/10, weight_decay=1e-5)
            print("Backbone unfrozen")
        
        # Train
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            cropped_img = batch['cropped_img'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            gt_translation = batch['translation'].to(device)
            bbox_base = batch['bbox_base'].to(device)  # (B, 4) [x_min, y_min, width, height]
            obj_id = batch['obj_id'].to(device).long()  # (B,)
            
            # Forward: ResNet predice SOLO quaternion
            pred_quaternion = model(cropped_img)  # (B, 4)
            
            # Calcola translation 3D usando geometria:
            # 1. Converti bbox da [x, y, w, h] a [x1, y1, x2, y2]
            bbox_xyxy = torch.stack([
                bbox_base[:, 0],  # x1 = x
                bbox_base[:, 1],  # y1 = y
                bbox_base[:, 0] + bbox_base[:, 2],  # x2 = x + w
                bbox_base[:, 1] + bbox_base[:, 3]   # y2 = y + h
            ], dim=1)
            
            # 2. Centro del bbox (u, v) in pixels
            center_2d_pixels = torch.stack([
                (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,  # u = (x1 + x2) / 2
                (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2   # v = (y1 + y2) / 2
            ], dim=1)
            
            # 3. Ottieni diametro per ogni oggetto nel batch
            batch_diameters = torch.tensor(
                [object_diameters[int(oid)] for oid in obj_id.cpu()],
                device=device, dtype=torch.float32
            )
            
            # 4. Calcola depth Z usando pinhole formula
            depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
            
            # 5. Unproject (u,v,Z) -> (X,Y,Z)
            pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
            
            # Loss
            losses = criterion(
                pred_quaternion,
                pred_translation,
                gt_quaternion,
                gt_translation
            )
            
            loss = losses['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                cropped_img = batch['cropped_img'].to(device)
                gt_quaternion = batch['quaternion'].to(device)
                gt_translation = batch['translation'].to(device)
                bbox_base = batch['bbox_base'].to(device)
                obj_id = batch['obj_id'].to(device).long()
                
                # Forward
                pred_quaternion = model(cropped_img)
                
                # Calcola translation da bbox + diametro
                bbox_xyxy = torch.stack([
                    bbox_base[:, 0],
                    bbox_base[:, 1],
                    bbox_base[:, 0] + bbox_base[:, 2],
                    bbox_base[:, 1] + bbox_base[:, 3]
                ], dim=1)
                
                center_2d_pixels = torch.stack([
                    (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                    (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
                ], dim=1)
                
                batch_diameters = torch.tensor(
                    [object_diameters[int(oid)] for oid in obj_id.cpu()],
                    device=device, dtype=torch.float32
                )
                
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
                pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
                
                losses = criterion(
                    pred_quaternion,
                    pred_translation,
                    gt_quaternion,
                    gt_translation
                )
                
                val_losses.append(losses['total_loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'image_mean': train_dataset.image_mean,
                'image_std': train_dataset.image_std,
                'camera_params': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
            }, os.path.join(checkpoint_dir, 'best_pose_model.pt'))
            print(f"✓ Saved best model")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    train(
        dataset_root="./datasets/linemod/DenseFusion/Linemod_preprocessed",
        epochs=50,
        batch_size=16,
        lr=1e-4,
        device='cuda',
        freeze_epochs=5
    )
