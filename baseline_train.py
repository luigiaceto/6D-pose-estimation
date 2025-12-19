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
import numpy as np
from tqdm import tqdm
import yaml

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from models.losses import PoseLoss


def train(
    train_dataset,
    train_loader,
    val_loader,
    cam_k,
    checkpoint_dir='./checkpoints',
    epochs=50,
    lr=1e-4,
    weight_dacay=1e-5,
    device='cuda',
    freeze_epochs=5, # epoche dopo le quali scongelare la backbone (pretrainata)
):
    """
    Training del modello di pose estimation.
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pinhole = PinholeCamera(cam_k)
    
    # Get object diameters from dataset
    object_diameters = train_dataset.get_object_diameters()
    
    # Model
    model = ResNetPose().to(device)
    
    # Freeze backbone inizialmente
    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen per {freeze_epochs} epochs")
    
    # lambda_translation=0.0 perché translation è calcolata geometricamente, non da ResNet!
    criterion = PoseLoss(lambda_rotation=1.0, lambda_translation=0.0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr, # iniziale
        weight_decay=weight_dacay # regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( # agisce sull'optimizer
        # tiene lr costante finchè la loss scende, poi lo dimezza
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Unfreeze backbone dopo freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.unfreeze_backbone()
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr/10, # lr ora più basso
                weight_decay=weight_dacay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
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
            # POTREBBE RALLENTARE IL TRAINING
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
                'scheduler_state_dict': scheduler.state_dict(),
                'lr': lr,
                'val_loss': avg_val_loss,
                'image_mean': train_dataset.image_mean,
                'image_std': train_dataset.image_std
            }, os.path.join(checkpoint_dir, 'best_pose_model.pt'))
            print(f"✓ Saved best model")
    
    print("\nTraining completed!")
