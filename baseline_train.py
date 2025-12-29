import os
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.ResNetPose import ResNetPose
from models.PinholeCamera import PinholeCamera
from models.losses import PoseLoss

def train(
    train_dataset,
    train_loader,
    val_loader,
    cam_k,
    checkpoint_dir='checkpoints',
    checkpoint_name='best_pose_model.pt',
    epochs=100,
    lr=1e-4,
    weight_decay=1e-6,
    device='cuda',
    freeze_epochs=0,
    warmup_epochs=3,
    resume_from_checkpoint=None
):
    """
    Training del modello di pose estimation (Versione Finale - Parallel Optimized).
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pinhole = PinholeCamera(cam_k)
    
    # Lookup table diametri
    object_diameters = train_dataset.get_object_diameters()
    max_obj_id = max(object_diameters.keys())
    diameter_lookup = torch.zeros(max_obj_id + 1, device=device, dtype=torch.float32)
    for obj_id, diameter in object_diameters.items():
        diameter_lookup[obj_id] = diameter
    
    # Model Setup
    model = ResNetPose().to(device)

    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen per {freeze_epochs} epochs")

    # PARALLEL GPU in caso siano disponibili più gpus (kaggle)
    if torch.cuda.device_count() > 1:
        print(f"Turbo Mode: Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Loss
    criterion = PoseLoss(lambda_rotation=1.0, lambda_translation=0.0)

    
    # Funzione helper per recuperare il modello "reale" (dentro o fuori DataParallel)
    def get_raw_model(m):
        return m.module if hasattr(m, 'module') else m

    raw_model = get_raw_model(model) 

    if freeze_epochs > 0:
        optimizer = optim.AdamW([
            {'params': raw_model.backbone.parameters(), 'lr': lr / 10, 'weight_decay': weight_decay},
            {'params': raw_model.fc_layers_r.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': raw_model.quaternion_head.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ])
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

    # Warmup
    warmup_scheduler = None
    if warmup_epochs > 0:
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # AMP Setup
    USE_AMP = True 
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint in caso si volesse rifar partire un training
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        
        get_raw_model(model).load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    print(f"Mixed Precision (AMP): {'ENABLED' if USE_AMP else 'DISABLED'}")
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        if epoch == freeze_epochs and freeze_epochs > 0:
            get_raw_model(model).unfreeze_backbone()
            print("Backbone unfrozen")
        
        # --- TRAINING PHASE ---
        model.train()
        train_losses = []
        
        if warmup_scheduler is not None and epoch < warmup_epochs:
            print(f"  [Warmup] LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        for batch in tqdm(train_loader, desc="Training"):
            cropped_img = batch['cropped_img'].to(device, non_blocking=True)
            gt_quaternion = batch['quaternion'].to(device, non_blocking=True)
            gt_translation = batch['translation'].to(device, non_blocking=True)
            bbox_base = batch['bbox_base'].to(device, non_blocking=True)
            obj_id = batch['obj_id'].to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                # 1. Prediction
                pred_quaternion = model(cropped_img) 
                
                # 2. Geometria
                bbox_xyxy = torch.stack([
                    bbox_base[:, 0], bbox_base[:, 1],
                    bbox_base[:, 0] + bbox_base[:, 2], bbox_base[:, 1] + bbox_base[:, 3]
                ], dim=1)
                
                center_2d_pixels = torch.stack([
                    (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                    (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
                ], dim=1)
                
                batch_diameters = diameter_lookup[obj_id]
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
                pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
                
                # 3. Loss
                losses = criterion(
                    pred_quaternion, pred_translation,
                    gt_quaternion, gt_translation,
                    class_ids=obj_id
                )
                loss = losses['total_loss']
            
            # Backward & Step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_losses = []
        val_rot_errors = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                cropped_img = batch['cropped_img'].to(device, non_blocking=True)
                gt_quaternion = batch['quaternion'].to(device, non_blocking=True)
                gt_translation = batch['translation'].to(device, non_blocking=True)
                bbox_base = batch['bbox_base'].to(device, non_blocking=True)
                obj_id = batch['obj_id'].to(device, non_blocking=True).long()
                
                # Forward
                pred_quaternion = model(cropped_img)
                
                # Calcoli geometrici
                bbox_xyxy = torch.stack([
                    bbox_base[:, 0], bbox_base[:, 1],
                    bbox_base[:, 0] + bbox_base[:, 2], bbox_base[:, 1] + bbox_base[:, 3]
                ], dim=1)
                center_2d_pixels = torch.stack([
                    (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                    (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
                ], dim=1)
                batch_diameters = diameter_lookup[obj_id]
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
                pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
                
                # Loss
                losses = criterion(
                    pred_quaternion, pred_translation,
                    gt_quaternion, gt_translation,
                    class_ids=obj_id
                )
                val_losses.append(losses['total_loss'].item())

                # Monitoraggio Errore Rotazione
                dot_prod = torch.abs(torch.sum(pred_quaternion * gt_quaternion, dim=1))
                dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
                angular_dist_rad = 2 * torch.acos(dot_prod)
                angular_dist_deg = torch.rad2deg(angular_dist_rad).mean().item()
                val_rot_errors.append(angular_dist_deg)
        
        avg_val_loss = np.mean(val_losses)
        avg_rot_error = np.mean(val_rot_errors)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Rot Error: {avg_rot_error:.2f}°")
        
        scheduler.step(avg_val_loss)
        
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Salviamo sempre il raw_model per compatibilità futura
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_raw_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'lr': lr,
                'val_loss': avg_val_loss,
                'rot_error': avg_rot_error
            }, str(Path(checkpoint_dir) / f"{checkpoint_name}"))
            print(f"✓ Saved best model (Err: {avg_rot_error:.2f}°)")
            
    print("\nTraining completed!")
    return model