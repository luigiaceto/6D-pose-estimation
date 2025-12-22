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
    checkpoint_name='best_pose_model.pt',
    epochs=50,
    lr=1e-4,
    weight_decay=1e-5,
    device='cuda',
    freeze_epochs=5,
    use_amp=False,
    use_cosine=False,
    warmup_epochs=0,
    use_add_loss=False  # ADD loss per oggetti simmetrici
):
    """
    Training del modello di pose estimation.
    
    Args:
        use_amp: Se True, usa Mixed Precision (1.5-2x speedup)
        use_cosine: Se True, usa Cosine Annealing invece di ReduceLROnPlateau
        warmup_epochs: Numero di epoche di warmup (consigliato 3)
        weight_decay: L2 regularization (corretto typo)
        use_add_loss: Se True, usa ADD loss invece di quaternion loss (per oggetti simmetrici)
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pinhole = PinholeCamera(cam_k)
    
    # Get object diameters from dataset and create GPU lookup tensor
    object_diameters = train_dataset.get_object_diameters()
    # Create a tensor lookup [obj_id] -> diameter (much faster than list comprehension!)
    max_obj_id = max(object_diameters.keys())
    diameter_lookup = torch.zeros(max_obj_id + 1, device=device, dtype=torch.float32)
    for obj_id, diameter in object_diameters.items():
        diameter_lookup[obj_id] = diameter
    
    # Carica model points per ADD loss (se richiesto)
    model_points_dict = {}  # {obj_id: tensor(N, 3)}
    if use_add_loss:
        print("Loading model points for ADD loss...")
        for obj_id in object_diameters.keys():
            model_points = train_dataset.get_model_points(obj_id).to(device)
            model_points_dict[obj_id] = model_points
        print(f"Loaded model points for {len(model_points_dict)} objects")
    
    # Model
    model = ResNetPose().to(device)
    
    # Setup loss: usa ADD loss se richiesto, altrimenti quaternion geodesic
    criterion = PoseLoss(
        lambda_rotation=1.0, 
        lambda_translation=0.0,
        use_add_loss=use_add_loss
    )
    
    if use_add_loss:
        print("⚠️  Using ADD-based loss (point-to-point distance) for symmetric objects")
    else:
        print("✅ Using Quaternion Geodesic loss (standard)")

    # Setup optimizer con parameter groups per freeze logic migliore
    # Invece di ricreare optimizer (perdi momentum), usiamo groups con requires_grad
    if freeze_epochs > 0:
        # Inizia con backbone frozen
        model.freeze_backbone()
        print(f"Backbone frozen per {freeze_epochs} epochs")
        
        # Optimizer con parameter groups: backbone ha LR più basso quando unfrozen
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': lr / 10, 'weight_decay': weight_decay},
            {'params': model.fc_layers_r.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': model.quaternion_head.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ])
    else:
        # No freeze: usa AdamW con weight_decay corretto
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    print(f"Using AdamW optimizer (decoupled weight decay) with lr={lr:.0e}, wd={weight_decay:.0e}")
    
    # Setup scheduler
    if use_cosine:
        # Cosine Annealing: smooth decay da lr iniziale a lr minimo
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        print(f"Using Cosine Annealing: {lr:.0e} → 1e-6 over {epochs} epochs")
    else:
        # ReduceLROnPlateau: dimezza lr quando loss si stabilizza
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        print("Using ReduceLROnPlateau scheduler")
    
    # Setup warmup scheduler
    warmup_scheduler = None
    if warmup_epochs > 0:
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        print(f"Warmup enabled for {warmup_epochs} epochs")
    
    # Setup Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed Precision (AMP) enabled - expect 1.5-2x speedup! 🚀")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Unfreeze backbone dopo freeze_epochs (senza ricreare optimizer)
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.unfreeze_backbone()
            print("Backbone unfrozen - parameter groups mantengono momentum!")
        
        # Train
        model.train()
        train_losses = []
        
        # Apply warmup scheduler for first N epochs
        if warmup_scheduler is not None and epoch < warmup_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [Warmup] LR: {current_lr:.6f}")
        
        for batch in tqdm(train_loader, desc="Training"):
            cropped_img = batch['cropped_img'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            gt_translation = batch['translation'].to(device)
            bbox_base = batch['bbox_base'].to(device)
            obj_id = batch['obj_id'].to(device).long()
            
            optimizer.zero_grad()
            
            # Forward pass with optional AMP
            with torch.cuda.amp.autocast() if use_amp else torch.enable_grad():
                # Forward: ResNet predice SOLO quaternion
                pred_quaternion = model(cropped_img)
                
                # Calcola translation 3D usando geometria
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
                
                batch_diameters = diameter_lookup[obj_id]
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
                pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
                
                # Prepara model_points per batch (se usa ADD loss)
                if use_add_loss:
                    # Per ogni sample nel batch, usa i suoi model_points
                    # Assumiamo batch omogeneo (stesso obj_id) per semplicità
                    # Se batch ha oggetti diversi, usa il primo
                    batch_obj_id = int(obj_id[0].item())
                    batch_model_points = model_points_dict[batch_obj_id]
                else:
                    batch_model_points = None
                
                # Loss
                losses = criterion(
                    pred_quaternion,
                    pred_translation,
                    gt_quaternion,
                    gt_translation,
                    model_points=batch_model_points
                )
                loss = losses['total_loss']
            
            # Backward with optional AMP
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

                batch_diameters = diameter_lookup[obj_id]  # Fast GPU indexing!
                
                depth = pinhole.compute_depth_from_bbox(bbox_xyxy, batch_diameters)
                pred_translation = pinhole.unproject_2d_to_3d(center_2d_pixels, depth)
                
                # Prepara model_points per batch (se usa ADD loss)
                if use_add_loss:
                    batch_obj_id = int(obj_id[0].item())
                    batch_model_points = model_points_dict[batch_obj_id]
                else:
                    batch_model_points = None
                
                losses = criterion(
                    pred_quaternion,
                    pred_translation,
                    gt_quaternion,
                    gt_translation,
                    model_points=batch_model_points
                )
                
                val_losses.append(losses['total_loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Step scheduler
        if use_cosine:
            scheduler.step()  # Cosine: step every epoch
        else:
            scheduler.step(avg_val_loss)  # ReduceLR: step on plateau
        
        # Step warmup scheduler (first N epochs)
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Get image normalization stats from dataset
            image_mean, image_std = train_dataset.get_image_mean_std()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'lr': lr,
                'val_loss': avg_val_loss,
                'image_mean': image_mean.tolist(),
                'image_std': image_std.tolist()
            }, os.path.join(checkpoint_dir, checkpoint_name))  # Usa checkpoint_name personalizzato
            print(f"✓ Saved best model to {checkpoint_name}")
    
    print("\nTraining completed!")
