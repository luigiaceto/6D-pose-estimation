from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from models.TridentNetPose import TridentNetPose
from models.ExtensionLoss import ExtensionLoss
from utils.pose_utils import load_models_points


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        device,
    ):

    model.eval() 
    
    # Safe attribute access per DataParallel
    real_model = model.module if hasattr(model, "module") else model
    
    # 2. Sblocca le parti che devono imparare (Training Mode)
    #    - fusion_fc, z_head, offset_head, rot_head: Hanno Dropout -> Serve .train()
    #    - depth_backbone: Ha BatchNorm nuove -> Serve .train() per calcolare le statistiche!
    real_model.fusion_fc.train()
    real_model.z_head.train()
    real_model.offset_head.train()
    real_model.rot_head.train() 
    real_model.depth_backbone.train() 

    # Inizializzazione Accumulatori (loss geometriche 3D + 2D)
    total_loss_sum = 0
    rot_loss_sum = 0          # Centered ADD/ADD-S
    trans_loss_sum = 0        # Pure Translation L1
    proj_loss_sum = 0         # 2D Projection
    trans_err_cm_sum = 0
    proj_err_px_sum = 0
    rot_err_deg_sum = 0
    
    pbar = tqdm(loader, desc="** Training **")
    for batch in pbar:
        cropped_img = batch['cropped_img'].to(device, non_blocking=True)
        gt_quaternion = batch['quaternion'].to(device, non_blocking=True)
        gt_translation = batch['translation'].to(device, non_blocking=True)
        bbox_center = batch['bbox_center_pixel'].to(device, non_blocking=True)
        cropped_depth = batch['cropped_depth'].to(device, non_blocking=True)
        obj_id = batch['obj_id'].to(device, non_blocking=True).long().view(-1)
        bbox_dims = batch['bbox_dims'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # Forward (z_geometric calcolato internamente dal modello)
            pred_quat, pred_trans, pred_uv = model(
                cropped_img, 
                cropped_depth, 
                bbox_center, 
                bbox_dims
            )
            
            # Calcola loss geometrica 3D + 2D (con pred_uv disaccoppiato)
            loss_dict = criterion(
                pred_quat=pred_quat, 
                pred_trans=pred_trans,
                gt_quat=gt_quaternion, 
                gt_trans=gt_translation, 
                class_ids=obj_id,
                pred_uv=pred_uv
            )

            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # logging (loss geometriche 3D + 2D)
        total_loss_sum += loss.item()
        rot_loss_sum += loss_dict['rot_loss'].item()
        trans_loss_sum += loss_dict['trans_loss'].item()
        proj_loss_sum += loss_dict['proj_loss'].item()
        trans_err_cm_sum += loss_dict['trans_err_cm'].item()
        proj_err_px_sum += loss_dict['proj_err_px'].item()
        rot_err_deg_sum += loss_dict['rot_err_deg'].item()
    
    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rot_loss_sum / len(loader),
        'trans_loss_avg': trans_loss_sum / len(loader),
        'proj_loss_avg': proj_loss_sum / len(loader),
        'trans_err_cm_avg': trans_err_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader),
        'rot_err_deg_avg': rot_err_deg_sum / len(loader)
    }

    return avg_metrics

def validate(
        model, 
        loader, 
        criterion, 
        device
    ):

    model.eval()
    
    # Inizializzazione accumulatori (loss geometriche 3D + 2D)
    total_loss_sum = 0
    rot_loss_sum = 0
    trans_loss_sum = 0
    proj_loss_sum = 0
    trans_err_cm_sum = 0
    proj_err_px_sum = 0
    rot_err_deg_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="** Validation **"):
            cropped_img = batch['cropped_img'].to(device, non_blocking=True)
            gt_quaternion = batch['quaternion'].to(device, non_blocking=True)
            gt_translation = batch['translation'].to(device, non_blocking=True)
            bbox_center = batch['bbox_center_pixel'].to(device, non_blocking=True)
            cropped_depth = batch['cropped_depth'].to(device, non_blocking=True)
            obj_id = batch['obj_id'].to(device, non_blocking=True).long()
            bbox_dims = batch['bbox_dims'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                # Forward (z_geometric calcolato internamente dal modello)
                pred_quat, pred_trans, pred_uv = model(
                    cropped_img, 
                    cropped_depth, 
                    bbox_center, 
                    bbox_dims
                )
                
                # Calcola loss geometrica 3D + 2D (con pred_uv disaccoppiato)
                loss_dict = criterion(
                    pred_quat=pred_quat, 
                    pred_trans=pred_trans,
                    gt_quat=gt_quaternion, 
                    gt_trans=gt_translation, 
                    class_ids=obj_id,
                    pred_uv=pred_uv
                )
            
            # logging (loss geometriche 3D + 2D)
            total_loss_sum += loss_dict['total_loss'].item()
            rot_loss_sum += loss_dict['rot_loss'].item()
            trans_loss_sum += loss_dict['trans_loss'].item()
            proj_loss_sum += loss_dict['proj_loss'].item()
            trans_err_cm_sum += loss_dict['trans_err_cm'].item()
            proj_err_px_sum += loss_dict['proj_err_px'].item()
            rot_err_deg_sum += loss_dict['rot_err_deg'].item()

    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rot_loss_sum / len(loader),
        'trans_loss_avg': trans_loss_sum / len(loader),
        'proj_loss_avg': proj_loss_sum / len(loader),
        'trans_err_cm_avg': trans_err_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader),
        'rot_err_deg_avg': rot_err_deg_sum / len(loader)
    }

    return avg_metrics

def train(
    dataset_root,
    train_loader,
    val_loader,
    cam_k,
    checkpoint_dir='checkpoints',
    epochs=50,
    lr_rgb_backbone=1e-5,
    lr_new_components=1e-4,
    weight_decay=1e-5,
    device='cuda',
    freeze_rgb_epochs=5,
    rot_weight=10.0,      # Peso fisso per Centered ADD/ADD-S
    trans_weight=10.0,    # Peso fisso per Pure Translation L1
    proj_weight=1.0,      # Peso fisso per 2D Projection
    resume_from_checkpoint=None
):
    """
    Training con LOSS GEOMETRICA 3D + 2D.
    
    Usa:
    - L_rot: Centered ADD/ADD-S (isola rotazione)
    - L_trans: Pure Translation L1 (ottimizza implicitamente depth + offset UV)
    - L_proj: 2D Projection (regolarizzazione, guida l'ottimizzazione)
    
    I pesi delle loss sono fissi per tutto il training.
    RGB Backbone: Partial unfreeze (solo layer4) dopo freeze_rgb_epochs.
    """
    
    points_dict = load_models_points(dataset_root, num_points=2000)

    model = TridentNetPose(
        cam_k=cam_k
    ).to(device)
    
    # Multi-GPU support con DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    # Safe attribute access per DataParallel
    real_model = model.module if hasattr(model, "module") else model

    criterion = ExtensionLoss(
        rot_weight=rot_weight,
        trans_weight=trans_weight,
        proj_weight=proj_weight,
        cam_k=cam_k,
        model_points_dict=points_dict,
    ).to(device)

    params = [
        {'params': real_model.rgb_backbone.parameters(), 'lr': lr_rgb_backbone},
        {'params': real_model.depth_backbone.parameters(), 'lr': lr_new_components},
        {'params': real_model.fusion_fc.parameters(), 'lr': lr_new_components},
        {'params': real_model.rot_head.parameters(), 'lr': lr_new_components},
        {'params': real_model.z_head.parameters(), 'lr': lr_new_components},
        {'params': real_model.offset_head.parameters(), 'lr': lr_new_components}
    ]

    optimizer = optim.AdamW(
        params,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=[1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
    )

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from_checkpoint is not None:
        print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        
        print(f"✅ Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
    
    print("Mixed Precision (AMP): ENABLED")

    real_model.freeze_rgb()
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Stampa LR dinamicamente in base al numero di gruppi
        if len(optimizer.param_groups) > 0:
            lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            print(f"Learning Rates: {' | '.join(lrs)}")
        else:
            print("Warning: No trainable parameters!")

        if epoch < freeze_rgb_epochs:
            real_model.freeze_rgb()
        elif epoch == freeze_rgb_epochs:
            real_model.unfreeze_rgb()  # Partial unfreeze (solo layer4)
            optimizer.param_groups[0]['lr'] = lr_rgb_backbone
            print(f" RGB Backbone Unfrozen - Partial (layer4 only) | LR reset to {lr_rgb_backbone:.2e}")
            
        train_avg_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(
            f"  Train Loss: {train_avg_metrics['total_loss_avg']:.4f}, "
            f"Rot: {train_avg_metrics['rot_loss_avg']:.4f}, Trans: {train_avg_metrics['trans_loss_avg']:.4f}, Proj: {train_avg_metrics['proj_loss_avg']:.4f} "
            f"(Rot: {train_avg_metrics['rot_err_deg_avg']:.2f}°, Trans: {train_avg_metrics['trans_err_cm_avg']:.2f}cm, Proj: {train_avg_metrics['proj_err_px_avg']:.2f}px)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        
        print(
            f"  Val Loss: {val_avg_metrics['total_loss_avg']:.4f}, "
            f"Rot: {val_avg_metrics['rot_loss_avg']:.4f}, Trans: {val_avg_metrics['trans_loss_avg']:.4f}, Proj: {val_avg_metrics['proj_loss_avg']:.4f} "
            f"(Rot: {val_avg_metrics['rot_err_deg_avg']:.2f}°, Trans: {val_avg_metrics['trans_err_cm_avg']:.2f}cm, Proj: {val_avg_metrics['proj_err_px_avg']:.2f}px)"
        )

        scheduler.step(val_avg_metrics['total_loss_avg'])

        if val_avg_metrics['total_loss_avg'] < best_loss:
            best_loss = val_avg_metrics['total_loss_avg']
            
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': real_model.state_dict(),  # Usa real_model per evitare prefisso module.
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }
            
            save_path = str(Path(checkpoint_dir) / "best_fusion_model.pt")
            torch.save(checkpoint_dict, save_path)
            print(f"✅ Checkpoint saved: {save_path} (Loss: {best_loss:.4f})")
        print()
    
    print("\nTraining completed!")
