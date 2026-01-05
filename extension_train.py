from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from models.TridentNetPose import TridentNetPose
from models.ExtensionLoss import ExtensionLoss
from utils.pose_utils import load_all_models_points


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        device
    ):

    model.train()
    
    total_loss_sum = 0
    add_loss_sum = 0
    proj_loss_sum = 0
    rot_loss_sum = 0  # ✅ AGGIUNTO
    trans_err_cm_sum = 0
    proj_err_px_sum = 0
    rot_err_asymm_deg_sum = 0
    
    pbar = tqdm(loader, desc="** Training **")
    for batch in pbar:
        cropped_img = batch['cropped_img'].to(device, non_blocking=True)
        gt_quaternion = batch['quaternion'].to(device, non_blocking=True)
        gt_translation = batch['translation'].to(device, non_blocking=True)
        bbox_center = batch['bbox_center_pixel'].to(device, non_blocking=True)
        cropped_depth = batch['cropped_depth'].to(device, non_blocking=True)
        obj_id = batch['obj_id'].to(device, non_blocking=True).long()
        bbox_dims = batch['bbox_dims'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # forward
            pred_quat, pred_trans, pred_2d = model(cropped_img, cropped_depth, bbox_center, bbox_dims)
                
            loss_dict = criterion(
                pred_quat=pred_quat, 
                pred_trans=pred_trans, 
                gt_quat=gt_quaternion, 
                gt_trans=gt_translation, 
                pred_2d=pred_2d,
                class_ids=obj_id
            )

            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        total_loss_sum += loss
        add_loss_sum += loss_dict['add_loss'].item()
        proj_loss_sum += loss_dict['proj_loss'].item()
        rot_loss_sum += loss_dict['rot_loss'].item()
        trans_err_cm_sum += loss_dict['trans_err_cm'].item()
        proj_err_px_sum += loss_dict['proj_err_px'].item()
        rot_err_asymm_deg_sum += loss_dict['rot_err_asymm_deg']
    
    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'add_loss_avg': add_loss_sum / len(loader),
        'proj_loss_avg': proj_loss_sum / len(loader),
        'rot_loss_avg': rot_loss_sum / len(loader),
        'trans_err_cm_avg': trans_err_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader),
        'rot_err_asymm_deg_avg': rot_err_asymm_deg_sum / len(loader)
    }

    return avg_metrics

def validate(
        model, 
        loader, 
        criterion, 
        device
    ):

    model.eval()
    
    total_loss_sum = 0
    add_loss_sum = 0
    proj_loss_sum = 0
    rot_loss_sum = 0
    trans_err_cm_sum = 0
    proj_err_px_sum = 0
    rot_err_asymm_deg_sum = 0
    
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
                pred_quat, pred_trans, pred_2d = model(cropped_img, cropped_depth, bbox_center, bbox_dims)
                
                loss_dict = criterion(
                    pred_quat=pred_quat, 
                    pred_trans=pred_trans, 
                    gt_quat=gt_quaternion, 
                    gt_trans=gt_translation, 
                    pred_2d=pred_2d,
                    class_ids=obj_id 
                )
            
            # logging
            total_loss_sum += loss_dict['total_loss']
            add_loss_sum += loss_dict['add_loss'].item()
            proj_loss_sum += loss_dict['proj_loss'].item()
            rot_loss_sum += loss_dict['rot_loss'].item()
            trans_err_cm_sum += loss_dict['trans_err_cm'].item()
            proj_err_px_sum += loss_dict['proj_err_px'].item()
            rot_err_asymm_deg_sum += loss_dict['rot_err_asymm_deg']

    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'add_loss_avg': add_loss_sum / len(loader),
        'proj_loss_avg': proj_loss_sum / len(loader),
        'rot_loss_avg': rot_loss_sum / len(loader),
        'trans_err_cm_avg': trans_err_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader),
        'rot_err_asymm_deg_avg': rot_err_asymm_deg_sum / len(loader)
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
    partial_unfreeze=False,
    resume_from_checkpoint=None,
    reset_training=False
):
    points_dict = load_all_models_points(dataset_root, num_points=1000)

    model = TridentNetPose(
        cam_k=cam_k
    ).to(device)

    criterion = ExtensionLoss(
        add_weight=100.0,
        proj_weight=0.2,
        cam_k=cam_k,
        model_points_dict=points_dict
    ).to(device)

    params = [
        {'params': model.rgb_backbone.parameters(), 'lr': lr_rgb_backbone},
        {'params': model.depth_backbone.parameters(), 'lr': lr_new_components},
        {'params': model.fusion_fc.parameters(), 'lr': lr_new_components},
        {'params': model.rot_head.parameters(), 'lr': lr_new_components},
        {'params': model.z_head.parameters(), 'lr': lr_new_components},
        {'params': model.offset_head.parameters(), 'lr': lr_new_components}
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
        
        if reset_training:
            # MODALITÀ FINE-TUNING / PHASE 2: Carica solo i pesi, riparte da zero
            print(">>> RESET TRAINING ATTIVO: Ignoro epoch e optimizer del checkpoint.")
            print(">>> Si riparte da Epoch 0 con i nuovi Learning Rate.")
            print(f"    - RGB Backbone: {lr_rgb_backbone:.2e}")
            print(f"    - New Components: {lr_new_components:.2e}")
            start_epoch = 0
            best_loss = float('inf')
            # Non carichiamo optimizer/scheduler/scaler state (usiamo quelli freschi appena creati)
        else:
            # MODALITÀ RESUME NORMALE: Continua da dove era rimasto
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            
            print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
        
        print(f"Resumed logic complete. Start Epoch: {start_epoch}")

    
    print("Mixed Precision (AMP): ENABLED")

    model.freeze_rgb()
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Stampa LR dinamicamente in base al numero di gruppi
        if len(optimizer.param_groups) > 0:
            lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            print(f"Learning Rates: {' | '.join(lrs)}")
        else:
            print("Warning: No trainable parameters!")

        if epoch < freeze_rgb_epochs:
            model.freeze_rgb()
        elif epoch == freeze_rgb_epochs:
            model.unfreeze_rgb(partial=partial_unfreeze)
            optimizer.param_groups[0]['lr'] = lr_rgb_backbone
            print(f">>> 🔓 RGB Backbone Unfrozen (LR reset to {lr_rgb_backbone:.2e})")
            
        train_avg_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(
            f"  Train Loss: {train_avg_metrics['total_loss_avg']:.4f}, ADD: {train_avg_metrics['add_loss_avg']:.2f}, Rot: {train_avg_metrics['rot_loss_avg']:.2f}, Proj: {train_avg_metrics['proj_loss_avg']:.2f} "
            f"(Rot Err: {train_avg_metrics['rot_err_asymm_deg_avg']:.2f}°, Trans Err: {train_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj Err: {train_avg_metrics['proj_err_px_avg']:.2f} px)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        
        print(
            f"  Val Loss: {val_avg_metrics['total_loss_avg']:.4f}, ADD: {val_avg_metrics['add_loss_avg']:.2f}, Rot: {val_avg_metrics['rot_loss_avg']:.2f}, Proj: {val_avg_metrics['proj_loss_avg']:.2f} "
            f"(Rot Err: {val_avg_metrics['rot_err_asymm_deg_avg']:.2f}°, Trans Err: {val_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj Err: {val_avg_metrics['proj_err_px_avg']:.2f} px)"
        )

        scheduler.step(val_avg_metrics['total_loss_avg'])

        if val_avg_metrics['total_loss_avg'] < best_loss:
            best_loss = val_avg_metrics['total_loss_avg']
            
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }
            
            save_path = str(Path(checkpoint_dir) / "best_fusion_model.pt")
            torch.save(checkpoint_dict, save_path)
            print(f"✅ Checkpoint salvato: {save_path} (Loss: {best_loss:.4f})")
        print()
    
    print("\nTraining completed!")
