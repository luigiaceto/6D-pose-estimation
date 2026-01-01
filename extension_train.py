from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from models.FusionPoseNet import FusionPoseNet
from models.ExtensionLoss import RGBDPoseLoss


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
    rotation_loss_sum = 0
    translation_error_cm_sum = 0
    proj_err_px_sum = 0
    
    pbar = tqdm(loader, desc="**Training**")
    for batch in pbar:
        # Sposta dati su GPU
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
        total_loss_sum += loss.item()
        rotation_loss_sum += loss_dict['rot_loss'].item()
        translation_error_cm_sum += loss_dict['trans_err_cm'].item()
        proj_err_px_sum += loss_dict['proj_err_px'].item()
    
    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rotation_loss_sum / len(loader),
        'trans_err_cm_avg': translation_error_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader)
    }

    return avg_metrics

def validate(model, loader, criterion, device):
    model.eval()
    
    total_loss_sum = 0
    rotation_loss_sum = 0
    translation_error_cm_sum = 0
    proj_err_px_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="**Validation**"):
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
            
            total_loss_sum += loss_dict['total_loss'].item()
            rotation_loss_sum += loss_dict['rot_loss'].item()
            translation_error_cm_sum += loss_dict['trans_err_cm'].item()
            proj_err_px_sum += loss_dict['proj_err_px'].item()

    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rotation_loss_sum / len(loader),
        'trans_err_cm_avg': translation_error_cm_sum / len(loader),
        'proj_err_px_avg': proj_err_px_sum / len(loader)
    }

    return avg_metrics

def train(
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
    resume_from_checkpoint=None
):
    model = FusionPoseNet(
        cam_k=cam_k
    ).to(device)

    criterion = RGBDPoseLoss(
        cam_k=cam_k
    ).to(device)

    params = [
        # Gruppo 1: Backbone RGB (Transfer Learning) -> LR molto basso
        {'params': model.rgb_backbone.parameters(), 'lr': lr_rgb_backbone}, 
        
        # Gruppo 2: Backbone Depth e Fusione
        {'params': model.depth_backbone.parameters(), 'lr': lr_new_components},
        {'params': model.fusion_fc.parameters(), 'lr': lr_new_components},
        
        # Gruppo 3: Le Tre Teste
        {'params': model.rot_head.parameters(), 'lr': lr_new_components},
        {'params': model.z_head.parameters(), 'lr': lr_new_components},      # Testa per Z (metri)
        {'params': model.offset_head.parameters(), 'lr': lr_new_components}, # Testa per Offset (pixel)

        # Gruppo 4: Parametri Learnable della Loss (s_rot, s_trans, s_proj)
        {'params': criterion.parameters(), 'lr': lr_new_components}
    ]

    optimizer = optim.AdamW(
        params,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=True
    )

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint se fornito
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
        
        print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
    
    print("Mixed Precision (AMP): ENABLED")

    # Freeze iniziale RGB (Transfer Learning)
    model.freeze_rgb()
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head = optimizer.param_groups[1]['lr']
        
        print(f"LR Backbone RGB: {lr_backbone:.2e} | LR Heads: {lr_head:.2e}")

        if epoch == freeze_rgb_epochs:
            model.unfreeze_rgb()
            print(">>> Unfreezing RGB backbone...")
            
        train_avg_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        # Converti rot_loss in gradi (assumendo che sia in radianti)
        train_rot_deg = np.degrees(train_avg_metrics['rot_loss_avg'])
        print(
            f"  Train Loss: {train_avg_metrics['total_loss_avg']:.4f} "
            f"(Rot: {train_rot_deg:.2f}°, Trans: {train_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj: {train_avg_metrics['proj_err_px_avg']:.2f} px)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        val_rot_deg = np.degrees(val_avg_metrics['rot_loss_avg'])
        print(
            f"  Val Loss: {val_avg_metrics['total_loss_avg']:.4f} "
            f"(Rot: {val_rot_deg:.2f}°, Trans: {val_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj: {val_avg_metrics['proj_err_px_avg']:.2f} px)"
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
            print(f"✓ Checkpoint salvato: {save_path} (Loss: {best_loss:.4f})")
        print()
    
    print("\nTraining completed!")
