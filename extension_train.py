from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from models.FusionPoseNet import FusionPoseNet
from models.ExtensionLoss import RGBDPoseLoss
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
    reset_training=False,
    loss_mode='add',
    freeze_config=None
):
    points_dict = load_all_models_points(dataset_root, num_points=1000)

    model = FusionPoseNet(
        cam_k=cam_k
    ).to(device)

    # Configurazione Loss in base a loss_mode
    if loss_mode == 'add':
        # Modalità standard: ADD + Projection
        criterion = RGBDPoseLoss(
            add_weight=100.0,
            proj_weight=0.2,
            cam_k=cam_k,
            model_points_dict=points_dict,
            loss_mode='add'
        ).to(device)
        print("\ud83c\udfaf Loss Mode: ADD (w_add=100.0, w_proj=0.2)")
    elif loss_mode == 'rotation':
        # Modalità chirurgia rotazione: solo rotation loss
        criterion = RGBDPoseLoss(
            add_weight=0.1,    # Molto basso, solo per coerenza geometrica
            proj_weight=0.0,   # Disattivata
            rot_weight=10.0,   # PRIORITÀ ASSOLUTA
            cam_k=cam_k,
            model_points_dict=points_dict,
            loss_mode='rotation'
        ).to(device)
        print("\ud83c\udfaf Loss Mode: ROTATION (w_rot=10.0, w_add=0.1)")
    else:
        raise ValueError(f"loss_mode non valida: {loss_mode}. Usa 'add' o 'rotation'.")
    
    # Default freeze_config se non specificato
    if freeze_config is None:
        freeze_config = {
            'rgb_backbone': False,
            'depth_backbone': False,
            'fusion': False,
            'rot_head': False,
            'z_head': False,
            'offset_head': False
        }

    # Configurazione optimizer in base a freeze_config
    params = []
    
    if not freeze_config['rgb_backbone']:
        params.append({'params': model.rgb_backbone.parameters(), 'lr': lr_rgb_backbone})
        print("  RGB Backbone: TRAINABLE")
    else:
        print(" RGB Backbone: FROZEN")
        for p in model.rgb_backbone.parameters():
            p.requires_grad = False
    
    if not freeze_config['depth_backbone']:
        params.append({'params': model.depth_backbone.parameters(), 'lr': lr_new_components})
        print(" Depth Backbone: TRAINABLE")
    else:
        print(" Depth Backbone: FROZEN")
        for p in model.depth_backbone.parameters():
            p.requires_grad = False
    
    if not freeze_config['fusion']:
        params.append({'params': model.fusion_fc.parameters(), 'lr': lr_new_components})
        print(" Fusion Layer: TRAINABLE")
    else:
        print(" Fusion Layer: FROZEN")
        for p in model.fusion_fc.parameters():
            p.requires_grad = False
    
    if not freeze_config['rot_head']:
        params.append({'params': model.rot_head.parameters(), 'lr': lr_new_components})
        print(" Rotation Head: TRAINABLE")
    else:
        print(" Rotation Head: FROZEN")
        for p in model.rot_head.parameters():
            p.requires_grad = False
    
    if not freeze_config['z_head']:
        params.append({'params': model.z_head.parameters(), 'lr': lr_new_components})
        print(" Z Head: TRAINABLE")
    else:
        print(" Z Head: FROZEN")
        for p in model.z_head.parameters():
            p.requires_grad = False
    
    if not freeze_config['offset_head']:
        params.append({'params': model.offset_head.parameters(), 'lr': lr_new_components})
        print("  Offset Head: TRAINABLE")
    else:
        print("  Offset Head: FROZEN")
        for p in model.offset_head.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(
        params,
        weight_decay=weight_decay
    )
    
    # Scheduler semplificato (compatibile con numero dinamico di param groups)
    min_lrs = [1e-8] * len(params)  # Un min_lr per ogni gruppo
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=min_lrs,
        # verbose=True
    )

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from_checkpoint is not None:
        print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        
        # Carica sempre i pesi del modello
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
    already_unfreezed = False # serve per evitare inconsistenze durante l'unfreezing partendo da un checkpoint
    
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
            # È il momento dello sblocco!
            model.unfreeze_rgb(partial=partial_unfreeze)
            
            # --- FIX CRITICO: RESET LEARNING RATE BACKBONE ---
            # Forziamo il LR della backbone al valore iniziale (1e-6), ignorando 
            # eventuali tagli fatti dallo scheduler durante il freeze.
            optimizer.param_groups[0]['lr'] = lr_rgb_backbone
            print(f" UNFREEZE COMPLETO: Backbone LR resettato a {lr_rgb_backbone:.2e}")
            
        train_avg_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(
            f"  Train Loss: {train_avg_metrics['total_loss_avg']:.4f}, ADD: {train_avg_metrics['add_loss_avg']:.2f}, Rot: {train_avg_metrics['rot_loss_avg']:.2f}, Proj: {train_avg_metrics['proj_loss_avg']:.2f} "
            f"(Rot Err: {train_avg_metrics['rot_err_deg_avg']:.2f}°, Trans Err: {train_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj Err: {train_avg_metrics['proj_err_px_avg']:.2f} px)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        print(
            f"  Val Loss: {val_avg_metrics['total_loss_avg']:.4f}, ADD: {val_avg_metrics['add_loss_avg']:.2f}, Rot: {val_avg_metrics['rot_loss_avg']:.2f}, Proj: {val_avg_metrics['proj_loss_avg']:.2f} "
            f"(Rot Err: {val_avg_metrics['rot_err_deg_avg']:.2f}°, Trans Err: {val_avg_metrics['trans_err_cm_avg']:.2f} cm, Proj Err: {val_avg_metrics['proj_err_px_avg']:.2f} px)"
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
