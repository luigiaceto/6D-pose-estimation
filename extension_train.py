from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from models.TridentNetPose import TridentNetPose
from models.ExtensionLoss import ExtensionLoss
from utils.pose_utils import load_models_points, compute_translation_from_depth_crop


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        device,
    ):

    model.eval() 
    
    # 2. Sblocca le parti che devono imparare (Training Mode)
    #    - fusion_fc, z_head, offset_head, rot_head: Hanno Dropout -> Serve .train()
    #    - depth_backbone: Ha BatchNorm nuove -> Serve .train() per calcolare le statistiche!
    model.fusion_fc.train()
    model.z_head.train()
    model.offset_head.train()
    model.rot_head.train() 
    model.depth_backbone.train()
    
    # Bug #3 Fix: Se RGB backbone è stata sbloccata, mettila in train() mode
    # (altrimenti BatchNorm usa statistiche ImageNet invece di aggiornarsi su LineMOD)
    if model.rgb_backbone[0].weight.requires_grad:
        model.rgb_backbone.train()  

    # Inizializzazione Accumulatori (solo loss geometriche)
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
            net_input_depth = cropped_depth.clone()
            
            # Setup camera intrinsics batch
            cam_k_batch = criterion.cam_k.repeat(len(obj_id), 1)
            
            # 1. PRE-CALCOLO Z GEOMETRIC PRIOR (usa bbox center come stima iniziale)
            z_prior_geom = compute_translation_from_depth_crop(
                cropped_depth=cropped_depth,      # Depth in METRI (già convertita dal dataset)
                pred_uv=bbox_center,              # USA BBOX CENTER come prior
                cam_k=cam_k_batch,
            )
            z_geometric = z_prior_geom[:, 2:3]  # (B, 1)
            
            # 2. Forward CON Z GEOMETRIC INJECTION
            pred_quat, pred_delta_z, pred_2d = model(
                cropped_img, 
                net_input_depth, 
                bbox_center, 
                bbox_dims, 
                z_geometric=z_geometric  # FIX: passa il prior alla rete
            )
            
            # 3. Calcola loss con z_geometric + delta_z
            loss_dict = criterion(
                pred_quat=pred_quat, 
                pred_delta_z=pred_delta_z,         
                gt_quat=gt_quaternion, 
                gt_trans=gt_translation, 
                pred_2d=pred_2d,
                class_ids=obj_id,
                z_geometric=z_geometric            
            )

            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # logging (solo loss geometriche)
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
    
    # Inizializzazione accumulatori (solo loss geometriche)
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
                net_input_depth = cropped_depth.clone()
                
                # Setup camera intrinsics batch
                cam_k_batch = criterion.cam_k.repeat(len(obj_id), 1)
                
                # 1. PRE-CALCOLO Z GEOMETRIC PRIOR (usa bbox center come stima iniziale)
                z_prior_geom = compute_translation_from_depth_crop(
                    cropped_depth=cropped_depth,      # Depth in METRI (già convertita dal dataset)
                    pred_uv=bbox_center,              # USA BBOX CENTER come prior
                    cam_k=cam_k_batch,
                )
                z_geometric = z_prior_geom[:, 2:3]  # (B, 1)
                
                # 2. Forward CON Z GEOMETRIC INJECTION
                pred_quat, pred_delta_z, pred_uv = model(
                    cropped_img, 
                    net_input_depth, 
                    bbox_center, 
                    bbox_dims, 
                    z_geometric=z_geometric  # FIX: passa il prior alla rete
                )
                
                # 3. Calcola loss
                loss_dict = criterion(
                    pred_quat=pred_quat, 
                    pred_delta_z=pred_delta_z,    
                    gt_quat=gt_quaternion, 
                    gt_trans=gt_translation, 
                    pred_2d=pred_uv,
                    class_ids=obj_id,
                    z_geometric=z_geometric      
                )
            
            # logging (solo loss geometriche)
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
    rot_weight=10.0,      # float o tuple (start, end) - Centered ADD/ADD-S
    trans_weight=10.0,  
    proj_weight=1.0,      # float o tuple (start, end) - 2D Projection (opzionale)
    switch_epoch=40,     # Epoca in cui switchare i pesi (per Curriculum Learning)
    partial_unfreeze=False,
    resume_from_checkpoint=None,
    reset_training=False 
):
    """
    Training con LOSS PURAMENTE GEOMETRICHE + Curriculum Learning.
    
    Usa solo:
    - L_rot: Centered ADD/ADD-S (isola rotazione)
    - L_trans: Pure Translation L1
    - L_proj: 2D Projection (opzionale)
    
    I parametri *_weight possono essere:
    - float: peso costante per tutto il training
    - tuple/list (start, end): peso cambia da start a end all'epoca switch_epoch
    """
    # Helper function per gestire pesi float o tuple
    def get_weight_value(weight, epoch, switch_epoch):
        """Ritorna il valore del peso in base all'epoca corrente."""
        if isinstance(weight, (tuple, list)):
            return weight[0] if epoch < switch_epoch else weight[1]
        else:
            return weight
    
    points_dict = load_models_points(dataset_root, num_points=2000)

    model = TridentNetPose(
        cam_k=cam_k
    ).to(device)

    # Inizializza loss con i valori iniziali (epoca 0)
    init_rot_weight = get_weight_value(rot_weight, 0, switch_epoch)
    init_trans_weight = get_weight_value(trans_weight, 0, switch_epoch)
    init_proj_weight = get_weight_value(proj_weight, 0, switch_epoch)
    
    criterion = ExtensionLoss(
        rot_weight=init_rot_weight,
        trans_weight=init_trans_weight,
        proj_weight=init_proj_weight,
        cam_k=cam_k,
        model_points_dict=points_dict,
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
            # 🔥 RESET MODE: Carica solo i pesi, riparte da Epoch 0
            print(">>> RESET TRAINING ATTIVO: Re-inizializzo le HEAD (Z e Rot) e riparto da Epoch 0.")
            print(">>> Si riparte da Epoch 0 con i nuovi Learning Rate.")
            print(f"    - RGB Backbone LR: {lr_rgb_backbone:.2e}")
            print(f"    - New Components LR: {lr_new_components:.2e}")
            
            # Re-init Rot Head
            torch.nn.init.xavier_uniform_(model.rot_head.weight, gain=0.01)
            model.rot_head.bias.data.fill_(0)
            model.rot_head.bias.data[0] = 1.0
            
            # Re-init Z Head (tutti i layer lineari)
            for m in model.z_head.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
            
            # Re-init Offset Head
            for m in model.offset_head.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
            
            start_epoch = 0
            best_loss = float('inf')
            # Non carichiamo optimizer/scheduler/scaler (usiamo quelli freschi)
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            
            print(f"✅ Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
    
    print("Mixed Precision (AMP): ENABLED")

    model.freeze_rgb()
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Aggiorna pesi dinamicamente
        current_rot = get_weight_value(rot_weight, epoch, switch_epoch)
        current_trans = get_weight_value(trans_weight, epoch, switch_epoch)
        current_proj = get_weight_value(proj_weight, epoch, switch_epoch)
        
        # Notifica switch se siamo esattamente all'epoca di cambio
        if epoch == switch_epoch:
            print(f"\n CURRICULUM SWITCH @ Epoch {epoch+1}:")
            if isinstance(rot_weight, (tuple, list)):
                print(f"   ROT:   {rot_weight[0]:.2f} → {rot_weight[1]:.2f}")
            if isinstance(trans_weight, (tuple, list)):
                print(f"   TRANS: {trans_weight[0]:.2f} → {trans_weight[1]:.2f}")
            if isinstance(proj_weight, (tuple, list)):
                print(f"   PROJ:  {proj_weight[0]:.2f} → {proj_weight[1]:.2f}")
            print()
        
        # Aggiorna pesi nella criterion (solo loss geometriche)
        criterion.w_rot = current_rot
        criterion.w_trans = current_trans
        criterion.w_proj = current_proj

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
            f"  Train Loss: {train_avg_metrics['total_loss_avg']:.4f}, "
            f"Trans: {train_avg_metrics['trans_loss_avg']:.4f}, Rot: {train_avg_metrics['rot_loss_avg']:.4f}, Proj: {train_avg_metrics['proj_loss_avg']:.4f} "
            f"(Rot Err: {train_avg_metrics['rot_err_deg_avg']:.4f}°, Trans Err: {train_avg_metrics['trans_err_cm_avg']:.4f} cm)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        
        print(
            f"  Val Loss: {val_avg_metrics['total_loss_avg']:.4f}, "
            f"Trans: {val_avg_metrics['trans_loss_avg']:.4f}, Rot: {val_avg_metrics['rot_loss_avg']:.4f}, Proj: {val_avg_metrics['proj_loss_avg']:.4f} "
            f"(Rot Err: {val_avg_metrics['rot_err_deg_avg']:.4f}°, Trans Err: {val_avg_metrics['trans_err_cm_avg']:.4f} cm)"
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
