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
        device
    ):

    model.train()
    
    total_loss_sum = 0
    rotation_loss_sum = 0
    translation_error_cm_sum = 0
    
    pbar = tqdm(loader, desc="**Training**")
    for batch in pbar:
        # Sposta dati su GPU
        rgb = batch['cropped_img'].to(device)
        depth = batch['cropped_depth'].to(device)
        bbox_center = batch['bbox_center_pixel'].to(device)
        
        gt_quat = batch['quaternion'].to(device)
        gt_trans = batch['translation'].to(device)
        obj_id = batch['obj_id'].to(device)
        
        optimizer.zero_grad()
        
        # eventualmente includere pred e loss in
        # 'with torch.cuda.amp.autocast(enabled=True):'

        # forward
        pred_quat, pred_trans = model(rgb, depth, bbox_center)
            
        # loss
        loss_dict = criterion(pred_quat, pred_trans, gt_quat, gt_trans, obj_id)
        loss = loss_dict['total_loss']
        
        # backward
        loss.backward()

        optimizer.step()
        
        # logging
        total_loss_sum += loss.item()
        rotation_loss_sum += loss_dict['rot_loss'].item()
        translation_error_cm_sum += loss_dict['trans_err_cm'].item()
    
    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rotation_loss_sum / len(loader),
        'trans_err_cm_avg': translation_error_cm_sum / len(loader)
    }

    return avg_metrics

def validate(model, loader, criterion, device):
    model.eval()
    
    total_loss_sum = 0
    rotation_loss_sum = 0
    translation_error_cm_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="**Validation**"):
            rgb = batch['cropped_img'].to(device)
            depth = batch['cropped_depth'].to(device)
            bbox_center = batch['bbox_center_pixel'].to(device)

            gt_quat = batch['quaternion'].to(device)
            gt_trans = batch['translation'].to(device)
            obj_id = batch['obj_id'].to(device)
            
            pred_quat, pred_trans = model(rgb, depth, bbox_center)
            
            loss_dict = criterion(pred_quat, pred_trans, gt_quat, gt_trans, obj_id)
            
            total_loss_sum += loss_dict['total_loss'].item()
            rotation_loss_sum += loss_dict['rot_loss'].item()
            translation_error_cm_sum += loss_dict['trans_err_cm'].item()

    avg_metrics = {
        'total_loss_avg': total_loss_sum / len(loader),
        'rot_loss_avg': rotation_loss_sum / len(loader),
        'trans_err_cm_avg': translation_error_cm_sum / len(loader)
    }

    return avg_metrics

def train(
    train_loader,
    val_loader,
    cam_k,
    checkpoint_dir='checkpoints',
    epochs=50,
    lr=1e-4,
    weight_decay=1e-5,
    device='cuda',
    freeze_epochs=5
):
    model = FusionPoseNet(cam_k=cam_k).to(device)

    criterion = RGBDPoseLoss(
        lambda_rot=10.0,
        lambda_trans=10.0
    ).to(device) # se uso alfa e beta non-learnable non serve lambda_rot/trans

    params = [
        # Gruppo 1: La backbone RGB (già pre-addestrata) -> Learning Rate molto basso
        {'params': model.rgb_backbone.parameters(), 'lr': 1e-5}, 
        
        # Gruppo 2: Tutto il resto (DepthEncoder, Heads, Fusion) -> Learning Rate normale
        {'params': model.depth_backbone.parameters(), 'lr': lr},
        {'params': model.fusion_fc.parameters(), 'lr': lr},
        {'params': model.rot_head.parameters(), 'lr': lr},
        {'params': model.z_head.parameters(), 'lr': lr},

        # Gruppo 3: i parametri della loss
        {'params': criterion.parameters(), 'lr': lr}
    ]

    # sto applicando bene il lr differenziale ???
    optimizer = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=1e-6
    )

    # Freeze iniziale RGB se vuoi (Transfer Learning)
    model.freeze_rgb()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # Stampa LR corrente (utile per debug)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.8f}")

        if epoch == freeze_epochs:
            # ATTENZIONE: nel caso si utilizzi unfreezing forse converrebbe
            # ridurre tipo di un fattore 10 il learning rate ??? Bisogna comunque
            # tenere conto che c'è anche la backbone CNN che va trainata from scratch
            # non come la ResNet che parte pre-addestrata
            model.unfreeze_rgb()
            print("Unfreezing RGB backbone...")
            
        train_avg_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(
            f"Train Loss: {train_avg_metrics['total_loss_avg']:.4f} "
            f"(Rot loss: {train_avg_metrics['rot_loss_avg']:.4f}, Transaltion Err: {train_avg_metrics['trans_err_cm_avg']:.2f} cm)"
        )

        val_avg_metrics = validate(model, val_loader, criterion, device)
        print(
            f"Val Loss: {val_avg_metrics['total_loss_avg']:.4f} "
            f"(Rot loss: {val_avg_metrics['rot_loss_avg']:.4f}, Translation Err: {val_avg_metrics['trans_err_cm_avg']:.2f} cm)\n"
        )
        
        scheduler.step()

        if val_avg_metrics['total_loss_avg'] < best_loss:
            best_loss = val_avg_metrics['total_loss_avg']
            torch.save(model.state_dict(), str(Path(checkpoint_dir) / "best_fusion_model.pt"))
