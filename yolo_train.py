from pathlib import Path
import torch
import shutil
from ultralytics import YOLO


def train_YOLO(
    epochs: int,
    batch_size: int,
    img_size: int,
    device,
    yolo_dataset_path,
    checkpoint_dir="checkpoints",
    pretrained_model="yolo11n.pt"
):
    """
    Finetune YOLO model on LineMOD.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to train on (cpu/cuda)
        yolo_dataset_path: Path to YOLO dataset folder (containing data.yml)
        checkpoint_dir: Directory to save checkpoints
        pretrained_model: Name of pretrained YOLO model
    """
    checkpoint_dir = Path(checkpoint_dir)
    yolo_dataset_path = Path(yolo_dataset_path)
    
    # Load pretrained model
    pretrained_path = checkpoint_dir / pretrained_model
    model = YOLO(str(pretrained_path))
    
    # Data config file
    data_yaml = yolo_dataset_path / "data.yml"
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=img_size,
        # Data augmentation
        hsv_h=0.1,
        hsv_s=0.1,
        hsv_v=0.1,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        translate=0.0,
        scale=0.0,
        erasing=0.0,
        exist_ok=True,
        patience=5,
        dropout=0.3
    )
    
    # Copy best model to checkpoints
    best_model_src = Path("runs") / "detect" / "train" / "weights" / "best_pose_model.pt"
    best_model_dst = checkpoint_dir / "best_yolo_model.pt"
    
    if best_model_src.exists():
        shutil.copy(str(best_model_src), str(best_model_dst))
        print(f"✅ Best model saved to: {best_model_dst}")
    else:
        print(f"⚠️  Warning: Could not find best model at {best_model_src}")
    
    return results