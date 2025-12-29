from pathlib import Path
import torch
from ultralytics import YOLO


def evaluate_YOLO(
    epochs: int,
    batch_size: int,
    img_size: int,
    device,
    yolo_dataset_path,
    model_path
):
    """
    Evaluate YOLO model on test split.
    
    Args:
        epochs: Number of epochs (for validation config)
        batch_size: Batch size for evaluation
        img_size: Input image size
        device: Device to evaluate on (cpu/cuda)
        yolo_dataset_path: Path to YOLO dataset folder (containing data.yml)
        model_path: Path to trained YOLO model weights
    """
    yolo_dataset_path = Path(yolo_dataset_path)
    model_path = Path(model_path)
    
    # Load model
    model = YOLO(str(model_path))
    
    # Data config file
    data_yaml = yolo_dataset_path / "data.yml"
    
    # Evaluate on test set
    results = model.val(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        split="test"
    )
    
    return results