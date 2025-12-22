from pathlib import Path
import torch
from ultralytics import YOLO


def evaluate_YOLO(epochs: int = None, batch_size: int = None, IMG_SIZE: int = None, device = torch.device("cpu")):
    """
    Evaluate (best) model on test split.
    Uses the 'best.pt' weights obtained after the finetuning.
    """

    
    model = YOLO(str(Path("checkpoints") / "best.pt"))

    results = model.val(
        data=str(Path("datasets") / "linemod" / "YOLO" / "datasets" / "data.yml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=IMG_SIZE,
        device=device,
        split="test" # testa il modello sul test set
    )