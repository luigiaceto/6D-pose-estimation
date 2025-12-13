import torch
from ultralytics import YOLO


def evaluate_YOLO(path: str = None, epochs: int = None, batch_size: int = None, IMG_SIZE: int = None, device = torch.device("cpu")):
    """
    Evaluate (best) model on test split.
    Uses the 'best.pt' weights obtained after the finetuning.
    """

    model = YOLO(f"{path}/checkpoints/best.pt")

    results = model.val(
        data=f"{path}/datasets/linemod/YOLO/datasets/data.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=IMG_SIZE,
        device=device,
        split="test" # testa il modello sul test set
    )