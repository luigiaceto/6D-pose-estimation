import torch
import shutil
from ultralytics import YOLO


def train_YOLO(path: str = None, epochs: int = None, batch_size: int = None, device = torch.device("cpu"), IMG_SIZE: int = None):
    """
    Finetune YOLO model on LineMOD.
    After training evaluate (in 'evaluate_YOLO.py') on validation set by returning metrics like mAP.
    Save model to checkpoints.
    """
    
    # se 'yolo11n.pt' (i pesi pre-addestrati) non sono già dentro 'checkpoints/' allora verranno scaricati sul momento
    model = YOLO(f"{path}/checkpoints/yolo11n.pt")

    # model will automatically scale the image and related bounding box according to imgsz.
    # Il metodo train stampa ad ogni epoca di validazione le metriche
    results = model.train(
        data=f"{path}/datasets/linemod/YOLO/datasets/data.yaml",
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=IMG_SIZE,
        # data augmentation
        hsv_h=0.1,
        hsv_s=0.1,
        hsv_v=0.1,
        flipud=0.0, # niente ribaltamento verticale
        fliplr=0.0, # ribaltamento orizzontale ???
        mosaic=0.0, # mosaic aumentation, potrebbe servire ???
        translate=0.0,
        scale=0.0,
        erasing=0.0,
        exist_ok=True,
        patience=5, # se per 5 epoche di fila il modello non migliora sul validation set allora il training si ferma. Evita anche overfit
        dropout=0.3
    )
    
    # si prende lo snapshot del modello che ha ottenuto le metriche migliori (ad ogni epoca c'è uno snapshot).
    # Si tenga a mente che YOLO tiene salvato solo last.pt e best.pt
    shutil.copy(f"./runs/detect/train/weights/best.pt", f"./checkpoints/best.pt")