from pathlib import Path
import torch
import shutil
from ultralytics import YOLO


def train_YOLO(epochs: int = None, batch_size: int = None, IMG_SIZE: int = None, device = torch.device("cpu")):
    """
    Finetune YOLO model on LineMOD.
    After training evaluate (in 'evaluate_YOLO.py') on validation set by returning metrics like mAP.
    Save model to checkpoints.

    Nota:
    YOLO non ha bisogno di nessun oggetto DataSet o DataLoader, gestisce tutto internamente.
    Basta fornigli il file yml e strutturare la cartella che contiene il dataset in una certa
    maniera.
    """
    
    # se 'yolo11n.pt' (i pesi pre-addestrati) non sono già dentro 'checkpoints/' allora verranno scaricati sul momento
    
    model = YOLO(str(Path("checkpoints") / "yolo11n.pt"))

    # model will automatically scale the image and related bounding box according to imgsz.
    # Il metodo train stampa ad ogni epoca di validazione le metriche
    results = model.train(
        data=str(Path("datasets") / "linemod" / "YOLO" / "datasets" / "data.yml"),
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
    shutil.copy(
        str(Path("runs") / "detect" / "train" / "weights" / "best.pt"), 
        str(Path("checkpoints") / "best.py")
    )