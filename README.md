# 6D Pose Estimation
This project addresses the task of 6D object pose estimation on the LineMOD preprocessed dataset.

## How to run the code
The project notebook has been written in order to run the experiments using colab. Once you logged in on Colab, just import the notebook of the project;
then just follow the notebook.

## 'datasets/' folder structure
L'obiettivo è arrivare alla seguente struttura del folder datasets/
```text
datasets/
└── linemod/
    ├── DenseFusion/                  <-- Scaricata da GDrive
    │   └── Linemod_preprocessed/     <-- Decompressa dallo zip
    │       ├── data/                 <-- Contiene le cartelle degli oggetti (01, 02, ecc.)
    │       │   ├── 01/
    │       │   │   ├── rgb/          <-- Immagini originali
    │       │   │   ├── depth/        <-- Immagini di profondità
    │       │   │   ├── mask/         <-- Maschere binarie
    │       │   │   ├── gt.yml        <-- Ground truth originale
    │       │   │   └── info.yml      <-- Matrice fotocamera (cam_K)
    │       │   ├── 02/
    │       │   └── ... (fino a 15)
    │       ├── models/               <-- Modelli 3D (.ply o .obj) degli oggetti
    │       ├── 01_gt.yml             <-- File GT copiato e modificato (con quaternioni)
    │       ├── 02_gt.yml             <-- File GT specifico modificato
    │       └── ..._gt.yml
    │
    └── YOLO/                         <-- Creata dallo script per YOLO
        └── datasets/
            ├── data.yaml             <-- File configurazione classi per YOLO
            ├── train/
            │   ├── images/           <-- Immagini rinominate (es. 01_0000.png)
            │   └── labels/           <-- File .txt con bounding box (es. 01_0000.txt)
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
```

## YOLO 'runs/' Folder Explained
La cartella **`runs/`** è il "diario di bordo" automatico di YOLO. Ultralytics è progettato per salvare *tutto* ciò che riguarda i tuoi esperimenti in modo organizzato, così non perdi mai i risultati o le configurazioni usate.

La struttura è gerarchica:
1.  **`runs/`**: La cartella radice.
2.  **`detect/`**: Indica il tipo di "task" (compito). Poiché stai facendo *Object Detection*, finisce qui. Se facessi segmentazione, troveresti `segment/`.
3.  **`train/`**: È il nome del tuo esperimento specifico.
    * *Nota:* Se lanciassi il training una seconda volta senza cambiare nome, YOLO creerebbe automaticamente `train2`, poi `train3`, ecc., per non sovrascrivere i dati precedenti.

### 📂 La cartella `weights/` (I Pesi)
Questa è la cartella più preziosa. Contiene il "cervello" addestrato del tuo modello.

* **📄 `best.pt`**: È il modello "campione". Durante le epoche, YOLO salva qui lo stato del modello che ha ottenuto il punteggio migliore (mAP più alta) sui dati di validazione.
    * *A cosa serve:* È quello che userai per fare predizioni (inference) nel mondo reale o nel tuo file `evaluate_YOLO.py`.
* **📄 `last.pt`**: È l'ultimo stato salvato al termine dell'addestramento (o al momento corrente se il training è ancora in corso).
    * *A cosa serve:* Se il training si interrompe per errore (es. salta la connessione a Colab), puoi riprenderlo esattamente da qui usando l'argomento `resume=True`.

### 📄 I File di Configurazione e Log

* **📄 `args.yaml`**: È la "ricetta" del tuo addestramento. Contiene tutti i parametri che hai passato (o quelli di default): `epochs: 50`, `batch: 64`, `imgsz: 640`, i percorsi dei dati, ecc.
    * *Perché è utile:* Se tra nel futuro vuoi rifare *esattamente* questo esperimento, guardi questo file per ricordarti che impostazioni avevi usato.

* **📄 `results.csv`**: È il report statistico grezzo. È un foglio di calcolo che aggiunge una riga per ogni epoca completata.
    * *Cosa contiene:* Colonne per `train/box_loss`, `val/box_loss`, `metrics/mAP50`, learning rate, ecc.
    * *Perché è utile:* Puoi aprirlo con Excel o Pandas per creare grafici personalizzati sull'andamento dell'addestramento.

### 🖼️ Le Immagini di Diagnostica (Fondamentali!)

YOLO genera queste immagini all'inizio per permetterti di controllare che i dati siano caricati correttamente.

* **🖼️ `labels.jpg`**: Ti dà una panoramica statistica del tuo dataset.
    * Solitamente contiene 4 grafici: quante istanze ci sono per ogni classe (è bilanciato?), la grandezza dei box (sono oggetti piccoli o grandi?), e la posizione dei box (sono tutti al centro o sparsi?).
    * *Controllo da fare:* Se vedi che una classe ha pochissime barre rispetto alle altre, il tuo dataset è sbilanciato.

* **🖼️ `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`**: Queste sono importantissime. Mostrano **esattamente cosa vede la rete neurale** durante il training.
    * Non sono le immagini originali, ma un **mosaico**. YOLO prende 4 o più immagini, le unisce, le taglia e applica le "augmentations" (cambi di colore, zoom, ecc.).
    * *Controllo da fare:* Apri queste immagini!
        1.  I rettangoli (bounding box) sono giusti? Combaciano con gli oggetti?
        2.  Le immagini sembrano corrette o sono troppo distorte/scure/rovinate?
        3.  Se i box sono sfasati qui, il modello non imparerà mai.

### Cosa manca (che potrebbe apparire alla fine)?
Quando il training finirà (o dopo un certo numero di epoche), potresti vedere apparire altri file utili:
* `results.png`: I grafici delle curve di Loss e mAP disegnati automaticamente.
* `confusion_matrix.png`: Ti dice quali classi il modello confonde tra loro (es. scambia spesso la classe "A" con la classe "B").
