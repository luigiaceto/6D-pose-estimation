# 6D Pose Estimation
This project addresses the task of 6D object pose estimation on the LineMOD preprocessed dataset.

## How to run the code
If you want to run the code use Google Colab and import just the notebook of the project.
Then just follow the notebook.

## datasets/ folder structure
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