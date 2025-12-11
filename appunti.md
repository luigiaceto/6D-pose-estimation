# Letteratura 6D Pose Estimation
- RGB-D -> immagini RGB con canale aggiuntivo dedicato alla profondità (informazioni di distanza, quanto è lontano il punto nel pixel dalla fotocamera?). L'immagine diventa H x W x 4

- In ambito di Computer Vision e Robotica, i low-textured objects (oggetti a bassa testurizzazione) sono oggetti che hanno superfici uniformi, senza pattern distintivi, disegni, variazioni di colore o dettagli grafici forti. Pensa a una tazza di ceramica bianca lucida, un tubo di metallo cromato, una parete intonacata di bianco o un foglio di carta vuoto. Per noi umani riconoscerli è banale. Per un computer, invece, sono un incubo

- known objects -> classe di oggetti vista almeno una volta in trainig

# Dataset LineMode
Ogni folder rappresenta la classe di oggetto a cui siamo interessati nelle immagini contenute nel folder (es. 01 è scimmia, 06 è gatto), tutti gli altri oggetti nelle foto di quel folder sono considerati disturbi/rumore. Infatti avremo, per ogni immagine, solo la ground truth di un oggetto.

Dentro ogni folder ci sono:
- cartella 'depth' -> informazioni di depth della foto i-esima

- cartella 'mask' -> maschera dell'oggetto a cui siamo interessati nella foto i-esima

- cartella 'rgb' -> foto RGB-only

- file 'gt.yml' -> ground truth di tutte le foto (matrice di rotazione, vettore di traslazione, bounding box e classe dell'oggetto)

- file 'info.yml' -> informazioni per mappare il mondo 3D in un'immagine 2D. Permette di disegnare il cubo attorno all'oggetto e serve per calcolare la metrica di validazione ADD. Si noti che contiene sempre la stessa matrice K, per ogni file e per ogni riga. Questo perchè K non dipende da dove viene posizionata la telecamera ma dalle caratteristiche fisiche di essa ovvero Lunghezza Focale (quanto zooma la lente, definita da f_x e f_y) e Centro Ottico (dove cade il centro della lente sul sensore -> "Quello che per la fisica è il centro (0, 0), nell'immagine corrisponde al pixel (c_x, c_y)"). Questi due valori non cambiano mai se non cambi fisicamente l'obiettivo o se tieni costante lo zoom. K (matrice 3x3) contiene, disposti in un certo modo, f_x, f_y, c_x e c_y.

- file 'test.txt' -> ci dice quali immagini, nella cartella, sono di test

- file 'train.txt' -> ci dice quali immagini, nella cartella, sono di training

L'origine del sistema 3D nelle immagini del dataset è il centro ottico della lente della fotocamera. Z punta in avanti dentro la foto (profondità), x punta verso la destra dell'immagine e y punta verso il basso dell'immagine.

Ogni oggetto ha il suo sistema di coordinate locale incollato addosso che è il centro geometrico dell'oggetto stesso. Le distanze sono definite proprio rispetto a questo centro. Il compito della 6D estimation è trovare la matrice di rotazione R ed il vettore di traslazione t che trasportano i punti del sistema 'Oggetto' al sistema 'Camera', secondo la formula:

PuntoOggetto_sistemaCamera = R*PuntoOggetto_sistemaOggetto + t

C'è anche il sistema di coordinate dell'immagine (2D). Questo ha origine in altro a sinistra: u è la coordinata orizzontale (colonna) e v è la coordinata verticale (riga). La matrice contenuta in info.yml (una per immagine) permette di mappare dal sistema 3D della foto al sistema dell'immagine 2D (pixel).

Si può notare che nelle immagini c'è una sorta di riquadro bianco e nero, quelli sono marker fiduciali che fanno da riferimenti di calibrazione. Infatti, quando i ricercatori hanno creato il dataset, non hanno indovinato la posizione degli oggetti ad occhio; hanno usato un software automatico per generare la ground-truth (R e t). Il software riconosce i quadratini e da questi calcola con alta precisione la posizione della telecamera rispetto il tavolo: una volta riconosciuta questa posizione si possono calcolare le posizioni degli oggetti che sono appoggiati sopra ottenendo il vettore che collega il centro dell'obiettivo al centro dell'oggetto e la rotazione del sistema di riferimento dell'oggetto rispetto il sistema di riferimento della telecamera.

Nota: ci si potrebbe domandare quale dovrebbe essere l'utilità pratica di K, visto che la rete (non specifico quale e come) guarda l'immagine e predice direttamente R e t. Come dovrei fare a verificare che la predizione sia effettivamente giusta? Non posso andare ad occhio: qui entra in gioco la mappatura 3D -> 2D; prendiamo la posa (R, t) predetta, prendiamo il modello 3D dell'oggetto, usiamo K (insieme a R e t) per proiettare il modello 3D (fatto di punti) nella foto 2D. A questo punto, se il modello (o cubo se vogliamo) combacia con l'oggetto nella foto allora la predizione è corretta.