# Deep Closest Point — Codebase Explanation

## Project Tree

```
DCP/
├── main.py       # Training loop, valutazione, logging, entry point
├── model.py      # Intera architettura DCP (embedding, pointer, head)
├── data.py       # Dataset ModelNet40, generazione coppie (src, tgt)
├── util.py       # Funzioni di supporto (quaternione, trasformazione, Eulero)
└── readme.md     # Istruzioni training/testing
```

---

## Il Problema

Dato un oggetto 3D acquisito due volte da posizioni diverse, si ottengono due **point cloud** della stessa scena: una sorgente `src` e un target `tgt`. L'obiettivo è trovare la **trasformazione rigida** `(R, t)` tale che:

```
R · src + t ≈ tgt
```

DCP risolve questo problema in modo **differenziabile end-to-end**, permettendo al segnale di errore di fluire fino ai pesi della rete tramite backpropagation.

---

## `model.py` — Architettura DCP

### Pipeline generale

```
src, tgt  (batch, 3, num_points)
    │
    ▼
[ emb_nn ]         PointNet o DGCNN — feature per ogni punto, pesi condivisi
    │
    ▼
[ pointer ]        Identity (DCP-v1) o Transformer (DCP-v2) — comunicazione tra cloud
    │
    ▼
[ head ]           SVDHead (soft matching + SVD) o MLPHead (regressione diretta)
    │
    ▼
R_ab, t_ab, R_ba, t_ba
```

I tre blocchi sono intercambiabili via argomento da riga di comando, mantenendo la stessa interfaccia. Questo permette confronti diretti tra varianti senza modificare il codice.

---

### Funzioni di supporto globali

#### `clones(module, N)`
Crea N copie indipendenti di un modulo PyTorch tramite `copy.deepcopy`.
Usata per costruire gli stack di layer dell'Encoder e del Decoder, garantendo che ogni layer abbia pesi separati e non condivisi.

#### `attention(query, key, value, mask, dropout)`
Implementa la **scaled dot-product attention**:
```
scores = (Q · K^T) / √d_k
p_attn = softmax(scores)
output = p_attn · V
```
La divisione per `√d_k` stabilizza i gradienti quando la dimensione degli embedding è grande — senza di essa i prodotti scalari crescono e il softmax satura verso valori estremi (0 o 1), annullando il gradiente.

#### `nearest_neighbor(src, dst)`
Trova il punto più vicino in `dst` per ogni punto in `src`, espandendo la distanza euclidea al quadrato in forma matriciale per evitare loop espliciti:
```
‖x - y‖² = ‖x‖² - 2x·y + ‖y‖²
```
Usata come utility geometrica, non durante il forward principale.

#### `knn(x, k)`
Trova i k punti più vicini per ogni punto della point cloud, costruendo la stessa espansione di distanza euclidea ma restituendo gli indici dei k vicini invece del solo più vicino. Base per la costruzione del grafo in DGCNN.

#### `get_graph_feature(x, k=20)`
Costruisce le **edge features** per DGCNN. Per ogni punto `i` e ciascuno dei suoi k vicini `j`, concatena le coordinate di entrambi: `[x_j, x_i]`. Il risultato ha shape `(batch, 6, num_points, k)` — 6 canali perché si concatenano 3 coordinate del vicino e 3 del centro. Questo permette alla rete di vedere simultaneamente la posizione assoluta del punto e la sua relazione geometrica con i vicini, codificando la struttura locale della superficie.

---

### Blocco 1 — Embedding NN

**Scopo:** trasformare le coordinate xyz grezze di ogni punto in un vettore di feature ad alta dimensione, tale che punti geometricamente simili abbiano feature simili — indipendentemente dalla cloud di appartenenza. I pesi sono condivisi tra `src` e `tgt`.

#### `PointNet`
Cinque `Conv1d` con kernel=1 (matematicamente identiche a Linear applicate punto per punto) seguite da BatchNorm e ReLU. Ogni punto viene processato **indipendentemente** — nessuna comunicazione con i vicini. Semplice ed efficiente, ma cieco alla geometria locale. Output: `(batch, emb_dims, num_points)`.

#### `DGCNN`
Rete più ricca che sfrutta la struttura locale del punto cloud:

1. `get_graph_feature` costruisce le edge features `(batch, 6, num_points, k)`
2. Quattro blocchi `Conv2d + BN + ReLU + MaxPool` su dim k — ogni blocco processa le relazioni punto-vicino e aggrega i k vicini in un singolo vettore per punto
3. I risultati dei quattro blocchi vengono **concatenati** prima dell'embedding finale:
   - `x1` → feature di basso livello, geometria grezza
   - `x2, x3` → feature intermedie
   - `x4` → feature semantiche ad alto livello
4. `Conv2d` finale fonde i 512 canali concatenati in `emb_dims`

La concatenazione multi-scala è la particolarità chiave: l'embedding finale di ogni punto contiene simultaneamente informazione a quattro livelli di astrazione diversi. In questa implementazione il grafo k-NN viene costruito **una sola volta** sulle coordinate raw — nel paper originale veniva ricostruito ad ogni layer (Dynamic Graph), qui è una semplificazione che riduce il costo computazionale.

---

### Blocco 2 — Pointer

**Scopo:** permettere alle due cloud di comunicare prima del matching, in modo che ogni punto aggiori il suo embedding tenendo conto dell'altra cloud.

#### `Identity` — DCP-v1
Non fa nulla. Restituisce gli embedding invariati. La somma residuale in `DCP.forward` diventa `embedding + embedding = 2 × embedding`, irrilevante per il matching perché SVDHead lavora su similarità relative. Esiste per mantenere la stessa interfaccia e confrontare DCP-v1 e DCP-v2 cambiando solo un argomento.

#### `Transformer` — DCP-v2
Usa un Encoder-Decoder nel senso originale di "Attention is All You Need". Il modello viene chiamato **due volte con src e tgt swappati**:

```python
tgt_embedding = self.model(src, tgt)  # src codifica, tgt cross-attende a src
src_embedding = self.model(tgt, src)  # tgt codifica, src cross-attende a tgt
```

Questo produce embedding **context-aware**: ogni punto di `src` sa dove si trovano i punti di `tgt` prima del matching, e viceversa. Il risultato viene sommato agli embedding originali — connessione residuale che preserva l'informazione originale arricchendola con il contesto dell'altra cloud.

I componenti interni del Transformer:

**`EncoderDecoder`** — wrapper che orchestra encoder e decoder. `src_embed`, `tgt_embed` e `generator` sono `nn.Sequential()` vuoti perché DGCNN produce già gli embedding upstream e SVDHead gestisce l'output.

**`Encoder`** — stack di N `EncoderLayer`, ognuno con self-attention + FFN + connessioni residuali. LayerNorm finale.

**`EncoderLayer`** — self-attention su una cloud: ogni punto aggiorna il suo embedding guardando tutti gli altri punti della stessa cloud. Due `SublayerConnection` (residual + norm).

**`Decoder`** — stack di N `DecoderLayer`, ognuno con self-attention + cross-attention + FFN.

**`DecoderLayer`** — tre sublayer: self-attention sulla cloud target, cross-attention verso la memoria dell'encoder (l'altra cloud), feedforward. È qui che avviene la comunicazione tra le due cloud.

**`MultiHeadedAttention`** — proietta Q, K, V in h sottospazi di dimensione `d_k = emb_dims / h`, calcola l'attention in parallelo su tutti gli head, riconcatena. Ogni head impara relazioni geometriche diverse nello spazio degli embedding. Quattro proiezioni lineari: una per Q, K, V e una per l'output.

**`PositionwiseFeedForward`** — MLP a due layer applicato indipendentemente su ogni punto. Aumenta la capacità espressiva del Transformer oltre la sola attention.

**`LayerNorm`** — normalizzazione sull'ultima dimensione con parametri learnable `a_2` (scala) e `b_2` (bias). Stabilizza il training negli stack profondi.

**`SublayerConnection`** — implementa il pattern `x + sublayer(LayerNorm(x))`. Pre-norm invece di post-norm rispetto al paper originale — più stabile in pratica.

---

### Blocco 3 — Head

**Scopo:** trasformare gli embedding in `(R, t)`.

#### `MLPHead`
Approccio black box:
1. Concatena `src_embedding` e `tgt_embedding` → `(batch, emb_dims*2, num_points)`
2. Max pooling su `num_points` → singolo vettore globale `(batch, emb_dims*2)`
3. MLP a tre layer → `(batch, emb_dims//8)`
4. Due proiezioni lineari: quaternione (4 valori) e traslazione (3 valori)
5. Quaternione normalizzato sulla sfera unitaria → `quat2mat` → matrice R

Non ragiona su corrispondenze punto-per-punto — collassa tutta l'informazione in un vettore globale e lascia che la rete impari implicitamente la trasformazione. Funziona peggio di SVDHead, esiste come ablation per dimostrare che la SVD è la scelta corretta.

#### `SVDHead`
Approccio geometricamente motivato e completamente differenziabile:

**Step 1 — Soft matching:**
```python
scores = softmax(E_src^T · E_tgt / √d_k, dim=2)
# (batch, num_points_src, num_points_tgt)
```
Ogni riga è una distribuzione di probabilità: quanto il punto `i` di `src` corrisponde a ciascun punto di `tgt`. È identico alla formula dell'attention — il matching emerge dalla similarità degli embedding.

**Step 2 — Corrispondenti virtuali:**
```python
src_corr = tgt · scores^T
# (batch, 3, num_points_src)
```
Ogni punto di `src` ottiene un corrispondente "morbido" in `tgt` — media pesata di tutti i punti di `tgt`. Non è un punto reale ma un punto virtuale che massimizza la coerenza con gli embedding.

**Step 3 — Cross-covarianza e SVD:**
```python
H = src_centered · src_corr_centered^T   # (batch, 3, 3)
U, S, V = svd(H)
R = V · U^T
```
Soluzione chiusa del problema di Procrustes. Se `det(R) < 0` la SVD ha prodotto una riflessione — si corregge moltiplicando V per `diag(1, 1, -1)` per forzare `det(R) = +1` (rotation propria).

**Step 4 — Traslazione:**
```python
t = mean(src_corr) - R · mean(src)
```
Una volta fissata R, la traslazione ottimale allinea i baricentri — soluzione analitica.

Il gradiente fluisce da `(R, t)` → SVD → `src_corr` → `scores` → embedding, addestrando DGCNN e Transformer end-to-end.

**Limite strutturale:** il softmax forza che ogni punto di `src` abbia sempre un corrispondente in `tgt` con pesi che sommano a 1 — anche in presenza di overlap parziale, dove corrispondenti reali potrebbero non esistere.

---

### `DCP` — il modello completo

```python
def forward(self, src, tgt):
    # 1. Embedding indipendente, pesi condivisi
    src_embedding = self.emb_nn(src)
    tgt_embedding = self.emb_nn(tgt)

    # 2. Pointer — comunicazione tra cloud
    src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

    # 3. Residual update
    src_embedding = src_embedding + src_embedding_p
    tgt_embedding = tgt_embedding + tgt_embedding_p

    # 4. Head — trasformazione
    rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt)

    # 5. Inversa B→A
    if self.cycle:
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)
    else:
        rotation_ba = rotation_ab.transpose(2, 1)
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

    return rotation_ab, translation_ab, rotation_ba, translation_ba
```

La trasformazione inversa B→A con `cycle=False` è la soluzione analitica: l'inversa di una rotazione è la sua trasposta, la traslazione inversa si ricava componendo con R_ba. Con `cycle=True` la rete la calcola direttamente — più flessibile, abilita la cycle consistency loss.

---

## `data.py` — Dataset

### `ModelNet40`
Dataset di 40 categorie di modelli CAD 3D. Per ogni sample genera una coppia `(src, tgt)` applicando una trasformazione rigida casuale:

```
anglex, angley, anglez ∈ [0, π/factor]   (default factor=4 → max 45°)
translation ∈ [-0.5, 0.5]³

tgt = R_ab · src + t_ab
```

Le rotazioni vengono costruite come prodotto di tre matrici elementari `Rx · Ry · Rz`. I punti vengono permutati casualmente prima di essere restituiti — DCP non deve dipendere dall'ordine dei punti. Al test viene fissato un seed per sample per garantire riproducibilità.

Restituisce anche `R_ba, t_ba` (la trasformazione inversa) e gli angoli di Eulero, necessari per il calcolo delle metriche in `main.py`.

Con `--unseen`: training sulle ultime 20 categorie, test sulle prime 20 — simula la generalizzazione a categorie mai viste.

---

## `util.py` — Funzioni di supporto

#### `quat2mat(quat)`
Converte un quaternione `(x, y, z, w)` in matrice di rotazione 3×3 tramite la formula standard, implementata in forma vettorizzata su batch. Usata da MLPHead per restituire R nella stessa forma di SVDHead.

#### `transform_point_cloud(point_cloud, rotation, translation)`
Applica `(R, t)` a una point cloud: `R · point_cloud + t`. Usata in `main.py` per calcolare l'errore geometrico diretto nello spazio 3D dopo aver applicato la trasformazione predetta.

#### `npmat2euler(mats, seq='zyx')`
Converte matrici di rotazione in angoli di Eulero in gradi via `scipy`. Necessaria per le metriche di rotazione — MSE direttamente su matrici 3×3 non è interpretabile geometricamente, in gradi un errore di 2° è immediatamente comprensibile.

---

## `main.py` — Training Loop

### Loss

```python
# Loss principale — supervisionata
loss = MSE(R_pred^T · R_gt, I) + MSE(t_pred, t_gt)

# Cycle loss — opzionale, peso 0.1
cycle_loss = MSE(R_ba · R_ab, I) + MSE(R_ba^T · t_ab + t_ba, 0)
loss = loss + cycle_loss * 0.1
```

La rotazione si confronta tramite composizione `R_pred^T · R_gt` invece di sottrazione diretta — se la predizione è perfetta il prodotto è l'identità, che è la metrica naturale sullo spazio SO(3).

La cycle loss forza che applicare A→B e poi B→A riporti all'origine. È una regularizzazione geometrica, non il segnale supervisionato principale — peso 0.1 per non dominare.

### Optimizer e Scheduler

```python
Adam(lr=0.001, weight_decay=1e-4)
MultiStepLR(milestones=[75, 150, 200], gamma=0.1)
```

Il learning rate scala per 0.1 agli epoch 75, 150, 200 su 250 totali: `0.001 → 0.0001 → 0.00001 → 0.000001`. Convergenza grossolana nelle prime epoche, raffinamento nelle ultime.

### Metriche

Tre livelli misurati separatamente, in entrambe le direzioni A→B e B→A:

| Metrica | Come si calcola | Cosa misura |
|---|---|---|
| MSE/RMSE/MAE geometrico | `mean((R·src + t - tgt)²)` | Errore diretto nello spazio 3D |
| MSE/RMSE/MAE rotazione | `mean((euler_pred - euler_gt)²)` in gradi | Errore angolare interpretabile |
| MSE/RMSE/MAE traslazione | `mean((t_pred - t_gt)²)` | Errore di traslazione in unità di spazio |

### Salvataggio

Salva `model.best.t7` ogni volta che il test loss migliora, più un checkpoint per ogni epoch. Backup automatico dei sorgenti insieme ai checkpoint per riproducibilità.

---

## DCP-v1 vs DCP-v2 — Differenza chiave

| | DCP-v1 | DCP-v2 |
|---|---|---|
| Pointer | Identity | Transformer |
| Comunicazione tra cloud | Nessuna | Cross-attention bidirezionale |
| Embedding al matching | Solo geometria locale | Context-aware rispetto all'altra cloud |
| Costo computazionale | Basso | Alto |
| Performance | Baseline | Significativamente migliore |

---

## Limite strutturale per scenari reali

DCP assume **overlap completo** tra le due cloud — il softmax forza corrispondenze per ogni punto anche quando non esistono. In scenari con point cloud parziali (acquisizioni sparse, occlusioni, sweep incompleti) questo degrada le performance perché i pesi si distribuiscono su punti geometricamente lontani senza corrispondente reale.

Metodi alternativi che gestiscono esplicitamente l'overlap parziale: RPM-Net (2020), OverlapPredator (2021), GeoTransformer (2022).
