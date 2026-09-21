# Paper summary: parking-lot segmentation pipeline

Qiam, Devunuri, Lehe. *A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation*. WACV 2025. [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Qiam_A_Pipeline_and_NIR-Enhanced_Dataset_for_Parking_Lot_Segmentation_WACV_2025_paper.pdf) · [arXiv:2412.13179](https://arxiv.org/abs/2412.13179)

This note walks the paper’s pipeline with the theory and math at each step. It is about **their** method (RGB or RGB+NIR, five models, GIS postprocess). fAIr / `pl-hot` implementation notes are only at the end.

---

## 0. Problem

US cities debate **minimum parking requirements**. Maps of **off-street surface parking lots** (the whole paved lot, not stall occupancy) are expensive to draw by hand. Existing work is mostly **object detection** of cars or stalls. This paper is **semantic segmentation**: every pixel is parking or not.

They also distinguish:

| Object | Meaning |
| --- | --- |
| Parking **space** | One stall |
| Parking **block** | A row / island of stalls |
| Parking **lot** | The paved off-street lot (this paper) |

**Contributions**

1. **ParkSeg12k**: 12,617 pairs of \(512\times 512\) tiles, ~35k lots, 45 US cities.
2. Five segmenters: FCN, DeepLabV3, SegFormer, Mask2Former, OneFormer.
3. Show that a **NIR** channel helps even after upsampling from coarser NAIP.
4. **Polygon postprocess**: holes, simplify, subtract buildings, subtract roads.

Best reported number: **OneFormer + RGB+NIR + full postprocess**, mIoU **84.9%**, PW **96.3%**.

---

## 1. End-to-end pipeline (Figure 6)

```text
TRAIN
  RGB (Google, 30 cm)  ±  NIR (NAIP, bilinear to 30 cm)
           +
  binary mask y ∈ {0,1}^{H×W}
           │
           ▼
  segmenter f_θ  →  logits  →  BCE-with-logits  →  update θ

INFER
  new tile(s) → f_θ → mask ŷ
           │
           ▼
  pixels → lat/lon polygons (GeoJSON)
           │
           ▼
  (i) drop holes < 60 m²
  (ii) Douglas–Peucker simplify
  (iii) ŷ ← ŷ \ buildings
  (iv) ŷ ← ŷ \ buffered OSM roads
```

Training is on **rasters**. Postprocess is on **vector polygons**. Metrics in Table 2 after (iii)/(iv) are therefore on the cleaned geometry, not raw logits.

---

## 2. Dataset construction

### 2.1 Labels

Sources: Parking Reform Network downtown lots + OSM in three cities. Students corrected in QGIS so polygons follow **pavement edges**, not parcel lines, and match current Google imagery.

Mask convention:

\[
y_{i,j} \in \{0,1\}
\quad\text{(0 = background, 1 = parking).}
\]

No empty tiles: every image has some parking. Mean parking fraction:

\[
\rho = \mathbb{E}\!\left[\frac{1}{HW}\sum_{i,j} y_{i,j}\right] \approx 0.21.
\]

So about **21% parking / 79% background**. That \(\rho\) is used again in the loss.

Stats: 297.7 km² tiles, 62.5 km² labeled parking, 35,127 lot polygons.

### 2.2 RGB tiles

Google Satellite exported at **30 cm/px**, **512×512**, 3 channels. Input tensor for RGB models:

\[
x_{\mathrm{RGB}} \in \mathbb{R}^{3 \times 512 \times 512}.
\]

### 2.3 NIR enrichment (optional 4th channel)

NAIP has RGB+NIR but as coarse as **1 m/px**. They keep Google RGB and **resample NIR** to 30 cm with **bilinear interpolation**.

For a query location \(u\) on the fine grid, if the four nearest coarse samples are \(I_{00}, I_{10}, I_{01}, I_{11}\) with fractional offsets \((\alpha,\beta)\in[0,1]^2\):

\[
I(u) = (1-\alpha)(1-\beta)\,I_{00}
+ \alpha(1-\beta)\,I_{10}
+ (1-\alpha)\beta\,I_{01}
+ \alpha\beta\,I_{11}.
\]

Theory: vegetation has high NIR reflectance. Lots are often edged by grass, so NIR sharpens **lot-vs-lawn** and reduces merging of nearby lots. Cost: the extra channel is **hallucinated detail** (no new high-frequency information below 1 m).

4-channel input:

\[
x_{\mathrm{RGB+NIR}} \in \mathbb{R}^{4 \times 512 \times 512}.
\]

First conv of each net is widened from 3 to 4 input channels for this setting.

Split: **90/10** random train/val on the 12,617 tiles. **Test** = 400 images from a **held-out city**.

---

## 3. Semantic segmentation (what the net computes)

A model \(f_\theta\) maps a chip to a **logit field**. For binary parking they treat it as one positive class vs background.

**Sigmoid** (maps logit to probability):

\[
\sigma(z) = \frac{1}{1+e^{-z}} \in (0,1).
\]

Predicted parking probability at pixel \(p\):

\[
s(p) = \sigma\!\big(f_\theta(x)_p\big).
\]

Hard mask at threshold \(\tau\) (implicitly \(0.5\) unless stated):

\[
\hat{y}(p) = \mathbb{1}\big[s(p) \ge \tau\big].
\]

This is **dense labeling**, not boxes: no NMS, no instance IDs.

### 3.1 Why transformers vs CNNs here

Lots vary hugely in **size and shape**, overlap cars/trees, and look like **roofs and roads**. CNNs have a limited receptive field unless dilated (DeepLab ASPP). Transformers mix information across the whole \(512\times 512\) tile (long-range context: “this grey patch is a lot because it sits next to that street grid”).

They **do not redesign** the five nets. Only:

- input stem if \(C=4\);
- output head for **2 labels** (parking + background).

Contrastive / multi-task losses in OneFormer are **dropped**; they only need semantic masks.

| Model | Idea | Paper backbone / pretrain |
| --- | --- | --- |
| FCN | Dense FCN on a classifier backbone | ResNet-50, COCO |
| DeepLabV3 | Atrous conv + ASPP multi-scale | ResNet-50, COCO |
| SegFormer | Hierarchical Mix Transformer + MLP decoder | **MiT-B0**, ADE20K |
| Mask2Former | Masked attention on predicted regions | Swin-L, Cityscapes |
| OneFormer | Task-token universal segmenter | Swin-L, COCO |

Atrous (dilated) convolution with rate \(r\) samples every \(r\)-th pixel so the kernel sees a larger field without extra params. ASPP runs several rates in parallel and concatenates.

SegFormer (Xie et al., 2021): overlapping patch embed + transformer blocks at 4 scales, then a lightweight MLP fuses them. Paper row is **B0 (3.7M params)** vs FCN **49.6M**.

---

## 4. Training

Hardware: RTX A4000 16 GB. Optimizer: **Adam**, \(\mathrm{lr}=10^{-5}\). Early stop **patience 10**.

### 4.1 Loss: BCE with logits + positive weight

Let \(x_n\) be the parking **logit** at pixel \(n\), \(y_n\in\{0,1\}\) the label. PyTorch `BCEWithLogitsLoss` is numerically stable BCE **after** an implicit sigmoid:

\[
L = \frac{1}{N}\sum_{n=1}^{N}
\Big[
-w_n\Big(
y_n\log\sigma(x_n)
+(1-y_n)\log\big(1-\sigma(x_n)\big)
\Big)
\Big].
\]

Here \(N\) is pixels in the batch (paper writes “batch size”; in code it is usually mean over pixels).

**Class imbalance.** If every pixel were weighted equally, the net can score well by predicting background. They set the **positive weight** from the dataset fraction \(\rho\approx 0.21\):

\[
w_n =
\begin{cases}
1/\rho \approx 4.76 & \text{if } y_n=1,\\
1 & \text{if } y_n=0.
\end{cases}
\]

In `BCEWithLogitsLoss` this is `pos_weight = 4.76` on the parking class: false negatives on parking cost ~4.76× more than false positives on background.

Adam (sketch): exponential moving averages of gradient \(m_t\) and squared gradient \(v_t\), bias-corrected, then

\[
\theta \leftarrow \theta - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t}+\varepsilon).
\]

No extra schedule is specified beyond early stopping.

---

## 5. Inference (before GIS)

1. Forward \(f_\theta(x)\) → logits (possibly at stride 4, then upsample to \(512\times 512\)).
2. \(\sigma\) (or softmax if a 2-channel head) → \(s(p)\).
3. Threshold → binary mask.
4. Known tile **bounding box** maps each pixel to lon/lat.
5. Raster mask → **polygons** (one GeoJSON for the test set).

Pixel \((i,j)\) in a tile with origin \((\lambda_0,\varphi_0)\) and GSD \(g=0.3\,\mathrm{m}\) (in a metric / projected system, then reproject):

\[
(X,Y) = \mathrm{origin} + (j,i)\cdot g.
\]

Then vectorize connected components of \(\{\hat{y}=1\}\).

---

## 6. Post-processing (on polygons)

Let \(\hat{P}\) be the predicted parking **multipolygon**. Operations are **set difference** in the plane (Shapely), not pixel morphology.

### 6.1 Holes and tiny blobs (\(60\,\mathrm{m}^2\))

Two error types:

- interior **gaps** inside a lot (false background islands);
- tiny **specks** that are themselves small polygons (false parking).

If a hole or component has area \(A < 60\,\mathrm{m}^2\), drop it. Threshold from trial-and-error: smaller holes are almost always errors; real islands exist but tend to be larger.

### 6.2 Douglas–Peucker simplify

A polyline \(p_0,\ldots,p_m\). Recursively: keep endpoints; find the vertex farthest from the chord \(p_0p_m\); if distance \(> \varepsilon\), keep it and split. Result: fewer vertices, similar shape.

Parking lots have **simple edges**. Jagged masks add false parking along the border and are hard to edit in GIS. They implement this via Mapshaper as a **maximum fraction of vertices removed** (not a metric \(\varepsilon\) in metres).

### 6.3 Buildings

Roofs look like lots. Yin et al. fed buildings as extra **input channels**. This paper subtracts Microsoft **building footprints** \(B\):

\[
\hat{P} \leftarrow \hat{P} \setminus B.
\]

### 6.4 Roads

OSM **centerlines** \(L\) (LineStrings). Buffer by lane-dependent width \(w(\text{lanes})\):

\[
R = \mathrm{buffer}(L, w),\qquad
\hat{P} \leftarrow \hat{P} \setminus R.
\]

Roads are the closest visual class to lots. Doing this in postprocess is cheaper than extra input channels.

Table 2 is **cumulative**: “w/ Building” = holes + simplify + buildings; “w/ Road” = that result minus roads.

Empirical gains (average across models): buildings block **+1.38 mIoU / +0.55 PW**; roads **+0.35 mIoU / +0.12 PW**. Weak CNNs gain more; transformers already leak less onto roofs/roads.

---

## 7. Metrics

Confusion over pixels (parking = positive):

\[
\mathrm{TP},\;\mathrm{FP},\;\mathrm{FN},\;\mathrm{TN}.
\]

**Pixel-wise accuracy (PW)** — fraction of pixels with the right class (parking **or** background):

\[
\mathrm{PW} = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}}.
\]

Dominated by background because \(\rho\approx 0.21\).

**IoU** of two sets \(A,B\):

\[
\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}.
\]

Per class:

\[
\mathrm{IoU}_{\mathrm{park}} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}},
\quad
\mathrm{IoU}_{\mathrm{bg}} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.
\]

**mIoU** = mean of class IoUs **including background**:

\[
\mathrm{mIoU} = \frac{1}{2}\big(\mathrm{IoU}_{\mathrm{bg}}+\mathrm{IoU}_{\mathrm{park}}\big).
\]

PW can look high (~95%) while lots are still sloppy; mIoU is the stricter overlap score.

---

## 8. Results (Table 2, condensed)

RGB, **raw** masks (no postprocess):

| Model | mIoU | PW |
| --- | --- | --- |
| FCN | 77.92 | 94.22 |
| DeepLabV3 | 79.62 | 94.76 |
| SegFormer B0 | 81.47 | 95.33 |
| Mask2Former | 82.04 | 95.22 |
| OneFormer | **83.23** | **95.72** |

NIR helps all models; **biggest lift on the weakest CNN** (FCN +1.3 mIoU). NIR reduces over-merge across grass and some roof confusion.

After RGB+NIR + buildings + roads, OneFormer reaches **84.86 / 96.34** (abstract rounds to 84.9 / 96.3). Transformers beat CNNs; SegFormer is the **cheap** transformer (3.7M vs FCN 49.6M, still +3.6 mIoU on RGB).

---

## 9. What this is not

- Not stall occupancy / car detection.
- Not instance IDs (two adjacent lots may merge until NIR/postprocess splits them).
- NIR is **US NAIP-specific**.
- Building/road subtract needs extra GIS layers, not just the satellite chip.

---

## 10. Relation to `pl-hot` / fAIr (not in the paper)

fAIr chips are **3-band RGB**, CPU **ONNX**. The authors’ **best** row (OneFormer + NIR + MS buildings + OSM roads) is out of scope for live `POST /predict`.

What we onboard: published **SegFormer-large** RGB weights (that file is **MiT-B5**, not the table’s B0), ImageNet 512 preprocess, decode + hole/simplify. Building/road subtract is left optional. Metrics in `pl-hot` use the same PW / 2-class mIoU names, on **raw pixels** unless you rasterize cleaned GeoJSON.

Implementation theory: [`../model/model.md`](../model/model.md).
