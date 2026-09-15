# Parking-lot SegFormer: architecture, inference, and evaluation

Theory for `pl-hot` against [Qiam et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Qiam_A_Pipeline_and_NIR-Enhanced_Dataset_for_Parking_Lot_Segmentation_WACV_2025_paper.pdf) and standard semantic-segmentation practice ([MMSegmentation `IoUMetric`](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/evaluation/metrics/iou_metric.py), Cityscapes). Inspect weights with `uv run --extra train python model/inspect_checkpoint.py`.

fAIr serve uses **RGB chips only**. The paper’s RGB+NIR gain is real and **out of scope** here.

---

## 1. Model architecture

The published file `SegFormer_large_parking.ckpt` is Hugging Face `SegformerForSemanticSegmentation` (MiT encoder + all-MLP decoder) inside a Lightning checkpoint.

| Part | What it is | This checkpoint |
| --- | --- | --- |
| Encoder | Mix Transformer (hierarchical, overlapping 7×7 patch embed) | **MiT-B5**, block depths `[3, 6, 40, 3]` |
| Decoder | All-MLP fuse of 4 scales | `decode_head.classifier` `(2, 768, 1, 1)` |
| Input | RGB NCHW | first conv `(64, 3, 7, 7)` → **3 channels** |
| Output | per-pixel logits | **2 classes**: background, parking |
| Spatial size of logits | stride 4 | for 512 input, typically **128×128**, then upsample |

**Paper vs this file.** The WACV table’s SegFormer row is **MiT-B0** (~3.7M params). The HF card “SegFormer-large” / this `.ckpt` is **B5**. Same family, heavier. Fine for fAIr RGB/ONNX; expect more CPU/RAM than B0.

**Loss (paper §4.1.6).** Binary parking vs not-parking, **BCE-with-logits**, positive weight \(w = 1/0.21 \approx 4.76\) because ~21% of pixels are parking. Adam, `lr=1e-5`, early-stop patience 10, 90/10 random split in the paper (fAIr uses **spatial OAM blocks** instead).

Hugging Face always emits `num_labels` channels. Training in `pl_hot.train` therefore collapses 2-class logits to one parking logit \(\ell_1 - \ell_0\) before BCE. That is the same as softmax-on-parking after sigmoid: \(\mathrm{softmax}([\ell_0,\ell_1])_1 = \sigma(\ell_1-\ell_0)\).

**Recommended for fAIr:** keep B5 + 2-class head as shipped; freeze encoder on small AOIs; do not add NIR.

---

## 2. Preprocessing (recommended)

Match ImageNet-pretrained SegFormer, not YOLO `/255` only.

1. Read GeoTIFF bands 1–3. Drop extra bands (no NIR).
2. Resize **square → square** to **512×512**, bilinear, **no letterbox**.
3. Scale to float32 `[0, 1]`.
4. ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
5. Layout **NCHW** `(1, 3, 512, 512)`.

OAM chips are often 256×256; the scale factors in `PreprocessMeta` map model pixels back to the chip.

---

## 3. Inference / decode (recommended)

There is **no NMS**. One forward pass is a dense map.

1. ONNX/Runtime: `image` → `logits` `(1, C, H, W)`.
2. If `C=2` (this ckpt): softmax over channel, keep parking = channel 1.
3. If `C=1` (paper-style BCE head): sigmoid.
4. Threshold **0.5** (paper default): `mask = (parking_prob >= 0.5)`.
5. Upsample logits **bilinear** to the chip / label size **before** thresholding (SegFormer / mmseg). Serve currently resizes the **binary** mask with nearest neighbour; torch `evaluate_segformer` interpolates logits. Prefer bilinear-on-logits if you change serve later.

`uint8` mask = `{0,1}` for vectorize. `float32` probability = polygon-interior mean confidence.

---

## 4. Postprocessing (recommended)

Paper §4.2 works on **polygons**, not on logits. Order:

| Step | Paper | In `pl-hot` |
| --- | --- | --- |
| Mask → polygons | GeoJSON from chip georef | `rasterio.features.shapes` |
| Holes | drop gaps **and** tiny blobs **< 60 m²** | `hole_area_m2=60`, `min_area_m2=60` |
| Edges | Douglas–Peucker (they used Mapshaper % of vertices) | Shapely `simplify` (`simplify_m=1.0`) |
| Buildings | subtract Microsoft footprints | **not implemented** (needs extra GIS, not on `POST /predict`) |
| Roads | OSM centerlines buffered by lane count, then subtract | **not implemented** |

Paper Table 2: hole+simplify+buildings ≈ **+1.4 mIoU**; roads ≈ **+0.35 mIoU**. Default fAIr path should keep hole fill + simplify + min area. Building/road subtract stays an optional later hook.

Do **not** score `evaluate_*` on these polygons unless you rasterize the cleaned GeoJSON back to the chip. Current eval is **raw pixels** (paper’s “before post-processing” column).

---

## 5. Evaluation — formulas and what was wrong

Paper names:

- **PW** (pixel-wise accuracy): fraction of pixels whose predicted class matches GT (parking or background).
- **mIoU**: mean of **per-class IoU**, **including background**.

This section explains exactly what `pl_hot.evaluate` computes and why.

### 5.1 From logits → probabilities → mask (what is being evaluated)

The model outputs **logits**, not probabilities. For each pixel \(p\):

- **Two-channel (this checkpoint)**: \(\ell_0(p)\) for background, \(\ell_1(p)\) for parking
- **One-channel (paper-style BCE head)**: a single parking logit \(\ell(p)\)

Convert logits to a parking probability \(s(p)\in[0,1]\):

- If \(C=1\): $(s(p)=\sigma(\ell(p))=\frac{1}{1+e^{-\ell(p)}}$)
- If \(C=2\): softmax over channels:

$[
P(\text{parking}\mid p)=\frac{e^{\ell_1(p)}}{e^{\ell_0(p)}+e^{\ell_1(p)}}
$]

There is a useful identity for $(C=2$):

$$
\frac{e^{\ell_1}}{e^{\ell_0}+e^{\ell_1}}
=
\frac{1}{1+e^{-(\ell_1-\ell_0)}}
=
\sigma(\ell_1-\ell_0)
$$


So “softmax parking prob” is the same as “sigmoid of logit difference”. This is why training collapses \(\ell_1-\ell_0\) before applying BCE-with-logits.

Then we create the **binary mask** $(\hat{y}(p)\in\{0,1\}$) with a threshold $(\tau$) (default $(\tau=0.5$)):

$[
\hat{y}(p)=\mathbb{1}\left[s(p)\ge\tau\right]
$]

In code:

- `evaluate_segformer()` bilinear-upsamples logits to the GT size, then thresholds (recommended practice).
- `evaluate_binary_masks()` reads PNGs and treats any value \(>0\) as foreground.

### 5.2 Confusion matrix (pixel counts)

Define the ground truth \(y(p)\in\{0,1\}\) and prediction \(\hat{y}(p)\in\{0,1\}\) for each pixel \(p\).

For the **parking** class (foreground), sum over all pixels in all chips:

$[
\mathrm{TP}=\sum_p \mathbb{1}[\hat{y}(p)=1 \wedge y(p)=1]
$]

$[
\mathrm{FP}=\sum_p \mathbb{1}[\hat{y}(p)=1 \wedge y(p)=0]
$]

$[
\mathrm{FN}=\sum_p \mathbb{1}[\hat{y}(p)=0 \wedge y(p)=1]
$]

$[
\mathrm{TN}=\sum_p \mathbb{1}[\hat{y}(p)=0 \wedge y(p)=0]
$]

Total pixels \(N=\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}\).

This is exactly what `_confusion()` accumulates inside `evaluate_binary_masks()`.

### 5.3 Pixel-wise accuracy (PW / aAcc)

Pixel-wise accuracy is the fraction of correctly classified pixels:

$[
\mathrm{PW} = \frac{\mathrm{TP}+\mathrm{TN}}{N}
$]

This is returned as `fair:accuracy`.

### 5.4 Intersection over Union (IoU) per class

IoU is “overlap divided by union” for each class.

For the **parking** class:

$[
\mathrm{IoU}_{\mathrm{parking}} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
$]

For the **background** class, you can think of background as “parking = 0”. The intersection is TN (both say background). The union is “all pixels that are background in either pred or GT”, which equals \(N - \mathrm{TP}\). Expanding that gives:

$[
\mathrm{IoU}_{\mathrm{bg}}
=
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}
$]

`evaluate.py` reports these as `iou_parking_lot` and `iou_background` when the denominator (union) is non-zero.

### 5.5 Mean IoU (mIoU) and how to average (micro vs macro)

The paper says “mIoU computed for all classes and the average is called mIoU”. For binary segmentation (2 classes):

$[
\mathrm{mIoU} = \frac{1}{2}\left(\mathrm{IoU}_{\mathrm{bg}} + \mathrm{IoU}_{\mathrm{parking}}\right)
$]

There are *two different averaging levels* people sometimes mix up:

- **Dataset-level (micro) aggregation (recommended; what mmseg/Cityscapes does):** sum TP/TN/FP/FN across all pixels first, then compute IoU from those totals once. This is what `evaluate_binary_masks()` does.
- **Per-image (macro) averaging:** compute IoU per chip, then average chips equally. This over-weights tiny/easy chips and is sensitive to empty images.

For fAIr, dataset-level aggregation is the safer default because chips can vary in content and (in general) size.

### 5.6 Empty-class edge case (why union==0 must not become 1.0)

If a class has **no pixels** in both GT and prediction, its union is 0. Example: a chip with no parking pixels and a model that predicts no parking pixels.

- Parking union \(= \mathrm{TP}+\mathrm{FP}+\mathrm{FN}=0\)
- “IoU = 1.0” would treat that chip as “perfect parking segmentation”, even though the chip contains no parking at all.

State-of-the-art evaluation implementations avoid inflating mIoU here by treating IoU as undefined for that class on that image, then skipping it in the mean, or by doing dataset-level aggregation (where empty chips contribute 0 to TP/FP/FN and therefore do not change the parking IoU).

In `pl_hot.evaluate`, `_iou()` returns NaN when union is 0, and `_metrics_from_confusion()` excludes NaNs from the mean. That aligns with common practice in segmentation tooling.

### 5.7 What `evaluate.py` did wrong (before the fix)

SOTA (mmseg `IoUMetric`, Cityscapes) computes these metrics from dataset-level (micro) pixel totals, not per-image averages:

| Issue | Old code | Why it is wrong |
| --- | --- | --- |
| Aggregation | mIoU and PW as **mean of per-chip** scores | SOTA is **micro** (pixel-weighted). Equal-weighting chips ≠ paper if chip sizes differ. |
| Empty parking | `union==0` → IoU **1.0** on that chip | An all-background chip with a correct empty pred is “perfect parking IoU”. That **inflates** mean mIoU. Dataset-level parking union ignores those chips (they add 0 to TP/FP/FN). mmseg uses NaN / skip when class union is 0. |
| Protocol | raw masks only | Fine if documented. Not comparable to paper rows **after** building/road subtract. |

Per-class IoU on a **single** chip is the same formula as dataset-level; the bug appears as soon as you average chips (especially empties).

`evaluate_binary_masks` now accumulates TP/TN/FP/FN globally. Class IoU with union 0 is omitted from the mean (not scored as 1.0).

Returned keys (from this library) are what `pipeline.py` `evaluate_model` should `log_evaluation_results(...)` and return:

| Library key | After fAIr log | STAC property | Meaning |
| --- | --- | --- | --- |
| `accuracy` | `fair/accuracy` | `fair:accuracy` | PW |
| `mean_iou` | `fair/mean_iou` | `fair:mean_iou` | mean of class IoUs (background + parking) |
| `iou_background` | `fair/iou_background` | `fair:iou_background` | background IoU |
| `iou_parking_lot` | `fair/iou_parking_lot` | `fair:iou_parking_lot` | parking IoU (`classification:classes` name) |

Declare the same names in the future pack’s `stac-item.json` `fair:metrics_spec` so promotion copies them onto the local model item.

```json
"fair:metrics_spec": [
  {"key": "fair:accuracy", "name": "Pixel-wise accuracy", "description": "Fraction of correctly classified pixels (parking + background) on the val split, micro-averaged."},
  {"key": "fair:mean_iou", "name": "Mean IoU", "description": "Mean of background IoU and parking_lot IoU, computed from dataset-level confusion counts."},
  {"key": "fair:iou_parking_lot", "name": "Parking IoU", "description": "Foreground-class IoU: TP / (TP+FP+FN) summed over all val pixels."},
  {"key": "fair:iou_background", "name": "Background IoU", "description": "Background-class IoU: TN / (TN+FP+FN) summed over all val pixels."}
]
```

Torch `evaluate_segformer` still bilinear-upsamples logits to the GT mask, thresholds at 0.5, then calls that scorer. It does not run hole fill or GIS subtract.

### Not used (and not required)

- Dice / F1: useful but not the paper’s table.
- Instance / polygon IoU: different task (dinov3 buildings). This model is **semantic**, not instance.
- Argmax vs threshold 0.5 on two classes: equivalent to \(\ell_1 \ge \ell_0\).
