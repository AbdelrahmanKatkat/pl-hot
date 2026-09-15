# `pl-hot`

Python library for the HOT **parking-lot segmenter** on fAIr. It owns the model logic — RGB chip preprocessing, ONNX mask decode, polygonize / clean, georeferencing, GeoJSON output, and SegFormer train/export helpers — so the fAIr model pack (`fAIr-models/models/<segformer_parking>/`) stays a thin STAC/ZenML/Docker adapter.

This README is the **implementation contract** (same role as [`spd-hot`](https://github.com/AbdelrahmanKatkat/spd-hot)). `pl-hot` now ships the core inference, training, evaluation, export, and validation utilities described below. Do not put ZenML, STAC, or Docker orchestration in this library.

**Repo:** [AbdelrahmanKatkat/pl-hot](https://github.com/AbdelrahmanKatkat/pl-hot)  
**Python:** ≥ 3.12

---

## Theory first: why this is not a copy of `spd-hot`

`spd-hot` wraps **YOLO object detection**. One forward pass returns boxes. Postprocess is threshold + NMS + bbox → polygon.

`pl-hot` wraps **semantic segmentation**. One forward pass returns a **per-pixel** parking / not-parking map. There is no NMS and no class-box tensor. Postprocess is:

1. logits → binary mask
2. mask → polygons
3. polygon cleanup (holes, simplify)
4. chip pixels → EPSG:4326 GeoJSON

That is the same *family* as fAIr `unet_segmentation` and `dinov3s_buildings`, not `yolo11m_swimming_pools`. Reuse the **library shape** from `spd-hot` (params / preprocess / decode / georef / postprocess / serve / train / export). Replace every YOLO-specific step.

| Question | Swimming pools (`spd-hot`) | Parking lots (`pl-hot`) |
| --- | --- | --- |
| fAIr task | `object-detection` | `semantic-segmentation` |
| Head | YOLO11m boxes | SegFormer per-pixel logits |
| Native tile | often 256, model 640 | paper tiles **512×512**, OAM chips often 256 → resize to 512 |
| ONNX input | `(1, 3, 640, 640)` NCHW `[0, 1]` | `(1, 3, 512, 512)` NCHW; **ImageNet mean/std** (verify from checkpoint) |
| ONNX output | `(1, 4+nc, anchors)` | `(1, 1, 512, 512)` or `(1, 2, 512, 512)` logits |
| Decode | conf + NMS → xyxy | sigmoid / softmax → binary mask |
| Vectorize | bbox corners | `rasterio.features.shapes` on the mask |
| Cleanup | min box area | fill holes, simplify edges; optional building/road subtract |
| Train labels | GeoJSON → YOLO `.txt` | GeoJSON → raster mask (or PNG mask pairs) |
| Train stack | Ultralytics | Hugging Face SegFormer / the published `.ckpt` |
| Inference image | CPU `onnxruntime` | CPU `onnxruntime` (no Transformers) |

fAIr only accepts **3-band RGB** chips and **vector** output ([contributing/model.md](https://github.com/hotosm/fAIr-models/blob/develop/docs/contributing/model.md)). The paper’s RGB+NIR gain is real, but **NIR is out of scope** for the fAIr pack. Train and serve the **RGB** checkpoint.

---

## What it is

`pl_hot` is a reusable package, not a standalone service.

| Area | Modules | Purpose |
| --- | --- | --- |
| **Inference** | `preprocess`, `decode`, `georef`, `postprocess`, `serve` | GeoTIFF chips → ONNX mask → EPSG:4326 GeoJSON |
| **Training** | `dataset`, `geo_to_mask`, `train`, `evaluate`, `export` | GeoJSON + chips → SegFormer fine-tune → mIoU / PW → ONNX bytes |
| **Config** | `params` | Defaults and dict → dataclass parsing (STAC hyperparameters) |

Core deps stay light (`numpy`, `Pillow`, `pyproj`, `rasterio`, `shapely`). `transformers` / Torch / ONNX export are **`[train]` extras**. The live-serve image must not import them.

---

## Upstream model and data

| Item | Value |
| --- | --- |
| Architecture | **SegFormer-large** (MiT encoder + all-MLP decoder) |
| Weights | [UTEL-UIUC/SegFormer-large-parking](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking) (`best_model.ckpt`) |
| Paper | [Qiam et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Qiam_A_Pipeline_and_NIR-Enhanced_Dataset_for_Parking_Lot_Segmentation_WACV_2025_paper.pdf) |
| Official mapping tool (reference, do not vendor) | [UTEL-UIUC/parking-lot-mapping-tool](https://github.com/UTEL-UIUC/parking-lot-mapping-tool) |
| Dataset (paper / official) | [UTEL-UIUC/parkseg12k](https://huggingface.co/datasets/UTEL-UIUC/parkseg12k) · [ParkSeg12k](https://github.com/UTEL-UIUC/ParkSeg12k) |
| Dataset (HF mirror you linked) | [carosellaja1/parkinglotseg](https://huggingface.co/datasets/carosellaja1/parkinglotseg) — `rgb`, `nir`, `mask`; train 11,355 / test 1,262 |
| Classes | binary — `parking_lot` (1) vs background (0) |
| Paper tile | 512×512 PNG, ~30 cm/px, Google RGB (optional NAIP NIR upsample) |
| Paper train setup | 90/10 split, Adam, `lr=1e-5`, BCE-with-logits, positive weight `1/0.21 ≈ 4.76`, early stop patience 10 |
| Paper metrics | pixel-wise accuracy (PW), mIoU (mean over background + parking) |
| Paper SegFormer (table) | MiT-B0, ADE20K pretrain, RGB mIoU ~81.5 before postprocess |
| HF card | “SegFormer-large”; the published ckpt is **MiT-B5**, 3-channel RGB, 2-class decode head |

The published HF model is **not** the paper’s best row (that is OneFormer + RGB+NIR + full postprocess, mIoU 84.9%). Onboard **SegFormer-large + RGB** because it exports cleanly to ONNX and matches fAIr’s 3-band contract.

---

## Functionality overview

### Inference path

```text
GeoTIFF chip dir  (fAIr / OAM, often 256×256 RGB)
    │
    ▼
preprocess_chip_for_onnx()     RGB → resize 512×512 → ImageNet norm → (1,3,512,512)
    │                          + PreprocessMeta (scale, transform, crs)
    ▼
onnxruntime session.run()      (provided by fAIr, not imported here)
    │
    ▼
decode_segformer_onnx_output() sigmoid/softmax → binary mask at model size
    │
    ▼
unscale mask to chip pixels
    │
    ▼
mask_to_feature_collection()   shapes → fill holes → simplify → EPSG:4326 polygons
    │
    ▼
GeoJSON FeatureCollection
```

fAIr entry: `pl_hot.serve.predict_session(session, input_images, params)`.

### Training path

```text
chips/ (RGB GeoTIFF) + labels/ (one .geojson of parking polygons)
    │
    ▼
prepare_seg_dataset_from_geojson()
    │  spatial split (OAM-{x}-{y}-{z} tile blocks)
    │  rasterize polygons onto each chip (0/1 mask)
    ▼
train_segformer()              fine-tune from HF / best_model.ckpt
    │                          BCE-with-logits, pos_weight ≈ 4.76
    │                          device = cuda if visible else cpu
    ▼
evaluate_segformer()           PW + mIoU → fAIr metric names ("accuracy", "mean_iou")
    │
    ▼
export_onnx_bytes()            validated ONNX for promotion
```

Do **not** convert labels to YOLO `.txt`. That is detection-only.

---

## Preprocess

`pl_hot.preprocess.preprocess_chip_for_onnx(path, cfg)` converts one RGB chip to SegFormer input.

**Why these steps**

- fAIr chips are 3-band GeoTIFFs. The paper trains on 512×512 RGB. OAM chips are often 256×256, so resize (square → square, **no letterbox**), same idea as `spd-hot` 256→640.
- Hugging Face SegFormer is almost always trained with **ImageNet** mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` on float `[0, 1]`. YOLO’s “divide by 255 only” will shift this model. Confirm against the checkpoint / training script; if the ckpt used `/255` only, set `imagenet_norm=False`.
- Tensor layout is **NCHW** for ONNX Runtime.

**Steps**

1. Open with `rasterio`, read bands 1/2/3 as HWC uint8. Ignore extra bands (no NIR).
2. Resize to `model_input_size` (default **512**) with bilinear.
3. Scale to float32 `[0, 1]`.
4. If `imagenet_norm=True`, subtract mean / divide std per channel.
5. Return `(1, 3, 512, 512)`.

**Metadata (`PreprocessMeta`)** — same fields as `spd-hot`:

| Field | Use |
| --- | --- |
| `width`, `height` | Original chip size |
| `scale_x`, `scale_y` | Model pixels → chip pixels (mask unscale) |
| `transform`, `crs` | Chip pixels → map coordinates |

`PreprocessParams`: `model_input_size=512`, `chip_size_hint=256`, `normalize_01=True`, `imagenet_norm=True`.

---

## Postprocess

Split decode (model space) from georef (map space), like `spd-hot`.

### Decode (`pl_hot.decode`)

`decode_segformer_onnx_output(output, threshold)`:

- Accept `(1, 1, H, W)` logits **or** `(1, 2, H, W)` class scores.
- Binary: `sigmoid(logit) > threshold` (paper uses BCE-with-logits; default threshold **0.5**).
- Two-class: `argmax` or parking-channel softmax > threshold.
- Return a `uint8` mask at **model** resolution (`1` = parking).

No NMS. No Ultralytics.

### Georef + GeoJSON (`pl_hot.postprocess` + `pl_hot.georef`)

`mask_to_feature_collection(mask, meta, cfg)`:

1. **Unscale** the mask to original chip size (nearest-neighbour).
2. `rasterio.features.shapes` → polygons in chip CRS (same idea as `unet_segmentation` / `dinov3_hot.postprocess.vectorize_binary_mask`).
3. Paper cleanup ([WACV 2025 §4.2](https://openaccess.thecvf.com/content/WACV2025/papers/Qiam_A_Pipeline_and_NIR-Enhanced_Dataset_for_Parking_Lot_Segmentation_WACV_2025_paper.pdf)):
   - **Fill / drop holes** smaller than **60 m²** (interior rings and tiny blobs).
   - **Douglas–Peucker** simplify (parking edges are simple; raw masks are noisy).
   - Drop polygons below `min_area_m2`.
4. Reproject to **EPSG:4326**.
5. Emit GeoJSON with `class_id`, `class_name="parking_lot"`, optional `confidence` (mean probability inside the polygon), `source`.

**Optional (not implemented in `pl-hot` v0.1.0):** subtract Microsoft building footprints and OSM road buffers. The paper gains ~1–2 mIoU from this, but fAIr `POST /predict` only has the chip + TMS. This remains an extension point (`subtract_buildings`, `subtract_roads`) and is intentionally left out of the default pipeline.

`InferenceParams`: `mask_threshold=0.5` (also accepts fAIr `confidence_threshold`).  
`PostprocessParams`: `min_area_m2=60`, `hole_area_m2=60`, `simplify_m=1.0`.

---

## Parameters (defaults)

| Dataclass | Key knobs | Defaults |
| --- | --- | --- |
| `PreprocessParams` | `model_input_size`, `chip_size_hint`, `normalize_01`, `imagenet_norm` | 512, 256, true, true |
| `InferenceParams` | `mask_threshold` (alias: `confidence_threshold`) | 0.5 |
| `PostprocessParams` | `min_area_m2`, `hole_area_m2`, `simplify_m` | 60, 60, 1.0 |
| `SplitParams` | `val_ratio`, `split_seed`, `block_size` | 0.1 (paper 90/10), 42, 4 |
| `TrainParams` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `pos_weight`, `early_stop_patience`, `freeze_encoder`, `sample_fraction`, `device` | 20, 4, 1e-5, 0.0, 4.76, 10, true, 1.0, `cpu` |

fAIr passes STAC dicts; `parse_*` applies defaults. `device` is `"0"` / `"cuda"` when `torch.cuda.is_available()`, else `"cpu"` — same rule as the swimming-pool pack. Inference never reads `device`.

Map metrics to fAIr keys: PW → `fair:accuracy`, mIoU → `fair:mean_iou`.

---

## Package layout

```text
src/pl_hot/
  params.py                 # defaults + parsing
  preprocess.py             # chip → NCHW tensor + PreprocessMeta
  decode.py                 # ONNX logits → binary mask
  georef.py                 # chip pixels → EPSG:4326
  postprocess.py            # mask → cleaned GeoJSON
  serve.py                  # predict_session()
  dataset.py                # GeoJSON + chips → image/mask folders
  geo_to_mask.py            # polygon → 0/1 raster aligned to chip
  train.py                  # SegFormer fine-tune wrapper
  evaluate.py               # model-driven PW + mIoU
  export.py                 # ONNX export bytes
  onnx_adapter.py           # LogitsOnly wrap for torch.onnx.export ([train] only)
  checkpoint.py             # checkpoint contract inspection (shape/norm/channels)
tests/
  test_preprocess.py
  test_decode.py
  test_postprocess.py
  test_dataset.py
  test_evaluate.py
  test_serve.py
```

Tests use synthetic GeoTIFFs and fake logits — no HF download.

---

## Dockerfile recommendation (fAIr pack, not this repo)

Copy the swimming-pool **five-stage** layout. Change the training wheels to SegFormer, not Ultralytics.

| Stage | What to install | Device |
| --- | --- | --- |
| **builder → runtime** (train / batch) | Torch from `whl/cu126` (CUDA; still runs on CPU if no GPU), `transformers`, `onnx`, `onnxscript`, `rasterio`, `pyproj`, `shapely`, `fair-py-ops[k8s,serve]`, `git+https://github.com/AbdelrahmanKatkat/pl-hot` | GPU if visible, else CPU |
| **test** | runtime + `fair-py-ops[test]` | CPU in CI (`FAIR_FORCE_CPU=1`) |
| **inference-builder → inference** | `fair-py-ops[serve]`, **`onnxruntime==1.28.0` (CPU)**, rasterio, pyproj, shapely, numpy, Pillow, `pl-hot` **without** `[train]` | **CPU only** |

Do **not** put `transformers` or Torch in the distroless inference image. Do **not** use `onnxruntime-gpu` for live serve.

Expected size order of magnitude (from the pool pack): inference ~1.2 GB; CUDA training image ~5.5–7 GB after the cu126 wheel.

STAC: `mlm:tasks=["semantic-segmentation"]`, `mlm:accelerator=cuda`, input shape `[-1, 3, 512, 512]`, class name `parking_lot`, keyword `parking_lot` (or `landuse`), geometry `polygon`.

---

## Checkpoint (read the published file, then freeze the contract)

`checkpoint.py` **reads** a Lightning/HF SegFormer `.ckpt`. It is not TerraTorch, TorchGeo, or [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) (Keras). Those stacks use different module names and will not load this state dict.

The local file is `model/SegFormer_large_parking.ckpt`. Keys are `model.segformer.encoder.*` and `model.decode_head.classifier.*` (Hugging Face `SegformerForSemanticSegmentation` inside PyTorch Lightning).

| README question | From the ckpt tensors |
| --- | --- |
| Encoder | **MiT-B5** (`block` depths `[3, 6, 40, 3]`) |
| Input size | **512** (paper / ADE B5; not stored in the weight file) |
| Output tensor | **2** classes, classifier shape `(2, 768, 1, 1)` |
| Normalization | **ImageNet** (HF SegFormer default; not stored in the ckpt) |
| First conv | `(64, 3, 7, 7)` → **3** RGB channels (safe for fAIr) |

Training defaults to `nvidia/segformer-b5-finetuned-ade-512-512`, then `load_segformer_from_checkpoint(path)` loads this Lightning file.

### Run the inspector

The CLI lives under `model/`, not in `src/`. It `torch.load`s the ckpt (needs `[train]`) and prints encoder/channel shapes from the tensors:

```bash
cd fair_models/parking_lot/pl-hot
uv run --extra train python model/inspect_checkpoint.py
# or: uv run --extra train python model/inspect_checkpoint.py model/SegFormer_large_parking.ckpt
```

A `.ckpt` filename is not `.zip`, but Lightning stores it as a zip of pickle + shards. The script does not unzip it; `torch.load` opens that format.

Tile size 512 and ImageNet norm are not in the weight file. They come from the paper / HF SegFormer-B5 card.

### MiT-B5 vs fAIr live inference

fAIr does not ban B5. Live serve is **CPU ONNX** in a distroless image (same contract as `dinov3s_buildings`). B5 is larger than the paper’s MiT-B0 table row (~80M vs ~4M params), so KNative cold start and CPU latency will be higher. Plan STAC `fair:memory_request` / `fair:memory_limit` in the same band as dinov3s (several GiB), not a tiny YOLO box. Training stays CUDA when a GPU is visible.

---

## Install and test

```bash
cd fair_models/parking_lot/pl-hot
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
uv pip install -e ".[train,test]"   # Torch / transformers / onnx
uv run --extra test pytest -q
```

Current local suite:

- `test_preprocess.py` - chip normalization and tensor shape
- `test_decode.py` - 1-channel and 2-channel ONNX decode paths
- `test_postprocess.py` - polygonization and polygon-interior confidence
- `test_dataset.py` - GeoJSON rasterization and split dataset prep
- `test_evaluate.py` - PW/mIoU metric mapping to `fair:*`
- `test_serve.py` - end-to-end `predict_session` behavior with a fake ONNX session
- `test_checkpoint.py` - MiT-B5 depth inference from state-dict keys

---

## fAIr integration (thin adapter)

```python
from pl_hot.serve import predict_session

def predict(session, input_images, params):
    return predict_session(session, input_images, params)
```

fAIr owns STAC, ZenML, Docker, and `load_session`. `pl-hot` owns everything after the ONNX session exists.

---

## License and citation

- This library (when published): match the fAIr pack license you choose (recommend Apache-2.0 if the HF weights stay Apache-2.0).
- Weights: [Apache-2.0](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking).
- Labels: derived from Parking Reform Network + OSM (ODbL for OSM-derived data).

```bibtex
@inproceedings{qiam2025pipeline,
  title={A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation},
  author={Qiam, Shirin and Devunuri, Saipraneeth and Lehe, Lewis J},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1227--1236},
  year={2025},
  organization={IEEE}
}
```
