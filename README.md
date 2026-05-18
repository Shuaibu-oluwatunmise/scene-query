# scene-query

**Video or images in. Queryable 3D scene out.**

Most reconstruction pipelines produce a point cloud or a Gaussian splat — a rendering artefact. This produces something a humanoid robot can actually use: a *scene memory* where objects have labels, space has structure, and you can ask questions in natural language.

Built for Humanoid's Perception & Spatial AI internship challenge.

---

## Pipeline

```
Video / images
      │
      ▼
  VGGT-Omega (Meta, CVPR 2026)     ← feed-forward, no per-scene optimisation
  camera poses + dense depth map
      │
      ├─────────────────────────────────┐
      ▼                                 ▼
  YOLOv8 office detector           depth → point cloud
  per-frame bounding boxes         (photo-coloured, full scene)
  (10 classes: bottle, chair,
   keyboard, monitor, mouse,
   mug, notebook, pen,
   printer, stapler)
      │
      ▼
  Mask lifting
  unproject masked depth → world-space labelled points
      │
      ▼
  Labelled point cloud  [outputs/scene/pointcloud.npz]
  XYZ + RGB + label + confidence, per point
      │
      ▼
  Query engine  (CPU, no models)
  "find the chair" → 3D centroid + bounding region + Rerun visualisation
```

The YOLO model is trained on 10 office classes with mAP@0.5 = 98.3%. Weights ship in the repo (`checkpoints/yolo_office/`).

---

## Quick start

### Prerequisites

- Python 3.10+
- CUDA GPU with ≥ 16 GB VRAM for `reconstruct.py`
- `query.py` runs fine on CPU / laptop — no GPU needed

### 1. Clone

```bash
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query
```

### 2. Install PyTorch

Pick the version matching your CUDA driver (`nvidia-smi` shows it in the top-right):

```bash
# CUDA 12.4 — find yours at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Run setup

```bash
python setup.py
```

This installs all Python dependencies and downloads VGGT-Omega weights (~4.3 GB) from Google Drive. YOLO weights are already in the repo — no extra download needed.

### 4. Reconstruct

```bash
python reconstruct.py \
    --video examples/test_scenery1.mp4 \
    --out outputs/scene1 \
    --save-rrd outputs/scene1.rrd
```

### 5. Query

```bash
python query.py outputs/scene1 "find the chair" --save-rrd outputs/query.rrd
python -m rerun outputs/query.rrd
```

---

## What you see in Rerun

**Reconstruction** (`--save-rrd` on reconstruct.py) — 3-panel layout, timeline at 0.1× speed:

```
┌──────────────────┬──────────────────┬──────────────────────────┐
│  Camera feed     │  Camera feed +   │  3D scene                │
│  (raw)           │  YOLO detections │  photo-coloured cloud +  │
│                  │  overlaid        │  moving camera           │
└──────────────────┴──────────────────┴──────────────────────────┘
```

**Query** (`query.py`) — 5-panel layout:

```
┌──────────────────┬──────────────────┬──────────────────┐
│  Camera feed     │  3D view through │  Segmentation    │
│  (timeline)      │  moving camera   │  overlay         │
│                  │  (timeline)      │  (timeline)      │
├────────────────────────┬────────────────────────────────┤
│  Grey scene +          │  Photo-coloured scene +        │
│  queried label         │  tight bounding box            │
│  highlighted (static)  │  (static)                      │
└────────────────────────┴────────────────────────────────┘
```

---

## All flags

### reconstruct.py

```
--video FILE          Input video (auto-saves frames)
--images DIR          Input image directory
--out DIR             Output scene directory (required)
--fps FLOAT           Frame sample rate (default: 2.0, video only)
--max-frames INT      Cap frames fed to VGGT (default: 50)
--backend yolo|gsam2  Semantics backend (default: yolo)
--save-rrd FILE       Also save a Rerun .rrd recording
--device cuda|cpu     (default: cuda)
--geometry-only       Skip semantics — VGGT geometry output only
```

### query.py

```
SCENE_DIR             Path to reconstructed scene directory
QUERY                 Natural-language object query, e.g. "find the chair"
--save-rrd FILE       Save Rerun .rrd (default: <scene_dir>/query.rrd)
--free-space          Query navigable floor space instead of an object
--reachable-from X Y Z --reach FLOAT
                      Objects within arm reach of a robot base position
```

---

## Output format

```
outputs/scene/
├── pointcloud.npz      # XYZ, RGB, label, confidence — one row per labelled point
├── scene_cloud.npz     # full VGGT cloud for background visualisation
├── poses.npz           # camera-to-world 4×4 matrices, one per frame
├── intrinsics.npz      # per-frame camera intrinsics + VGGT image_size
├── labels.json         # label → point-index list for fast lookup
└── frames/             # extracted frames (only when input was --video)
```

---

## Repository layout

```
scene-query/
├── reconstruct.py           # video/images → labelled scene directory
├── query.py                 # scene directory → answer + Rerun visualisation
├── setup.py                 # one-shot setup: downloads weights + installs deps
├── src/scene_query/
│   ├── geometry.py          # VGGT-Omega wrapper (poses + depth + point cloud)
│   ├── semantics.py         # YOLO detection backend (+ GSAM2 fallback)
│   ├── lift.py              # unproject masked depth into world-space labels
│   └── query_engine.py      # text + spatial query logic
├── scripts/
│   ├── download_models.py   # download VGGT-Omega + YOLO weights from Google Drive
│   └── install_deps.py      # install requirements.txt + vggt-omega package
├── checkpoints/
│   ├── yolo_office/         # YOLOv8 weights + training artefacts (in git)
│   └── vggt_omega/          # VGGT-Omega weights (downloaded by setup.py)
├── examples/
│   ├── tabletop/            # 25 still images of a tabletop scene
│   └── test_scenery*.mp4    # office scene videos
└── docs/
    └── design_note.md       # rationale: choices, tradeoffs, robot relevance
```
