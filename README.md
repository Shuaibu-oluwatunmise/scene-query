# scene-query

**Images or video in. Queryable 3D scene out.**

Most reconstruction pipelines produce a point cloud or a Gaussian splat — a rendering artefact. This project produces something a humanoid robot can actually use: a *scene memory* where objects have labels, space has structure, and you can ask questions.

Built for Humanoid's Perception & Spatial AI internship challenge. See [docs/design_note.md](docs/design_note.md) for the full rationale.

---

## Quick start

### Prerequisites

- Python 3.10+
- CUDA GPU with **16 GB+ VRAM** (tested: RTX 6000 Ada 48 GB, CUDA 12.4)

### 1. Clone

```bash
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query
```

### 2. Install PyTorch

Install the version matching your CUDA driver — check with `nvidia-smi` (CUDA version in top-right corner):

```bash
# CUDA 12.4 example — pick yours at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model weights

```bash
# VGGT — geometry model (~2 GB)
python scripts/download_vggt.py

# SAM 2.1 + Grounding DINO — segmentation models (~3.5 GB)
python scripts/download_sam2.py
python scripts/download_grounding_dino.py
```

### 5. Install Grounded-SAM-2 from source

```bash
git clone --depth=1 https://github.com/IDEA-Research/Grounded-SAM-2.git /tmp/gsam2
pip install -e /tmp/gsam2
pip install --no-build-isolation -e /tmp/gsam2/grounding_dino

# Make it importable
python -c "
import site, pathlib
pth = pathlib.Path(site.getsitepackages()[0]) / 'grounded_sam2.pth'
pth.write_text('/tmp/gsam2\n')
print('Written:', pth)
"
```

### 6. Run

**Option A — your own video:**

```bash
# Reconstruct: label whatever objects are in your scene
python reconstruct.py --video myvideo.mp4 --out outputs/myscene/ \
    --labels "table,chair,monitor,keyboard"

# Query an object and generate the 5-panel Rerun recording
python query.py outputs/myscene/ "find the table" --save-rrd outputs/table.rrd

# View
python -m rerun outputs/table.rrd
```

**Option B — included test images** (25 tabletop images, no download needed):

```bash
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/ \
    --labels "table,chair,mug"

python query.py outputs/tabletop/ "find the table" --save-rrd outputs/table.rrd

python -m rerun outputs/table.rrd
```

> A test video will be added to `examples/` shortly for an out-of-the-box video demo.

---

## What you'll see in Rerun

A 5-panel layout focused on your queried object:

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

- **Top row** — scrub through frames: original feed, 3D reconstruction locked to the camera, back-projected segmentation mask
- **Bottom row** — free-orbit static views: label points highlighted against the grey scene (left), bounding box over the photo-coloured scene (right). Both start from the first camera frame's point of view.

---

## Pipeline

```
Images or video
      │
      ▼
  VGGT (Meta, 2024)          ← feed-forward, no per-scene optimisation
  camera poses + dense depth + point cloud
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
  Grounded-SAM-2                        point cloud
  per-frame semantic masks (text-prompted)
      │
      ▼
  Mask lifting  [src/scene_query/lift.py]
  unproject masked depth → world-space labelled points
      │
      ▼
  Labelled point cloud  [outputs/room/pointcloud.npz]
  XYZ + RGB + label + confidence, per point
      │
      ▼
  Query engine + Rerun visualisation
  "find the table" → 3D bounding region + centroid
```

---

## Output format

```
outputs/myscene/
├── pointcloud.npz     # XYZ, RGB, label (str), confidence (float), per point
├── poses.npz          # camera-to-world 4x4 matrices, one per frame
├── intrinsics.npz     # camera intrinsics (per frame) + VGGT image_size
├── depth/             # per-frame metric depth maps (.npy)
├── labels.json        # label -> point-index list, for fast lookup
└── frames/            # extracted frames (only when input was --video)
```

---

## All query types

```bash
# Find a specific object
python query.py outputs/myscene/ "find the chair"

# Navigable floor space
python query.py outputs/myscene/ --free-space --floor-z 0.0

# Objects within arm reach of a robot base position
python query.py outputs/myscene/ --reachable-from 0.0 0.0 0.9 --reach 0.7
```

---

## On RunPod or persistent GPU machines

Pass `--dir` to keep weights on the volume so they survive pod restarts:

```bash
python scripts/download_vggt.py --dir /workspace/checkpoints/vggt
python scripts/download_sam2.py --dir /workspace/checkpoints/sam2
python scripts/download_grounding_dino.py --dir /workspace/checkpoints/grounding_dino
```

After a pod migration, restore the full environment in one command:

```bash
bash scripts/setup_pod.sh
```

---

## Repository layout

```
scene-query/
├── reconstruct.py              # entry point: images/video -> scene dir
├── query.py                    # entry point: scene dir -> answers + Rerun vis
├── src/scene_query/
│   ├── geometry.py             # VGGT wrapper (pose + depth + point cloud)
│   ├── semantics.py            # Grounded-SAM-2 wrapper (per-frame masks)
│   ├── lift.py                 # unproject masked depth into world-space labels
│   └── query_engine.py         # text + spatial query logic
├── scripts/
│   ├── download_vggt.py        # install VGGT + download weights
│   ├── download_sam2.py        # download SAM 2.1 weights
│   ├── download_grounding_dino.py  # download Grounding DINO weights
│   ├── download_examples.py    # download additional example images
│   ├── test_vggt.py            # smoke test: VGGT on example images -> .ply
│   └── setup_pod.sh            # restore pip environment after pod migration
├── docs/
│   └── design_note.md          # rationale: choices, tradeoffs, robot relevance
├── checkpoints/
│   └── README.md               # documents where model weights live
└── examples/
    └── tabletop/               # 25 images of a tabletop scene (00.png–24.png)
```
