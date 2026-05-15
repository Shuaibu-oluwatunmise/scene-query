# scene-query

**Phone video in. Queryable 3D scene out.**

Most reconstruction pipelines produce a point cloud or a Gaussian splat — a rendering artefact. This project produces something a humanoid robot can actually use: a *scene memory* where objects have labels, space has structure, and you can ask questions.

```bash
python reconstruct.py examples/room_test.mp4 --out outputs/room/
python query.py outputs/room/ "find the chair"
python query.py outputs/room/ --free-space --floor-z 0.0
python query.py outputs/room/ --reachable-from 0.0 0.0 0.9
```

Built for Humanoid's Perception & Spatial AI internship challenge. See [docs/design_note.md](docs/design_note.md) for the full rationale.

## Pipeline

```
Phone video (30–60 s)
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
  "find the chair" → 3D bounding region + centroid
```

## Output format

```
outputs/room/
├── pointcloud.npz     # XYZ, RGB, label (str), confidence (float), per point
├── poses.npz          # camera-to-world 4×4 matrices, one per frame
├── depth/             # per-frame metric depth maps (.npy)
└── labels.json        # label → point-index list, for fast lookup
```

## Usage

```bash
# Full reconstruction
python reconstruct.py <video> --out <dir> [--fps 5] [--labels "chair,table,door"]

# Query by object name
python query.py <dir> "find the chair"

# Query for navigable free space
python query.py <dir> --free-space [--floor-z <metres>]

# Query for objects reachable from a robot base position
python query.py <dir> --reachable-from <x> <y> <z> [--reach 0.7]
```

## Getting started

**Requirements:** Python 3.10+, CUDA-capable GPU (tested: RTX 6000 Ada, CUDA 12.4)

```bash
# 1. Clone
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query

# 2. Install PyTorch (pick the right CUDA version for your machine)
#    https://pytorch.org/get-started/locally/
#    e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install everything else
pip install -r requirements.txt

# 4. Download VGGT (installs package + weights, ~2 GB)
python scripts/download_vggt.py

# 5. Download Grounded-SAM-2 models (~3.5 GB total)
python scripts/download_sam2.py
python scripts/download_grounding_dino.py
#    Also install Grounded-SAM-2 from source:
#    https://github.com/IDEA-Research/Grounded-SAM-2

# 6. Run
python reconstruct.py examples/room_test.mp4 --out outputs/room/
python query.py outputs/room/ "find the chair"
```

On RunPod or any machine with a persistent volume, pass `--dir` to keep weights
out of the repo and on the volume:
```bash
python scripts/download_vggt.py --dir /workspace/checkpoints/vggt
```

## Repository layout

```
scene-query/
├── reconstruct.py           # entry point: video → scene dir
├── query.py                 # entry point: scene dir → answers + Rerun vis
├── src/scene_query/
│   ├── geometry.py          # VGGT wrapper (pose + depth + point cloud)
│   ├── semantics.py         # Grounded-SAM-2 wrapper (per-frame masks)
│   ├── lift.py              # unproject masked depth into world-space labels
│   └── query_engine.py      # text + spatial query logic
├── docs/
│   └── design_note.md       # rationale: choices, tradeoffs, robot relevance
└── examples/
    └── room_test.mp4         # test capture (small indoor room, handheld)
```
