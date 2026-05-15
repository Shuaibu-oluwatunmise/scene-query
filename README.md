# scene-query

**Images or video in. Queryable 3D scene out.**

Most reconstruction pipelines produce a point cloud or a Gaussian splat — a rendering artefact. This project produces something a humanoid robot can actually use: a *scene memory* where objects have labels, space has structure, and you can ask questions.

```bash
# Run on the included example images
python reconstruct.py --images examples/room/ --out outputs/room/

# Or on your own video
python reconstruct.py --video path/to/video.mp4 --out outputs/room/

# Query the scene
python query.py outputs/room/ "find the chair"
python query.py outputs/room/ --free-space --floor-z 0.0
python query.py outputs/room/ --reachable-from 0.0 0.0 0.9
```

Built for Humanoid's Perception & Spatial AI internship challenge. See [docs/design_note.md](docs/design_note.md) for the full rationale.

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
  "find the chair" → 3D bounding region + centroid
```

## Output format

```
outputs/room/
├── pointcloud.npz     # XYZ, RGB, label (str), confidence (float), per point
├── poses.npz          # camera-to-world 4x4 matrices, one per frame
├── depth/             # per-frame metric depth maps (.npy)
└── labels.json        # label -> point-index list, for fast lookup
```

## Usage

```bash
# Reconstruct from a directory of images
python reconstruct.py --images examples/room/ --out outputs/room/

# Reconstruct from a video file
python reconstruct.py --video room.mp4 --out outputs/room/ [--fps 5]

# Specify object labels (default: chair,table,sofa,door,window,bed,desk)
python reconstruct.py --images examples/room/ --out outputs/room/ --labels "chair,table,door"

# Query by object name
python query.py outputs/room/ "find the chair"

# Query for navigable free space
python query.py outputs/room/ --free-space [--floor-z <metres>]

# Query for objects reachable from a robot base position
python query.py outputs/room/ --reachable-from <x> <y> <z> [--reach 0.7]
```

## Getting started

**Requirements:** Python 3.10+, CUDA-capable GPU (tested: RTX 6000 Ada, CUDA 12.4)

```bash
# 1. Clone — example images are included, no separate download needed
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query

# 2. Install PyTorch with the right CUDA version for your machine
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

# 6. Run on the included room images
python reconstruct.py --images examples/room/ --out outputs/room/
python query.py outputs/room/ "find the chair"
```

**On RunPod or any machine with a persistent volume**, pass `--dir` to keep
weights on the volume so they survive pod restarts:
```bash
python scripts/download_vggt.py --dir /workspace/checkpoints/vggt
python scripts/download_sam2.py --dir /workspace/checkpoints/sam2
python scripts/download_grounding_dino.py --dir /workspace/checkpoints/grounding_dino
```

After a pod migration, restore the environment in one command:
```bash
bash scripts/setup_pod.sh
```

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
    └── room/                   # 8 room images for development and demo
        ├── no_overlap_1.png
        └── no_overlap_2-8.jpg
```
