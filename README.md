# scene-query

**Images or video in. Queryable 3D scene out.**

Most reconstruction pipelines produce a point cloud or a Gaussian splat — a rendering artefact. This project produces something a humanoid robot can actually use: a *scene memory* where objects have labels, space has structure, and you can ask questions.

```bash
# Reconstruct from your own video — label whatever is in your scene
python reconstruct.py --video myvideo.mp4 --out outputs/myscene/ \
    --labels "table,chair,monitor,keyboard"

# Query an object and save a 5-panel Rerun recording
python query.py outputs/myscene/ "find the table" --save-rrd outputs/table.rrd

# View it
python -m rerun outputs/table.rrd
```

Or use the included example images to try it immediately:

```bash
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/
python query.py outputs/tabletop/ "find the table" --save-rrd outputs/table.rrd
python -m rerun outputs/table.rrd
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
├── intrinsics.npz     # camera intrinsics (per frame) + VGGT image_size
├── depth/             # per-frame metric depth maps (.npy)
├── labels.json        # label -> point-index list, for fast lookup
└── frames/            # extracted frames (only when input was --video)
```

## Usage

```bash
# Reconstruct from a directory of images
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/

# Reconstruct from a video file
python reconstruct.py --video scene.mp4 --out outputs/tabletop/ [--fps 5]

# Specify object labels (default: chair,table,sofa,door,window,bed,desk)
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/ --labels "chair,table,door"

# Also save a Rerun recording of the full reconstruction pipeline
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/ --save-rrd outputs/tabletop.rrd

# Query by object name
python query.py outputs/tabletop/ "find the chair"

# Query for navigable free space
python query.py outputs/tabletop/ --free-space [--floor-z <metres>]

# Query for objects reachable from a robot base position
python query.py outputs/tabletop/ --reachable-from <x> <y> <z> [--reach 0.7]

# Save a focused 5-panel Rerun recording for an object query
# (if reconstructed from --images, pass --images here too)
python query.py outputs/tabletop/ "find the chair" \
    --save-rrd outputs/chair_query.rrd \
    --images examples/tabletop/

# If you reconstructed from --video, frames are saved automatically to
# outputs/tabletop/frames/ and query.py picks them up with no extra flags:
python query.py outputs/tabletop/ "find the chair" --save-rrd outputs/chair_query.rrd
```

## Rerun Visualisation

Two Rerun recordings can be generated:

**Reconstruction recording** (`reconstruct.py --save-rrd`): shows the full pipeline — original frames, semantic segmentation overlay, camera trajectory, and the labelled 3D point cloud with per-object bounding boxes. Useful for inspecting reconstruction quality and segmentation accuracy.

**Focused query recording** (`query.py --save-rrd --images`): a 5-panel layout built around the queried object.

```
┌──────────────────┬──────────────────┬──────────────────┐
│  Camera feed     │  3D (cam-locked) │  Segmentation    │
│  (timeline)      │  (timeline)      │  overlay         │
│                  │                  │  (timeline)      │
├──────────────────┴──────────────────┴──────────────────┤
│  row_shares=[3, 2]                                      │
├────────────────────────┬────────────────────────────────┤
│  Grey scene +          │  Photo-coloured scene +        │
│  queried label         │  tight bounding box            │
│  highlighted (static)  │  (static)                      │
└────────────────────────┴────────────────────────────────┘
```

- **Panel 1** — Original RGB frames on a scrubable timeline
- **Panel 2** — 3D point cloud viewed *through* the moving camera (tracking_entity lock)
- **Panel 3** — Back-projected segmentation: all 3D label points projected into each frame, density-smoothed and thresholded to produce a filled 2D mask overlay
- **Panel 4** — Static grey scene with only the queried label's points coloured; outlier points outside the bounding box are suppressed
- **Panel 5** — Static photo-coloured scene with the percentile-trimmed bounding box overlaid

Both panels 4 and 5 start from the first camera frame's point of view (free-orbit from there).

To open a recording in the Rerun viewer:

```bash
# Local viewer
py -3.12 -m rerun outputs/chair_query.rrd

# Or load from a remote machine after SCP
scp root@<pod-ip>:<pod-path>/chair_query.rrd .
py -3.12 -m rerun chair_query.rrd
```

## Getting started

**Requirements:** Python 3.10+, CUDA GPU with 16 GB+ VRAM (tested: RTX 6000 Ada 48 GB, CUDA 12.4)

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

# 5. Download Grounded-SAM-2 models and install from source (~3.5 GB)
python scripts/download_sam2.py
python scripts/download_grounding_dino.py

#    Clone Grounded-SAM-2 and install both sub-packages:
git clone --depth=1 https://github.com/IDEA-Research/Grounded-SAM-2.git /tmp/gsam2
pip install -e /tmp/gsam2
pip install --no-build-isolation -e /tmp/gsam2/grounding_dino

#    Make grounded_sam2 importable as a top-level package:
python -c "
import site, pathlib
pth = pathlib.Path(site.getsitepackages()[0]) / 'grounded_sam2.pth'
pth.write_text('/tmp/gsam2\n')
print('Written:', pth)
"

# 6. Run on the included tabletop images
python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/
python query.py outputs/tabletop/ "find the chair"
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
    └── tabletop/               # 25 images of a tabletop scene (00.png–24.png)
```
