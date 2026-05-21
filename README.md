# scene-query

Point a camera at a scene. Ask it where things are.

**scene-query** takes a video or a folder of images, builds a dense 3D reconstruction, and lets you query it in plain English — *"find the bulldozer"*, *"find the chair"*, *"find the monitor"* — returning a tight 3D bounding box around the object, visualised with a full camera timeline.

No fixed vocabulary. No retraining. Any object you can name, it can find.

---

## How it works

**Reconstruct** — feed in your images or video. [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) runs a single feed-forward pass and produces camera poses, per-frame depth maps, and a photo-coloured 3D point cloud. No optimisation loop, no SfM pipeline — just one forward pass.

**Query** — ask for any object by name. [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) detects it across every frame open-vocabulary, [SAM 2](https://github.com/facebookresearch/sam2) segments it precisely, and the masks are back-projected using VGGT depth to lift the object into 3D. The result is a tight oriented bounding box ready for robot manipulation, navigation, or scene understanding.

---

## Scenario 1 — just explore my outputs (no GPU needed)

All you need is [Rerun](https://rerun.io) and Git LFS to pull the output files.

```bash
git lfs install
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query

pip install rerun-sdk
```

**Open the 3D reconstruction:**
```bash
rerun outputs/tabletop/tabletop.rrd
```

**Open the bulldozer query result:**
```bash
rerun outputs/tabletop/query_bulldozer.rrd
```

### What you will see

**Reconstruction** — raw camera feed alongside the live 3D point cloud. Camera frustums move through the scene as you scrub the timeline:

![Tabletop reconstruction](outputs/tabletop/tabletop.png)

**Query: `bulldozer`** — four panels: raw camera feed, 3D reconstruction, 2D detections per frame, and the 3D oriented bounding box localising the object in the scene:

![Bulldozer query](outputs/tabletop/query.png)

The object here is a LEGO Technic bulldozer — chosen deliberately to show that queries can be specific and unconventional. There was no bulldozer class trained anywhere. Grounding DINO found it from the text prompt alone.

---

## Scenario 2 — run it yourself (CUDA GPU required)

### Prerequisites

> **OS note:** `reconstruct.py` and `query.py` require a **Linux environment** (or WSL2 on Windows) due to Grounding DINO's CUDA compilation step. Viewing outputs with Rerun works on Windows, macOS, and Linux.

- NVIDIA GPU (RTX 3080 or better recommended, ≥ 16 GB VRAM)
- CUDA 12.x
- Python 3.10+
- PyTorch with CUDA — install this first:

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Full list: https://pytorch.org/get-started/locally/

### 1. Clone and install

```bash
git lfs install
git clone https://github.com/Shuaibu-oluwatunmise/scene-query.git
cd scene-query

pip install -r requirements.txt
```

### 2. Download model weights

```bash
python setup.py
```

Downloads VGGT-Omega (~4.3 GB), Grounding DINO, and SAM 2.1 weights into `checkpoints/`.

### 3. Reconstruct

**From a folder of images:**
```bash
python reconstruct.py \
    --images examples/tabletop \
    --out outputs/tabletop \
    --save-rrd outputs/tabletop/tabletop.rrd
```

**From a video:**
```bash
python reconstruct.py \
    --video examples/my_scene.mp4 \
    --out outputs/my_scene \
    --save-rrd outputs/my_scene/scene.rrd
```

Open the result:
```bash
rerun outputs/tabletop/tabletop.rrd
```

Outputs saved to the `--out` directory:

| File | Description |
|---|---|
| `scene_cloud.npz` | Photo-coloured 3D point cloud |
| `depth.npz` | Per-frame depth maps |
| `poses.npz` | Camera-to-world poses |
| `intrinsics.npz` | Camera intrinsics |
| `tabletop.rrd` | Rerun recording |

### 4. Query

**Specific object:**
```bash
python query.py outputs/tabletop "find the bulldozer" \
    --images examples/tabletop \
    --save-rrd outputs/tabletop/query_bulldozer.rrd
```

The `"find the"` prefix is optional — `bulldozer` works just as well.

**Multiple objects — comma-separated in plain English:**
```bash
python query.py outputs/tabletop "find the bulldozer, cup, bottle" \
    --images examples/tabletop \
    --save-rrd outputs/tabletop/query_multi.rrd
```

**Full environment scan using a preset:**
```bash
python query.py outputs/my_scene \
    --preset office \
    --images examples/my_scene \
    --save-rrd outputs/my_scene/query_office.rrd
```

Open the result:
```bash
rerun outputs/tabletop/query_bulldozer.rrd
```

Available presets: `office`, `home`, `classroom`, `kitchen`, `warehouse`.

---

## Examples

### Tabletop — image input, specific object query *(included)*

| Input | 25 images of a tabletop scene |
|---|---|
| Query | `bulldozer` |
| Result | Detected in all 25 frames at 0.89 confidence, 522K points lifted to 3D |

| Reconstruction | Query |
|---|---|
| ![Reconstruction](outputs/tabletop/tabletop.png) | ![Query](outputs/tabletop/query.png) |

### Office / home walkthrough — video input, preset query *(coming soon)*

| Input | Video walkthrough of an office space |
|---|---|
| Query | `--preset office` |
| Result | Every common object in the scene found and localised in one pass |

---

## Project structure

```
scene-query/
├── reconstruct.py          # Images / video → 3D geometry
├── query.py                # Text query → 3D bounding box
├── setup.py                # One-shot setup: weights + deps
├── src/scene_query/
│   ├── geometry.py         # VGGT-Omega wrapper
│   ├── lift.py             # Mask back-projection (2D + depth → 3D)
│   ├── semantics.py        # Grounding DINO + SAM 2
│   └── query_engine.py     # Scene loading, OBB fitting
├── scripts/
│   ├── install_deps.py     # Dependency installer
│   └── download_models.py  # Weight downloader
├── examples/
│   └── tabletop/           # 25 input images
└── outputs/
    └── tabletop/           # Reconstruction + query outputs (Git LFS)
```
