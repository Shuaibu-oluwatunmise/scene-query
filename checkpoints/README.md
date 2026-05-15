# Model checkpoints

Downloaded at install time, not committed. Place weights in the subdirectories below.

| Directory          | Model            | Size  | Source |
|--------------------|------------------|-------|--------|
| `vggt/`            | VGGT             | ~2 GB | HuggingFace `facebook/vggt` |
| `sam2/`            | SAM 2.1          | ~2.5 GB | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `grounding_dino/`  | Grounding DINO   | ~1 GB | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |

Download instructions are in each wrapper module (`src/scene_query/geometry.py`, `src/scene_query/semantics.py`).
