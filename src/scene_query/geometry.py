"""VGGT wrapper: video frames → camera poses, depth maps, point cloud."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_model(device: str = "cuda"):
    """Load VGGT from HuggingFace (`facebook/vggt`); cache in checkpoints/vggt/.

    Returns the model in eval mode on `device`.
    """
    ...


def extract_frames(video_path: str | Path, fps: float = 5.0) -> list[np.ndarray]:
    """Sample video uniformly at `fps` and return frames as RGB uint8 (H, W, 3).

    Lower fps reduces VGGT memory footprint; 5 fps is enough for a slow pan.
    """
    ...


def run_vggt(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
) -> dict:
    """Single forward pass through VGGT — no per-scene optimisation.

    Parameters
    ----------
    frames : list of (H, W, 3) uint8 RGB — all at the same resolution

    Returns
    -------
    poses      : (N, 4, 4) float32  camera-to-world transforms
    depth      : (N, H, W) float32  metric depth in metres
    intrinsics : (N, 3, 3) float32  per-frame camera intrinsics
    xyz        : (M, 3)   float32  world-space point cloud
    rgb        : (M, 3)   float32  per-point colour [0, 1]
    """
    ...
