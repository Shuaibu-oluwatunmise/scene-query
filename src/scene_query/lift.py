"""2D mask + depth + pose → labelled world-space point cloud."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def unproject(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Unproject a depth map to camera-space 3D points.

    Parameters
    ----------
    depth      : (H, W) float32 metric depth in metres
    intrinsics : (3, 3) float32 camera intrinsic matrix
    mask       : (H, W) bool — if given, unproject only masked pixels

    Returns
    -------
    (N, 3) float32 camera-space XYZ, where N = number of unmasked pixels
    """
    ...


def lift_masks(
    frames: list[np.ndarray],
    depth_maps: np.ndarray,
    intrinsics: np.ndarray,
    poses: np.ndarray,
    masks_per_frame: list[list[dict]],
) -> dict:
    """Unproject masked depth into world space and fuse labels across views.

    For each labelled mask in each frame: unproject the masked pixels using
    depth and intrinsics, then transform to world space using the camera pose.
    Points seen from multiple frames accumulate label votes; confidence is the
    fraction of votes that agree on the winning label.

    Parameters
    ----------
    frames          : list of N (H, W, 3) uint8 RGB
    depth_maps      : (N, H, W) float32 metric depth
    intrinsics      : (3, 3) or (N, 3, 3) float32
    poses           : (N, 4, 4) float32 camera-to-world
    masks_per_frame : output of segment_frames — N lists of detection dicts

    Returns
    -------
    xyz        : (M, 3) float32  world-space positions
    rgb        : (M, 3) float32  colour [0, 1]
    label      : (M,)  object   str label per point
    confidence : (M,)  float32  label vote confidence [0, 1]
    """
    ...


def save_scene(scene: dict, output_dir: str | Path) -> None:
    """Persist a labelled scene to disk.

    Writes:
        pointcloud.npz  — xyz, rgb, label, confidence arrays
        poses.npz       — camera-to-world pose matrices
        labels.json     — {label: [point_indices]} for fast lookup
    """
    ...
