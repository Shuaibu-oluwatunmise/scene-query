"""Text and spatial queries over a labelled point cloud scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_scene(scene_dir: str | Path) -> dict:
    """Load pointcloud.npz + labels.json from a scene directory.

    Returns a dict with keys: xyz, rgb, label, confidence, poses, label_index.
    label_index maps each label string to its point indices for O(1) lookup.
    """
    ...


def query_object(scene: dict, label: str) -> dict:
    """Find a named object and return its 3D extent.

    Parameters
    ----------
    label : object name, e.g. "chair" — matched against scene labels

    Returns
    -------
    label       : str
    centroid    : (3,) float32
    bbox_min    : (3,) float32  axis-aligned bounding box corners
    bbox_max    : (3,) float32
    n_points    : int
    confidence  : float  mean label confidence of matched points
    indices     : (K,) int  point indices into scene arrays
    """
    ...


def query_free_space(
    scene: dict,
    floor_z: float,
    voxel_size: float = 0.05,
    clearance: float = 1.8,
) -> np.ndarray:
    """Return navigable floor voxel centres as (K, 3) float32.

    A voxel column is traversable if it sits above floor_z, is below
    clearance height, and has no occupied voxel directly above it.
    voxel_size controls resolution; 5 cm is suitable for step planning.
    """
    ...


def query_reachable(
    scene: dict,
    base_xyz: np.ndarray,
    reach: float = 0.7,
    height_range: tuple[float, float] = (0.3, 1.4),
) -> list[dict]:
    """Return all labelled objects reachable from a robot base position.

    An object is reachable if its centroid is within `reach` metres
    horizontally and within `height_range` vertically above base_xyz.
    Each entry is the full output of query_object for that label.

    Parameters
    ----------
    base_xyz     : (3,) robot base position in world space
    reach        : horizontal reach radius in metres (default 0.7 m)
    height_range : (min, max) reachable height band above base_xyz[2]
    """
    ...
