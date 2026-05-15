"""Text and spatial queries over a labelled point cloud scene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_scene(scene_dir: str | Path) -> dict:
    """Load pointcloud.npz + labels.json from a scene directory.

    Returns a dict with keys: xyz, rgb, label, confidence, poses, label_index.
    label_index maps each label string to its point indices for O(1) lookup.
    """
    scene_dir = Path(scene_dir)

    pc = np.load(scene_dir / "pointcloud.npz", allow_pickle=True)

    poses = None
    poses_path = scene_dir / "poses.npz"
    if poses_path.exists():
        poses = np.load(poses_path)["poses"]

    with open(scene_dir / "labels.json") as f:
        label_index: dict[str, list[int]] = json.load(f)

    return {
        "xyz":         pc["xyz"],
        "rgb":         pc["rgb"],
        "label":       pc["label"],
        "confidence":  pc["confidence"],
        "poses":       poses,
        "label_index": label_index,
    }


def query_object(scene: dict, label: str) -> dict:
    """Find a named object and return its 3D extent.

    Parameters
    ----------
    label : object name, e.g. "chair" -- matched against scene labels

    Returns
    -------
    label, centroid (3,), bbox_min (3,), bbox_max (3,), n_points, confidence, indices
    """
    label_index = scene["label_index"]
    label = label.lower().strip()

    # Exact match first, then substring
    if label in label_index:
        matched = label
    else:
        candidates = [k for k in label_index if label in k or k in label]
        if not candidates:
            available = list(label_index.keys())
            raise KeyError(
                f"Label '{label}' not found. Available labels: {available}"
            )
        # Pick the candidate with most points
        matched = max(candidates, key=lambda k: len(label_index[k]))

    idxs = np.array(label_index[matched], dtype=np.int64)
    pts  = scene["xyz"][idxs]

    return {
        "label":      matched,
        "centroid":   pts.mean(axis=0).astype(np.float32),
        "bbox_min":   pts.min(axis=0).astype(np.float32),
        "bbox_max":   pts.max(axis=0).astype(np.float32),
        "n_points":   len(idxs),
        "confidence": float(scene["confidence"][idxs].mean()),
        "indices":    idxs,
    }


def query_free_space(
    scene: dict,
    floor_z: float,
    voxel_size: float = 0.05,
    clearance: float = 1.8,
) -> np.ndarray:
    """Return navigable floor voxel centres as (K, 3) float32.

    A voxel column is traversable if it has points at floor level
    and no obstacle points in the clearance band above.
    """
    xyz = scene["xyz"]

    # Restrict to the relevant height band
    band = (xyz[:, 2] >= floor_z) & (xyz[:, 2] <= floor_z + clearance)
    pts = xyz[band]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xy_min = pts[:, :2].min(axis=0)
    vij = np.floor((pts[:, :2] - xy_min) / voxel_size).astype(np.int32)

    z_rel = pts[:, 2] - floor_z
    floor_layer = z_rel < 0.1   # within 10 cm of floor
    obs_layer   = z_rel >= 0.1  # obstacle height

    floor_voxels = set(map(tuple, vij[floor_layer].tolist()))
    obs_voxels   = set(map(tuple, vij[obs_layer].tolist()))

    free_voxels = floor_voxels - obs_voxels
    if not free_voxels:
        return np.zeros((0, 3), dtype=np.float32)

    result = [
        [xy_min[0] + (i + 0.5) * voxel_size,
         xy_min[1] + (j + 0.5) * voxel_size,
         floor_z]
        for i, j in free_voxels
    ]
    return np.array(result, dtype=np.float32)


def query_reachable(
    scene: dict,
    base_xyz: np.ndarray,
    reach: float = 0.7,
    height_range: tuple[float, float] = (0.3, 1.4),
) -> list[dict]:
    """Return all labelled objects reachable from a robot base position.

    An object is reachable if its centroid is within `reach` metres
    horizontally and within `height_range` vertically above base_xyz.
    """
    results = []
    for label in scene["label_index"]:
        try:
            obj = query_object(scene, label)
        except KeyError:
            continue

        centroid = obj["centroid"]
        horiz    = float(np.linalg.norm(centroid[:2] - base_xyz[:2]))
        height   = float(centroid[2] - base_xyz[2])

        if horiz <= reach and height_range[0] <= height <= height_range[1]:
            results.append(obj)

    return results
