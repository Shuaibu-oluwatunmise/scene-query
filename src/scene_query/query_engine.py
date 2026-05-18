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

    # Full VGGT scene cloud for background visualisation (independent of labels)
    scene_xyz, scene_rgb = None, None
    scene_cloud_path = scene_dir / "scene_cloud.npz"
    if scene_cloud_path.exists():
        sc = np.load(scene_cloud_path)
        scene_xyz, scene_rgb = sc["xyz"], sc["rgb"]

    return {
        "xyz":         pc["xyz"],
        "rgb":         pc["rgb"],
        "label":       pc["label"],
        "confidence":  pc["confidence"],
        "poses":       poses,
        "label_index": label_index,
        "scene_xyz":   scene_xyz,
        "scene_rgb":   scene_rgb,
    }


def _clean_and_fit_obb(pts: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Remove outliers and disconnected blobs from a label point cluster.

    1. Statistical outlier removal (SOR) — kills mask-edge bleed and depth speckle.
    2. Keep only the largest cluster by Euclidean distance — drops reflections /
       occluded fragments that survive SOR but sit far from the main object.
    """
    from scipy.spatial import cKDTree

    if len(pts) == 0:
        return pts

    # SOR: remove points whose mean k-NN distance exceeds mean + 2 sigma
    k = min(20, len(pts) - 1)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    keep = mean_d < (mean_d.mean() + 2.0 * mean_d.std())
    pts = pts[keep]
    if len(pts) == 0:
        return pts

    # Largest cluster: BFS/union-find via radius neighbours
    # radius = 3× median NN distance — connects the object, splits far fragments
    radius = float(np.median(mean_d[keep])) * 3.0
    tree2  = cKDTree(pts)
    pairs  = tree2.query_pairs(radius)

    # Union-find
    parent = list(range(len(pts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    labels = np.array([find(i) for i in range(len(pts))])
    counts = np.bincount(labels)
    biggest = counts.argmax()
    return pts[labels == biggest]


def _gravity_aligned_obb(pts: np.ndarray, poses: np.ndarray) -> dict:
    """Fit a gravity-aligned oriented bounding box.

    Projects the cleaned point cluster onto the horizontal plane (up derived
    from camera poses), fits cv2.minAreaRect to the 2D footprint, then
    extrudes vertically — giving a flat, yaw-only rotated box.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    up = (-poses[:, :3, 1]).mean(axis=0)   # avg camera up (negate OpenCV Y-down)
    up = (up / np.linalg.norm(up)).astype(np.float64)

    ref    = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    horiz1 = np.cross(ref, up);  horiz1 /= np.linalg.norm(horiz1)
    horiz2 = np.cross(up, horiz1)

    pts_h1 = (pts @ horiz1).astype(np.float32)
    pts_h2 = (pts @ horiz2).astype(np.float32)
    pts_2d = np.stack([pts_h1, pts_h2], axis=1)

    rect = cv2.minAreaRect(pts_2d)
    (cx2, cy2), (rw, rh), angle_deg = rect
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    obb_h1 = cos_a * horiz1 + sin_a * horiz2
    obb_h2 = -sin_a * horiz1 + cos_a * horiz2

    pts_up = pts @ up
    v_mid  = (pts_up.min() + pts_up.max()) / 2.0
    v_half = (pts_up.max() - pts_up.min()) / 2.0

    center = (cx2 * horiz1 + cy2 * horiz2 + v_mid * up).astype(np.float32)
    half   = np.array([rw / 2.0, rh / 2.0, v_half], dtype=np.float32)

    R = np.stack([obb_h1, obb_h2, up], axis=1).astype(np.float64)
    if np.linalg.det(R) < 0:
        R[:, 1] = -R[:, 1]
    quat = Rotation.from_matrix(R).as_quat().astype(np.float32)

    return {"center": center, "half": half, "quat": quat}


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

    # Scene was already globally cleaned at reconstruction time; skip per-label SOR.
    center = pts.mean(axis=0).astype(np.float32)
    obb    = _gravity_aligned_obb(pts, scene["poses"])

    return {
        "label":        matched,
        "centroid":     center,
        "bbox_min":     pts.min(axis=0).astype(np.float32),
        "bbox_max":     pts.max(axis=0).astype(np.float32),
        "obb_center":   obb["center"],
        "obb_half":     obb["half"],
        "obb_quat":     obb["quat"],
        "n_points":     len(idxs),
        "confidence":   float(scene["confidence"][idxs].mean()),
        "indices":      idxs,
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
