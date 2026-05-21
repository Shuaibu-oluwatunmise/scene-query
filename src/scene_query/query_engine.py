"""Text and spatial queries over a labelled point cloud scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_scene(scene_dir: str | Path) -> dict:
    """Load geometry from a scene directory produced by reconstruct.py.

    Returns a dict with keys: poses, scene_xyz, scene_rgb.
    """
    scene_dir = Path(scene_dir)

    poses = None
    poses_path = scene_dir / "poses.npz"
    if poses_path.exists():
        poses = np.load(poses_path)["poses"]

    scene_xyz, scene_rgb = None, None
    scene_cloud_path = scene_dir / "scene_cloud.npz"
    if scene_cloud_path.exists():
        sc = np.load(scene_cloud_path)
        scene_xyz, scene_rgb = sc["xyz"], sc["rgb"]

    return {
        "poses":     poses,
        "scene_xyz": scene_xyz,
        "scene_rgb": scene_rgb,
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


