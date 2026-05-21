"""2D mask + depth + pose -> world-space point cloud."""
from __future__ import annotations

import cv2
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
    mask       : (H, W) bool -- if given, unproject only masked pixels

    Returns
    -------
    (N, 3) float32 camera-space XYZ
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    if mask is not None:
        uu, vv, d = uu[mask], vv[mask], depth[mask]
    else:
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        d = depth.reshape(-1)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy
    z = d

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def lift_masks(
    frames: list[np.ndarray],
    depth_maps: np.ndarray,
    intrinsics: np.ndarray,
    poses: np.ndarray,
    masks_per_frame: list[list[dict]],
) -> dict:
    """Unproject masked depth into world space and fuse labels across views.

    Parameters
    ----------
    frames          : list of N (H, W, 3) uint8 RGB
    depth_maps      : (N, Hd, Wd) float32 metric depth at VGGT resolution
    intrinsics      : (N, 3, 3) or (3, 3) float32
    poses           : (N, 4, 4) float32 camera-to-world
    masks_per_frame : output of segment_frames -- N lists of detection dicts

    Returns
    -------
    dict with keys: xyz (M,3), rgb (M,3), label (M,), confidence (M,)
    """
    all_xyz:  list[np.ndarray] = []
    all_rgb:  list[np.ndarray] = []
    all_lbl:  list[str]        = []
    all_conf: list[float]      = []

    N = len(frames)
    for i in range(N):
        frame  = frames[i]           # (Ho, Wo, 3) uint8
        depth  = depth_maps[i]       # (Hd, Wd) float32
        K      = intrinsics[i] if intrinsics.ndim == 3 else intrinsics
        pose   = poses[i]            # (4, 4) cam-to-world

        Ho, Wo = frame.shape[:2]
        Hd, Wd = depth.shape

        for det in masks_per_frame[i]:
            mask_orig: np.ndarray = det["mask"]   # (Ho, Wo) bool
            label:     str        = det["label"]
            conf:      float      = det["confidence"]

            # Resize mask to depth resolution
            mask_d = cv2.resize(
                mask_orig.astype(np.uint8), (Wd, Hd),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            # Only unproject pixels with valid, finite depth
            valid = mask_d & (depth > 1e-2) & np.isfinite(depth)
            if not valid.any():
                continue

            # Unproject -> camera space
            pts_cam = unproject(depth, K, mask=valid)   # (M, 3)

            # Camera space -> world space via 4x4 pose
            pts_h = np.c_[pts_cam, np.ones(len(pts_cam), dtype=np.float32)]
            pts_world = (pose @ pts_h.T).T[:, :3]

            # Sample RGB from original frame at corresponding pixels
            v_d, u_d = np.where(valid)
            u_o = (u_d * Wo / Wd).astype(np.int32).clip(0, Wo - 1)
            v_o = (v_d * Ho / Hd).astype(np.int32).clip(0, Ho - 1)
            rgb = frame[v_o, u_o].astype(np.float32) / 255.0

            all_xyz.append(pts_world)
            all_rgb.append(rgb)
            all_lbl.extend([label] * len(pts_world))
            all_conf.extend([conf] * len(pts_world))

    if not all_xyz:
        return {
            "xyz":        np.zeros((0, 3), dtype=np.float32),
            "rgb":        np.zeros((0, 3), dtype=np.float32),
            "label":      np.array([], dtype=object),
            "confidence": np.zeros(0, dtype=np.float32),
        }

    return {
        "xyz":        np.concatenate(all_xyz,  axis=0),
        "rgb":        np.concatenate(all_rgb,  axis=0),
        "label":      np.array(all_lbl,        dtype=object),
        "confidence": np.array(all_conf,       dtype=np.float32),
    }


