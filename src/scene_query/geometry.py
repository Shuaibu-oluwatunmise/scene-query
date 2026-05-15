"""VGGT wrapper: video frames → camera poses, depth maps, point cloud."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def load_model(weights_dir: str | Path, device: str = "cuda"):
    """Load VGGT-1B from a local weights directory.

    Download weights first with: python scripts/download_vggt.py
    """
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained(str(weights_dir))
    return model.to(device).eval()


def extract_frames(video_path: str | Path, fps: float = 5.0) -> list[np.ndarray]:
    """Sample video uniformly at `fps` and return frames as RGB uint8 (H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / fps))

    frames, i = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def _preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Resize frames to VGGT's expected input size and convert to tensor.

    Replicates load_and_preprocess_images (crop mode) for numpy arrays so we
    don't have to write frames to disk.
    Returns (N, 3, H, W) float32 in [0, 1].
    """
    to_tensor = TF.ToTensor()
    target = 518
    images = []

    for frame in frames:
        img = Image.fromarray(frame)
        w, h = img.size
        new_w = target
        new_h = round(h * (new_w / w) / 14) * 14          # divisible by 14
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(img)                                   # (3, H, W) [0,1]
        if new_h > target:                                   # centre-crop height
            y0 = (new_h - target) // 2
            t = t[:, y0: y0 + target, :]
        images.append(t)

    return torch.stack(images)                               # (N, 3, H, W)


def run_vggt(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
) -> dict:
    """Single forward pass through VGGT.

    Parameters
    ----------
    frames : list of (H, W, 3) uint8 RGB — from extract_frames

    Returns
    -------
    poses      : (N, 4, 4) float32  camera-to-world transforms
    extrinsics : (N, 3, 4) float32  world-to-camera [R|t] (OpenCV convention)
    intrinsics : (N, 3, 3) float32  per-frame camera intrinsics (pixels)
    depth      : (N, H, W) float32  metric depth in metres
    depth_conf : (N, H, W) float32  depth confidence [0, ∞)
    xyz        : (M, 3)   float32  world-space point cloud
    rgb        : (M, 3)   float32  per-point colour [0, 1]
    xyz_conf   : (M,)     float32  per-point confidence
    image_size : (H, W)   int      spatial size of the processed frames
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = _preprocess_frames(frames).to(device)           # (N, 3, H, W)
    _, _, H, W = images.shape

    with torch.no_grad():
        preds = model(images)

    # --- Poses ---
    # pose_enc: (1, N, 9) = [T(3), quaternion(4), fov_h, fov_w]
    pose_enc = preds["pose_enc"]                             # (1, N, 9)
    extri, intri = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W)
    )
    # extri: (1, N, 3, 4) world-to-camera; intri: (1, N, 3, 3)
    extri = extri[0].cpu().float().numpy()                   # (N, 3, 4)
    intri = intri[0].cpu().float().numpy()                   # (N, 3, 3)

    # Convert world-to-camera → camera-to-world 4×4
    N = len(extri)
    poses = np.eye(4, dtype=np.float32)[None].repeat(N, 0)  # (N, 4, 4)
    R = extri[:, :3, :3]                                     # (N, 3, 3)
    t = extri[:, :3, 3]                                      # (N, 3)
    R_inv = R.transpose(0, 2, 1)                             # (N, 3, 3)
    t_inv = -np.einsum("nij,nj->ni", R_inv, t)              # (N, 3)
    poses[:, :3, :3] = R_inv
    poses[:, :3, 3] = t_inv

    # --- Depth ---
    depth = preds["depth"][0, :, :, :, 0].cpu().float().numpy()       # (N, H, W)
    depth_conf = preds["depth_conf"][0].cpu().float().numpy()          # (N, H, W)

    # --- Point cloud ---
    xyz_vol = preds["world_points"][0].cpu().float().numpy()           # (N, H, W, 3)
    conf_vol = preds["world_points_conf"][0].cpu().float().numpy()     # (N, H, W)
    rgb_vol = preds["images"][0].cpu().float().numpy()                 # (N, 3, H, W)
    rgb_vol = rgb_vol.transpose(0, 2, 3, 1)                           # (N, H, W, 3)

    xyz = xyz_vol.reshape(-1, 3)
    rgb = rgb_vol.reshape(-1, 3)
    xyz_conf = conf_vol.reshape(-1)

    return {
        "poses":      poses,        # (N, 4, 4)
        "extrinsics": extri,        # (N, 3, 4) raw world-to-cam, kept for reference
        "intrinsics": intri,        # (N, 3, 3)
        "depth":      depth,        # (N, H, W)
        "depth_conf": depth_conf,   # (N, H, W)
        "xyz":        xyz,          # (M, 3)
        "rgb":        rgb,          # (M, 3)
        "xyz_conf":   xyz_conf,     # (M,)
        "image_size": (H, W),
    }
