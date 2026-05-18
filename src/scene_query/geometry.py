"""VGGT wrapper: video frames → camera poses, depth maps, point cloud.

Supports both VGGT-1B (original) and VGGT-Omega (CVPR 2026).
Auto-detection: if the weights directory contains a .pt/.pth file, Omega is used;
otherwise VGGT-1B from_pretrained is assumed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def _detect_omega(weights_dir: Path) -> Path | None:
    """Return path to .pt checkpoint if this looks like a VGGT-Omega weights dir."""
    for ext in ("*.pt", "*.pth"):
        matches = list(weights_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def load_model(weights_dir: str | Path, device: str = "cuda"):
    """Load VGGT (1B or Omega) from a local weights directory.

    If the directory contains a .pt/.pth file, VGGT-Omega is loaded.
    Otherwise falls back to VGGT-1B via from_pretrained.
    """
    weights_dir = Path(weights_dir)
    ckpt = _detect_omega(weights_dir)

    if ckpt is not None:
        from vggt_omega.models import VGGTOmega
        model = VGGTOmega()
        state = torch.load(ckpt, map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"  Loaded VGGT-Omega from {ckpt}")
        return model.to(device).eval()

    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained(str(weights_dir))
    print(f"  Loaded VGGT-1B from {weights_dir}")
    return model.to(device).eval()


def extract_frames(
    video_path: str | Path,
    fps: float = 2.0,
    max_frames: int = 50,
) -> list[np.ndarray]:
    """Sample video uniformly at `fps`, then subsample to at most `max_frames`.

    VGGT quality degrades beyond ~50 frames (transformer attention overhead).
    The cap ensures consistent quality regardless of video length.
    """
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

    # Uniform subsample down to max_frames if needed
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[j] for j in indices]

    return frames


def load_frames_from_dir(images_dir: str | Path) -> list[np.ndarray]:
    """Load all images from a directory, sorted by filename. Returns RGB uint8 (H, W, 3).

    Accepts .png, .jpg, .jpeg.
    """
    images_dir = Path(images_dir)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))
    print(f"  Loaded {len(frames)} images from {images_dir}")
    return frames


def _preprocess_frames(
    frames: list[np.ndarray],
    target: int = 518,
    patch: int = 14,
) -> torch.Tensor:
    """Resize frames to model input size and convert to tensor.

    Returns (N, 3, H, W) float32 in [0, 1].
    """
    to_tensor = TF.ToTensor()
    images = []

    for frame in frames:
        img = Image.fromarray(frame)
        w, h = img.size
        new_w = target
        new_h = round(h * (new_w / w) / patch) * patch
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(img)                                   # (3, H, W) [0,1]
        if new_h > target:
            y0 = (new_h - target) // 2
            t = t[:, y0: y0 + target, :]
        images.append(t)

    return torch.stack(images)                               # (N, 3, H, W)


def _depth_to_pointcloud(
    depth: np.ndarray,        # (N, H, W)
    depth_conf: np.ndarray,   # (N, H, W)
    intrinsics: np.ndarray,   # (N, 3, 3)
    poses: np.ndarray,        # (N, 4, 4) cam-to-world
    rgb_tensor: np.ndarray,   # (N, 3, H, W) [0,1]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject depth maps to a world-space point cloud.

    Returns xyz (M, 3), rgb (M, 3) [0,1], xyz_conf (M,).
    """
    N, H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")

    all_xyz, all_rgb, all_conf = [], [], []

    for i in range(N):
        d = depth[i]          # (H, W)
        K = intrinsics[i]
        P = poses[i]          # (4, 4)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        x_cam = (xs - cx) * d / fx
        y_cam = (ys - cy) * d / fy
        z_cam = d

        pts = np.stack([x_cam, y_cam, z_cam,
                        np.ones((H, W), dtype=np.float32)], axis=-1)  # (H, W, 4)
        pts_w = (P @ pts.reshape(-1, 4).T).T[:, :3].astype(np.float32)

        rgb_frame = rgb_tensor[i].transpose(1, 2, 0).reshape(-1, 3)   # (H*W, 3)
        conf_flat = depth_conf[i].reshape(-1)

        all_xyz.append(pts_w)
        all_rgb.append(rgb_frame)
        all_conf.append(conf_flat)

    return (
        np.concatenate(all_xyz, axis=0),
        np.concatenate(all_rgb, axis=0),
        np.concatenate(all_conf, axis=0).astype(np.float32),
    )


def _run_vggt_1b(frames, model, device):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = _preprocess_frames(frames, target=518, patch=14).to(device)
    _, _, H, W = images.shape

    with torch.no_grad():
        preds = model(images)

    pose_enc = preds["pose_enc"]
    extri, intri = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))
    extri = extri[0].cpu().float().numpy()    # (N, 3, 4) world-to-cam
    intri = intri[0].cpu().float().numpy()    # (N, 3, 3)

    N = len(extri)
    poses = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    R = extri[:, :3, :3]
    t = extri[:, :3, 3]
    R_inv = R.transpose(0, 2, 1)
    t_inv = -np.einsum("nij,nj->ni", R_inv, t)
    poses[:, :3, :3] = R_inv
    poses[:, :3, 3] = t_inv

    depth     = preds["depth"][0, :, :, :, 0].cpu().float().numpy()   # (N, H, W)
    depth_conf = preds["depth_conf"][0].cpu().float().numpy()          # (N, H, W)

    xyz_vol  = preds["world_points"][0].cpu().float().numpy()          # (N, H, W, 3)
    conf_vol = preds["world_points_conf"][0].cpu().float().numpy()     # (N, H, W)
    rgb_vol  = preds["images"][0].cpu().float().numpy()                # (N, 3, H, W)
    rgb_vol  = rgb_vol.transpose(0, 2, 3, 1)

    xyz      = xyz_vol.reshape(-1, 3)
    rgb      = rgb_vol.reshape(-1, 3)
    xyz_conf = conf_vol.reshape(-1)

    return {
        "poses":      poses,
        "extrinsics": extri,
        "intrinsics": intri,
        "depth":      depth,
        "depth_conf": depth_conf,
        "xyz":        xyz,
        "rgb":        rgb,
        "xyz_conf":   xyz_conf,
        "image_size": (H, W),
    }


def _run_vggt_omega(frames, model, device):
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = _preprocess_frames(frames, target=512, patch=14).to(device)
    _, _, H, W = images.shape

    with torch.inference_mode():
        preds = model(images)

    extrinsics_t, intrinsics_t = encoding_to_camera(
        preds["pose_enc"], preds["images"].shape[-2:]
    )
    # encoding_to_camera returns (N, 3, 4) extrinsic (world-to-cam) and (N, 3, 3) intrinsic
    extri = extrinsics_t.cpu().float().numpy()   # (N, 3, 4)
    intri = intrinsics_t.cpu().float().numpy()   # (N, 3, 3)

    N = len(extri)
    poses = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    R = extri[:, :3, :3]
    t = extri[:, :3, 3]
    R_inv = R.transpose(0, 2, 1)
    t_inv = -np.einsum("nij,nj->ni", R_inv, t)
    poses[:, :3, :3] = R_inv
    poses[:, :3, 3] = t_inv

    depth      = preds["depth"].cpu().float().numpy()       # (N, H, W) or (N, H, W, 1)
    depth_conf = preds["depth_conf"].cpu().float().numpy()  # (N, H, W) or (N, H, W, 1)
    if depth.ndim == 4:
        depth      = depth[..., 0]
        depth_conf = depth_conf[..., 0]

    rgb_np = preds["images"].cpu().float().numpy()          # (N, 3, H, W)

    xyz, rgb, xyz_conf = _depth_to_pointcloud(depth, depth_conf, intri, poses, rgb_np)

    return {
        "poses":      poses,
        "extrinsics": extri,
        "intrinsics": intri,
        "depth":      depth,
        "depth_conf": depth_conf,
        "xyz":        xyz,
        "rgb":        rgb,
        "xyz_conf":   xyz_conf,
        "image_size": (H, W),
    }


def run_vggt(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
) -> dict:
    """Single forward pass through VGGT (1B or Omega).

    Parameters
    ----------
    frames : list of (H, W, 3) uint8 RGB

    Returns
    -------
    poses      : (N, 4, 4) float32  camera-to-world transforms
    extrinsics : (N, 3, 4) float32  world-to-camera [R|t]
    intrinsics : (N, 3, 3) float32  per-frame camera intrinsics (pixels)
    depth      : (N, H, W) float32  metric depth in metres
    depth_conf : (N, H, W) float32  depth confidence
    xyz        : (M, 3)   float32   world-space point cloud
    rgb        : (M, 3)   float32   per-point colour [0, 1]
    xyz_conf   : (M,)     float32   per-point confidence
    image_size : (H, W)   int       spatial size of processed frames
    """
    try:
        from vggt_omega.models import VGGTOmega
        if isinstance(model, VGGTOmega):
            return _run_vggt_omega(frames, model, device)
    except ImportError:
        pass

    return _run_vggt_1b(frames, model, device)
