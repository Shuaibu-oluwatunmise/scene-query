"""VGGT-Omega wrapper: video frames → camera poses, depth maps, point cloud."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def load_model(weights_dir: str | Path, device: str = "cuda"):
    """Load VGGT-Omega from a local weights directory (any .pt/.pth file inside)."""
    weights_dir = Path(weights_dir)
    matches = list(weights_dir.glob("*.pt")) + list(weights_dir.glob("*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No .pt/.pth checkpoint found in {weights_dir}\n"
            "Run python setup.py to download VGGT-Omega weights."
        )
    ckpt = matches[0]
    from vggt_omega.models import VGGTOmega
    model = VGGTOmega()
    state = torch.load(ckpt, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"  Loaded VGGT-Omega from {ckpt}")
    return model.to(device).eval()


def extract_frames(
    video_path: str | Path,
    fps: float = 2.0,
    max_frames: int = 50,
) -> list[np.ndarray]:
    """Sample video uniformly at `fps`, then subsample to at most `max_frames`.

    VGGT quality degrades beyond ~50 frames (transformer attention overhead).
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

    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[j] for j in indices]

    return frames


def load_frames_from_dir(images_dir: str | Path) -> list[np.ndarray]:
    """Load all images from a directory, sorted by filename. Returns RGB uint8 (H, W, 3)."""
    images_dir = Path(images_dir)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    print(f"  Loaded {len(frames)} images from {images_dir}")
    return frames


def _preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Resize frames to VGGT-Omega input size (512px wide, patch-14 aligned).

    Returns (N, 3, H, W) float32 in [0, 1].
    """
    target, patch = 512, 14
    to_tensor = TF.ToTensor()
    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        w, h = img.size
        new_w = target
        new_h = round(h * (new_w / w) / patch) * patch
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(img)
        if new_h > target:
            y0 = (new_h - target) // 2
            t = t[:, y0: y0 + target, :]
        images.append(t)
    return torch.stack(images)


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
        d = depth[i]
        K = intrinsics[i]
        P = poses[i]

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        x_cam = (xs - cx) * d / fx
        y_cam = (ys - cy) * d / fy
        pts = np.stack([x_cam, y_cam, d, np.ones((H, W), dtype=np.float32)], axis=-1)
        pts_w = (P @ pts.reshape(-1, 4).T).T[:, :3].astype(np.float32)

        all_xyz.append(pts_w)
        all_rgb.append(rgb_tensor[i].transpose(1, 2, 0).reshape(-1, 3))
        all_conf.append(depth_conf[i].reshape(-1))

    return (
        np.concatenate(all_xyz),
        np.concatenate(all_rgb),
        np.concatenate(all_conf).astype(np.float32),
    )


def run_vggt(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
) -> dict:
    """Run VGGT-Omega on a list of frames.

    Returns
    -------
    poses      : (N, 4, 4) float32  camera-to-world
    extrinsics : (N, 3, 4) float32  world-to-camera [R|t]
    intrinsics : (N, 3, 3) float32  per-frame intrinsics (pixels)
    depth      : (N, H, W) float32  metric depth in metres
    depth_conf : (N, H, W) float32  depth confidence
    xyz        : (M, 3)   float32   world-space point cloud
    rgb        : (M, 3)   float32   per-point colour [0, 1]
    xyz_conf   : (M,)     float32   per-point confidence
    image_size : (H, W)   int       spatial size of processed frames
    """
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = _preprocess_frames(frames).to(device)
    _, _, H, W = images.shape

    with torch.inference_mode():
        preds = model(images)

    depth_t      = preds["depth"]
    depth_conf_t = preds["depth_conf"]
    # Handle (1, N, H, W, 1), (1, N, H, W) output variants
    if depth_t.dim() == 5:
        depth_t, depth_conf_t = depth_t[0, :, :, :, 0], depth_conf_t[0]
    elif depth_t.dim() == 4 and depth_t.shape[0] == 1:
        depth_t, depth_conf_t = depth_t[0], depth_conf_t[0]
    depth      = depth_t.cpu().float().numpy()
    depth_conf = depth_conf_t.cpu().float().numpy()
    H_d, W_d   = depth.shape[-2:]

    # FOV angles in pose_enc are resolution-invariant; pass depth resolution
    # for correct pixel-space focal lengths during unprojection.
    extrinsics_t, intrinsics_t = encoding_to_camera(preds["pose_enc"], (H_d, W_d))
    extri = extrinsics_t[0].cpu().float().numpy()   # (N, 3, 4)
    intri = intrinsics_t[0].cpu().float().numpy()   # (N, 3, 3)

    N = len(extri)
    poses = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    R, t  = extri[:, :3, :3], extri[:, :3, 3]
    R_inv = R.transpose(0, 2, 1)
    poses[:, :3, :3] = R_inv
    poses[:, :3, 3]  = -np.einsum("nij,nj->ni", R_inv, t)

    rgb_np = preds["images"][0].cpu().float().numpy()   # (N, 3, H_img, W_img)
    if rgb_np.shape[-2:] != (H_d, W_d):
        import torch.nn.functional as F
        rgb_np = F.interpolate(
            torch.from_numpy(rgb_np), size=(H_d, W_d),
            mode="bilinear", align_corners=False,
        ).numpy()

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
