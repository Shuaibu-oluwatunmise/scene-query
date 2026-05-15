"""
Smoke-test VGGT end to end.

Downloads VGGT's own example images, runs a forward pass, and saves the
output as a .ply file you can scp to your laptop and open in MeshLab.

Usage:
    python scripts/test_vggt.py
    python scripts/test_vggt.py --out outputs/vggt_test.ply
    python scripts/test_vggt.py --weights /workspace/checkpoints/vggt

What you get:
    A .ply point cloud file. Open it in MeshLab (free download) to see
    whether VGGT produced sensible geometry from the example images.

If this works, VGGT is installed correctly and you can move on to your
own video via reconstruct.py.
"""

import argparse
import struct
import sys
from pathlib import Path
import urllib.request

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "checkpoints" / "vggt"
DEFAULT_OUT = REPO_ROOT / "outputs" / "vggt_test.ply"

# VGGT's own example images from their GitHub repo
# raw.githubusercontent.com serves the actual file content (needed for download)
EXAMPLE_IMAGE_URLS = [
    f"https://raw.githubusercontent.com/facebookresearch/vggt/main/examples/kitchen/images/{i:02d}.png"
    for i in range(0, 25)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .ply path")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="VGGT weights directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def download_examples(tmp_dir: Path) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in EXAMPLE_IMAGE_URLS:
        dest = tmp_dir / Path(url).name
        if not dest.exists():
            print(f"  Downloading {Path(url).name}...")
            urllib.request.urlretrieve(url, dest)
        paths.append(dest)
    return paths


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path) -> None:
    """Write a point cloud to a binary PLY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    # Build as a structured numpy array and write in one shot — avoids per-row Python loops
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb_uint8[:, 0], rgb_uint8[:, 1], rgb_uint8[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def main() -> None:
    args = parse_args()
    print(f"Device : {args.device}")
    print(f"Weights: {args.weights}")
    print(f"Output : {args.out}\n")

    # --- Load model ---
    print("Loading VGGT model...")
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError:
        print("ERROR: vggt not installed. Run: python scripts/download_vggt.py")
        sys.exit(1)

    model = VGGT.from_pretrained(str(args.weights))
    model = model.to(args.device).eval()
    print("Model loaded.\n")

    # --- Get example images ---
    print("Fetching VGGT example images...")
    tmp_dir = REPO_ROOT / "outputs" / "_vggt_examples"
    image_paths = download_examples(tmp_dir)
    print(f"  {len(image_paths)} images ready.\n")

    # --- Run inference ---
    print("Running VGGT forward pass (this may take 30-60 s the first time)...")
    images = load_and_preprocess_images([str(p) for p in image_paths])
    images = images.to(args.device)

    with torch.no_grad():
        predictions = model(images)

    print("Inference done.\n")

    # --- Extract point cloud ---
    # world_points: [B, S, H, W, 3] — one 3D point per pixel per frame
    # world_points_conf: [B, S, H, W] — confidence per point
    # images: [B, S, 3, H, W] — original frames, used for colour
    print("Predictions keys:", list(predictions.keys()))
    for k, v in predictions.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")

    # XYZ: drop batch dim, flatten (S, H, W, 3) → (S*H*W, 3)
    xyz = predictions["world_points"][0].cpu().float().numpy()   # (S, H, W, 3)
    conf = predictions["world_points_conf"][0].cpu().float().numpy()  # (S, H, W)

    # Colour: images are (B, S, 3, H, W) → (S, H, W, 3)
    imgs = predictions["images"][0].cpu().float().numpy()        # (S, 3, H, W)
    imgs = imgs.transpose(0, 2, 3, 1)                           # (S, H, W, 3)

    # Flatten everything
    xyz = xyz.reshape(-1, 3)
    rgb = imgs.reshape(-1, 3)
    conf = conf.reshape(-1)

    # Step 1: confidence filter — drop bottom 25%
    threshold = np.percentile(conf, 25)
    mask = conf > threshold
    xyz, rgb = xyz[mask], rgb[mask]
    print(f"  After confidence filter : {len(xyz):,} points")

    # Step 2: statistical outlier removal — drop points whose neighbours are far away
    # For each point, compute mean distance to its k nearest neighbours.
    # Points with mean distance > (global_mean + 2*std) are outliers.
    from scipy.spatial import cKDTree
    print("  Running outlier removal...")
    k = 16
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)   # k+1 because first result is the point itself
    mean_dists = dists[:, 1:].mean(axis=1)
    dist_threshold = mean_dists.mean() + 2.0 * mean_dists.std()
    mask = mean_dists < dist_threshold
    xyz, rgb = xyz[mask], rgb[mask]

    print(f"  After outlier removal   : {len(xyz):,} points")

    # --- Save .ply ---
    save_ply(xyz, rgb, args.out)
    print(f"\nSaved → {args.out}")
    print("\nTo view on your laptop:")
    print(f"  scp <pod-user>@<pod-ip>:<pod-path>/{args.out.name} .")
    print("  Open in MeshLab: File → Import Mesh")


if __name__ == "__main__":
    main()
