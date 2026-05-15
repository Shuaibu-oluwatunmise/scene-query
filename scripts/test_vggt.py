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
    for i in range(1, 9)
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
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<fffBBB", xyz[i, 0], xyz[i, 1], xyz[i, 2],
                                rgb_uint8[i, 0], rgb_uint8[i, 1], rgb_uint8[i, 2]))


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
    # VGGT returns world-space points and per-point colours
    # Key names may vary — print what we got if something's wrong
    print("Predictions keys:", list(predictions.keys()))

    # Try common key names
    xyz_key = next((k for k in predictions if "point" in k.lower() or "xyz" in k.lower() or "world" in k.lower()), None)

    if xyz_key is None:
        print("\nCould not find point cloud key automatically. Available keys:")
        for k, v in predictions.items():
            shape = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape}")
        print("\nCheck the keys above and re-run with the right key name.")
        sys.exit(0)

    xyz = predictions[xyz_key].squeeze().cpu().numpy()
    if xyz.ndim == 3:
        xyz = xyz.reshape(-1, 3)

    # Try to find colour
    rgb_key = next((k for k in predictions if "color" in k.lower() or "rgb" in k.lower()), None)
    if rgb_key is not None:
        rgb = predictions[rgb_key].squeeze().cpu().numpy()
        if rgb.ndim == 3:
            rgb = rgb.reshape(-1, 3)
    else:
        rgb = np.ones_like(xyz) * 0.7  # grey fallback

    print(f"Point cloud: {len(xyz):,} points")

    # --- Save .ply ---
    save_ply(xyz, rgb, args.out)
    print(f"\nSaved → {args.out}")
    print("\nTo view on your laptop:")
    print(f"  scp <pod-user>@<pod-ip>:<pod-path>/{args.out.name} .")
    print("  Open in MeshLab: File → Import Mesh")


if __name__ == "__main__":
    main()
