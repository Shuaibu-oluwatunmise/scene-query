"""
Download Grounding DINO weights and config.

Run this before using Grounded-SAM-2 (NOT needed for VGGT-only testing).

Usage:
    python scripts/download_grounding_dino.py                        # default path
    python scripts/download_grounding_dino.py --dir /custom/path     # custom dir

Default weights location: <repo>/checkpoints/grounding_dino/
On RunPod, pass --dir /workspace/checkpoints/grounding_dino to keep weights
on the persistent volume so they survive pod restarts.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DIR = REPO_ROOT / "checkpoints" / "grounding_dino"

WEIGHT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
EXPECTED_SIZE_MB = 700


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Where to save weights")
    return p.parse_args()


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {pct:.1f}%  ({mb:.0f} MB)", end="", flush=True)


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\nERROR: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Grounding DINO to {checkpoint_dir}/\n")
    _download(WEIGHT_URL, checkpoint_dir / "groundingdino_swint_ogc.pth", f"weights (~{EXPECTED_SIZE_MB} MB)")
    _download(CONFIG_URL, checkpoint_dir / "GroundingDINO_SwinT_OGC.py", "config")

    print(f"\nDone. Files in {checkpoint_dir}/:")
    for f in sorted(checkpoint_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
