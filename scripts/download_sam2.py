"""
Download SAM 2.1 weights.

Run this before using Grounded-SAM-2 (NOT needed for VGGT-only testing).

Usage:
    python scripts/download_sam2.py                        # default path
    python scripts/download_sam2.py --dir /custom/path     # custom weights dir

Default weights location: <repo>/checkpoints/sam2/
On RunPod, pass --dir /workspace/checkpoints/sam2 to keep weights on the
persistent volume so they survive pod restarts.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DIR = REPO_ROOT / "checkpoints" / "sam2"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
WEIGHT_FILENAME = "sam2.1_hiera_large.pt"
EXPECTED_SIZE_MB = 2400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Where to save weights")
    return p.parse_args()


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {pct:.1f}%  ({mb:.0f} MB)", end="", flush=True)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.dir
    weight_file = checkpoint_dir / WEIGHT_FILENAME

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if weight_file.exists():
        size_mb = weight_file.stat().st_size / 1e6
        print(f"Already downloaded: {weight_file}  ({size_mb:.0f} MB)")
        return

    print(f"Downloading SAM 2.1 large (~{EXPECTED_SIZE_MB} MB)...")
    print(f"  Source : {WEIGHT_URL}")
    print(f"  Dest   : {weight_file}\n")

    try:
        urllib.request.urlretrieve(WEIGHT_URL, weight_file, reporthook=_progress)
    except Exception as e:
        print(f"\nERROR: Download failed — {e}")
        if weight_file.exists():
            weight_file.unlink()
        sys.exit(1)

    print(f"\n\nDone. Saved to {weight_file}")


if __name__ == "__main__":
    main()
