"""
Download SAM 2.1 weights to /workspace/checkpoints/sam2/.

Run this before using Grounded-SAM-2 (NOT needed for VGGT-only testing).

Usage:
    python scripts/download_sam2.py

What it does:
    Downloads SAM 2.1 (Segment Anything Model 2.1) from Meta.
    We use the 'large' variant — best accuracy, fits in 48 GB VRAM alongside
    Grounding DINO.

Output:
    /workspace/checkpoints/sam2/sam2.1_hiera_large.pt   <- the weights file
"""

import sys
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path("/workspace/checkpoints/sam2")
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
WEIGHT_FILE = CHECKPOINT_DIR / "sam2.1_hiera_large.pt"
EXPECTED_SIZE_MB = 2400  # ~2.4 GB


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {pct:.1f}%  ({mb:.0f} MB)", end="", flush=True)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if WEIGHT_FILE.exists():
        size_mb = WEIGHT_FILE.stat().st_size / 1e6
        print(f"Already downloaded: {WEIGHT_FILE}  ({size_mb:.0f} MB)")
        return

    print(f"Downloading SAM 2.1 large (~{EXPECTED_SIZE_MB} MB)...")
    print(f"Source: {WEIGHT_URL}")
    print(f"Destination: {WEIGHT_FILE}\n")

    try:
        urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_FILE, reporthook=_progress)
    except Exception as e:
        print(f"\nERROR: Download failed — {e}")
        if WEIGHT_FILE.exists():
            WEIGHT_FILE.unlink()
        sys.exit(1)

    print(f"\n\nDone. Saved to {WEIGHT_FILE}")


if __name__ == "__main__":
    main()
