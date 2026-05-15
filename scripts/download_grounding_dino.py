"""
Download Grounding DINO weights to /workspace/checkpoints/grounding_dino/.

Run this before using Grounded-SAM-2 (NOT needed for VGGT-only testing).
You also need Grounding DINO installed: see Grounded-SAM-2 install instructions.

Usage:
    python scripts/download_grounding_dino.py

What it does:
    Downloads the Grounding DINO SwinT-OGC checkpoint.
    This is the detector half of Grounded-SAM-2 — it finds bounding boxes
    for text-prompted labels before SAM 2 refines them into masks.

Output:
    /workspace/checkpoints/grounding_dino/groundingdino_swint_ogc.pth
    /workspace/checkpoints/grounding_dino/GroundingDINO_SwinT_OGC.py  (config)
"""

import sys
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path("/workspace/checkpoints/grounding_dino")

WEIGHT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

WEIGHT_FILE = CHECKPOINT_DIR / "groundingdino_swint_ogc.pth"
CONFIG_FILE = CHECKPOINT_DIR / "GroundingDINO_SwinT_OGC.py"

EXPECTED_SIZE_MB = 700


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {pct:.1f}%  ({mb:.0f} MB)", end="", flush=True)


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\nERROR: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    _download(WEIGHT_URL, WEIGHT_FILE, f"Grounding DINO weights (~{EXPECTED_SIZE_MB} MB)")
    _download(CONFIG_URL, CONFIG_FILE, "Grounding DINO config")

    print(f"\nDone. Files in {CHECKPOINT_DIR}:")
    for f in sorted(CHECKPOINT_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
