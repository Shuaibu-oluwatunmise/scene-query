"""
Download VGGT weights to /workspace/checkpoints/vggt/.

Run this once on the pod before using reconstruct.py.

Usage:
    python scripts/download_vggt.py

What it does:
    Downloads the VGGT model from HuggingFace (facebook/vggt).
    Saves weights to /workspace/checkpoints/vggt/ — this is the persistent
    volume, so you only need to run this once per pod lifetime.

Output:
    /workspace/checkpoints/vggt/   <- weights land here
"""

import os
import sys
from pathlib import Path

CHECKPOINT_DIR = Path("/workspace/checkpoints/vggt")
HF_MODEL_ID = "facebook/vggt"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading VGGT weights from HuggingFace ({HF_MODEL_ID})...")
    print(f"Destination: {CHECKPOINT_DIR}")
    print("This is ~2 GB. Should take 2-5 minutes on the pod.\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Point HF cache at the persistent volume so nothing lands on container disk
    os.environ["HF_HOME"] = "/workspace/checkpoints/.hf_cache"

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )

    print(f"\nDone. Weights saved to {CHECKPOINT_DIR}")
    print("Files downloaded:")
    for f in sorted(CHECKPOINT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(CHECKPOINT_DIR)}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
