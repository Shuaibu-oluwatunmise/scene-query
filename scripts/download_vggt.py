"""
Install VGGT and download its weights.

Run this once on the pod. It handles everything — no separate pip install needed.

Usage:
    python scripts/download_vggt.py

What it does:
    1. pip installs the VGGT package from GitHub
    2. Downloads the VGGT weights from HuggingFace (~2 GB)
       and saves them to /workspace/checkpoints/vggt/ (persistent volume)

Output:
    /workspace/checkpoints/vggt/   <- weights land here
"""

import os
import subprocess
import sys
from pathlib import Path

CHECKPOINT_DIR = Path("/workspace/checkpoints/vggt")
HF_MODEL_ID = "facebook/vggt"
VGGT_REPO = "git+https://github.com/facebookresearch/vggt.git"


def install_vggt():
    print("Step 1/2 — Installing VGGT package...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", VGGT_REPO],
        check=False,
    )
    if result.returncode != 0:
        print("\nERROR: pip install failed. See output above.")
        sys.exit(1)
    print("VGGT package installed.\n")


def download_weights():
    print("Step 2/2 — Downloading VGGT weights from HuggingFace...")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Dest  : {CHECKPOINT_DIR}")
    print("  Size  : ~2 GB — takes 2-5 minutes on the pod\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep HF cache on the persistent volume, not the container disk
    os.environ["HF_HOME"] = "/workspace/checkpoints/.hf_cache"

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )

    print(f"\nWeights saved to {CHECKPOINT_DIR}/")
    print("Files:")
    for f in sorted(CHECKPOINT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(CHECKPOINT_DIR)}  ({size_mb:.0f} MB)")


def main():
    install_vggt()
    download_weights()
    print("\nAll done. VGGT is ready.")


if __name__ == "__main__":
    main()
