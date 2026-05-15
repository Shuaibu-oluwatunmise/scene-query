"""
Install VGGT and download its weights.

Run this once before using reconstruct.py. Handles everything — no separate
pip install needed.

Usage:
    python scripts/download_vggt.py                        # default path
    python scripts/download_vggt.py --dir /custom/path     # custom weights dir
    python scripts/download_vggt.py --skip-weights         # install only, no download

Default weights location: <repo>/checkpoints/vggt/
On RunPod, pass --dir /workspace/checkpoints/vggt to keep weights on the
persistent volume so they survive pod restarts.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DIR = REPO_ROOT / "checkpoints" / "vggt"
HF_MODEL_ID = "facebook/vggt"
VGGT_REPO = "git+https://github.com/facebookresearch/vggt.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Where to save weights")
    p.add_argument("--skip-weights", action="store_true", help="Install package only, skip weight download")
    return p.parse_args()


def install_vggt() -> None:
    print("Step 1/2 — Installing VGGT package...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", VGGT_REPO],
        check=False,
    )
    if result.returncode != 0:
        print("\nERROR: pip install failed. See output above.")
        sys.exit(1)
    print("VGGT package installed.\n")


def download_weights(checkpoint_dir: Path) -> None:
    print("Step 2/2 — Downloading VGGT weights from HuggingFace...")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Dest  : {checkpoint_dir}")
    print("  Size  : ~2 GB — takes 2-5 minutes on a fast connection\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Keep HF cache alongside the weights so nothing lands in a hidden ~/.cache
    os.environ["HF_HOME"] = str(checkpoint_dir.parent / ".hf_cache")

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nWeights saved to {checkpoint_dir}/")
    print("Files:")
    for f in sorted(checkpoint_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(checkpoint_dir)}  ({size_mb:.0f} MB)")


def main() -> None:
    args = parse_args()
    install_vggt()
    if args.skip_weights:
        print("--skip-weights set. Done (no weights downloaded).")
    else:
        download_weights(args.dir)
        print("\nAll done. VGGT is ready.")


if __name__ == "__main__":
    main()
