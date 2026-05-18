"""Install VGGT-Omega and download its weights (requires HuggingFace access).

VGGT-Omega is a gated model — you must request access at:
  https://huggingface.co/facebook/VGGT-Omega

Then run:
    python scripts/download_vggt_omega.py --token YOUR_HF_TOKEN

Or set the HF_TOKEN environment variable and omit --token.

Default weights location: <repo>/checkpoints/vggt_omega/
On RunPod: pass --dir /workspace/checkpoints/vggt_omega
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DIR = REPO_ROOT / "checkpoints" / "vggt_omega"
OMEGA_REPO_GH = "https://github.com/facebookresearch/vggt-omega.git"
CHECKPOINT_FILE = "vggt_omega_1b_512.pt"
HF_MODEL_ID = "facebook/VGGT-Omega"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Where to save weights")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--skip-weights", action="store_true",
                   help="Install package only — skip weight download")
    return p.parse_args()


def install_package() -> None:
    print("Step 1/2 — Installing vggt-omega Python package from GitHub...")
    tmp = Path("/tmp/vggt-omega-src")
    if not tmp.exists():
        subprocess.run(["git", "clone", "--depth=1", OMEGA_REPO_GH, str(tmp)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(tmp), "--no-deps"], check=True)
    print("vggt-omega installed.\n")


def download_weights(weights_dir: Path, token: str | None) -> None:
    print("Step 2/2 — Downloading VGGT-Omega weights from HuggingFace...")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  File  : {CHECKPOINT_FILE}")
    print(f"  Dest  : {weights_dir}")
    print("  Size  : ~4.3 GB\n")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download

    weights_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=HF_MODEL_ID,
        filename=CHECKPOINT_FILE,
        local_dir=str(weights_dir),
        token=token,
    )

    pt_path = weights_dir / CHECKPOINT_FILE
    size_gb = pt_path.stat().st_size / 1e9
    print(f"\nWeights saved to {pt_path}  ({size_gb:.1f} GB)")


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    install_package()
    if args.skip_weights:
        print("--skip-weights set. Done.")
    else:
        if not token:
            print("ERROR: Provide --token or set HF_TOKEN env var.")
            print("  Request access at https://huggingface.co/facebook/VGGT-Omega")
            sys.exit(1)
        download_weights(args.dir, token)
        print("\nVGGT-Omega ready. Reconstruct with:")
        print("  python reconstruct.py --video <video> --out <out>")
        print("  (--weights-vggt defaults to checkpoints/vggt_omega automatically)")


if __name__ == "__main__":
    main()
