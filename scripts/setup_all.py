"""One-shot setup: install everything and download all model weights.

This script sets up the full GPU pipeline (VGGT-Omega + YOLO semantics).
Run once on a GPU machine before using reconstruct.py.

Usage:
    python scripts/setup_all.py --hf-token YOUR_TOKEN
    python scripts/setup_all.py --hf-token YOUR_TOKEN --vggt-dir /workspace/checkpoints/vggt_omega

Requirements:
  - Python 3.10+, pip, git
  - CUDA GPU (for reconstruction)
  - HuggingFace account with access to facebook/VGGT-Omega
    Request at: https://huggingface.co/facebook/VGGT-Omega

For running query.py only (no GPU needed):
  pip install -r requirements.txt
  # then download a pre-built scene directory from the project
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-token", type=str, required=True,
                   help="HuggingFace token (need access to facebook/VGGT-Omega)")
    p.add_argument("--vggt-dir", type=Path,
                   default=REPO_ROOT / "checkpoints" / "vggt_omega",
                   help="Where to save VGGT-Omega weights (default: checkpoints/vggt_omega)")
    p.add_argument("--skip-vggt", action="store_true",
                   help="Skip VGGT-Omega download (weights already present)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pip = [sys.executable, "-m", "pip", "install"]

    print("=" * 60)
    print("scene-query full setup")
    print("=" * 60)

    # 1. Core Python deps
    print("\n[1/4] Installing Python dependencies...")
    run(pip + ["-r", str(REPO_ROOT / "requirements.txt")])

    # 2. Ultralytics (for YOLO semantics)
    print("\n[2/4] Installing ultralytics (YOLO)...")
    run(pip + ["ultralytics"])

    # 3. VGGT-Omega package + weights
    if not args.skip_vggt:
        print("\n[3/4] Installing VGGT-Omega...")
        run([sys.executable, str(REPO_ROOT / "scripts" / "download_vggt_omega.py"),
             "--dir", str(args.vggt_dir),
             "--token", args.hf_token])
    else:
        print("\n[3/4] Skipping VGGT-Omega (--skip-vggt set).")

    # 4. Verify YOLO weights
    yolo_weights = REPO_ROOT / "checkpoints" / "yolo_office" / "best.pt"
    print(f"\n[4/4] Checking YOLO office weights at {yolo_weights}...")
    if yolo_weights.exists():
        size_mb = yolo_weights.stat().st_size / 1e6
        print(f"  OK — {size_mb:.1f} MB")
    else:
        print("  WARNING: YOLO weights not found. The model weights are in the Git repo but")
        print("  may be gitignored. Download best.pt from:")
        print("  https://github.com/Shuaibu-oluwatunmise/office-item-classifier/tree/main/runs/detect/yolov8n_detect_V5/weights")
        print(f"  and place at: {yolo_weights}")

    print("\n" + "=" * 60)
    print("Setup complete! Usage:")
    print()
    print("  # Reconstruct a scene from video (GPU required):")
    print("  python reconstruct.py --video examples/test_scenery1.mp4 --out outputs/scene1")
    print()
    print("  # Query the scene (CPU/laptop OK):")
    print("  python query.py outputs/scene1 'find the chair'")
    print("=" * 60)


if __name__ == "__main__":
    main()
