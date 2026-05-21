"""One-shot setup for scene-query.

Downloads VGGT-Omega + YOLO weights and installs all Python dependencies.
Run this once on a GPU machine before using reconstruct.py.

Usage:
    python setup.py

Prerequisites (must be done BEFORE running this):
  - Python 3.10+
  - PyTorch with CUDA installed — see https://pytorch.org/get-started/locally/
    Check your CUDA version first: nvidia-smi (top-right corner)
    Example (CUDA 12.4): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    Example (CUDA 12.1): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    Example (CUDA 11.8): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"


def check_torch() -> None:
    """Verify PyTorch is installed and CUDA is available before proceeding."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print()
        print("Install it first, matching your CUDA driver version:")
        print("  1. Run: nvidia-smi  (CUDA version shown in top-right corner)")
        print("  2. Find the right command at: https://pytorch.org/get-started/locally/")
        print()
        print("  Examples:")
        print("    CUDA 12.4: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("    CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("    CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("Then re-run: python setup.py")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("WARNING: PyTorch is installed but CUDA is not available.")
        print("  reconstruct.py requires a CUDA GPU. query.py works fine on CPU.")
        print("  If you have a GPU, make sure you installed the CUDA-enabled PyTorch build.")
        print(f"  Installed: torch {torch.__version__}")
        print()
    else:
        print(f"  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  "
              f"GPU: {torch.cuda.get_device_name(0)}")


def run(script: Path) -> None:
    print(f"\n{'='*60}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"\nSetup failed at {script.name}. Fix the error above and re-run.")
        sys.exit(result.returncode)


def main() -> None:
    print("=" * 60)
    print("scene-query setup")
    print("=" * 60)

    print("\n[0/3] Checking PyTorch + CUDA...")
    check_torch()

    print("\n[1/2] Downloading model weights...")
    run(SCRIPTS / "download_models.py")

    print("\n[2/2] Installing Python dependencies...")
    run(SCRIPTS / "install_deps.py")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("  Reconstruct a scene (GPU required):")
    print("    python reconstruct.py --video examples/myscene.mp4 \\")
    print("        --out outputs/scene --save-rrd outputs/scene.rrd")
    print()
    print("  Query a known object (uses lifted point cloud — fast):")
    print('    python query.py outputs/scene chair --save-rrd outputs/query.rrd')
    print()
    print("  Query any object (Grounding DINO + SAM2 fallback — slower):")
    print('    python query.py outputs/scene bulldozer --save-rrd outputs/query.rrd')
    print()
    print("  Show all detected objects:")
    print('    python query.py outputs/scene --save-rrd outputs/query.rrd')
    print("=" * 60)


if __name__ == "__main__":
    main()
