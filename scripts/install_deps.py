"""Install all Python dependencies for scene-query.

Installs:
  - requirements.txt (numpy, scipy, opencv, rerun, ultralytics, gdown, etc.)
  - vggt-omega Python package (cloned from GitHub)
  - Grounding DINO (open-vocabulary detection fallback)
  - SAM 2 (segmentation for Grounding DINO fallback)

Usage:
    python scripts/install_deps.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
OMEGA_GH   = "https://github.com/facebookresearch/vggt-omega.git"
GDINO_GH   = "https://github.com/IDEA-Research/GroundingDINO.git"


def run(cmd: list, label: str) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: '{label}' failed (exit {result.returncode}).")
        sys.exit(result.returncode)


def _ensure_cuda_home() -> None:
    """Detect and export CUDA_HOME — required for GroundingDINO CUDA compilation."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        print(f"  CUDA_HOME already set: {os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')}")
        return
    candidates = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-11.8",
    ]
    for c in candidates:
        if Path(c).exists():
            os.environ["CUDA_HOME"] = c
            print(f"  CUDA_HOME auto-detected: {c}")
            return
    print("  WARNING: CUDA_HOME not found — GroundingDINO CUDA extension may fail to compile.")
    print("  If compilation fails, run:  export CUDA_HOME=/usr/local/cuda  then re-run setup.py")


def main() -> None:
    pip = [sys.executable, "-m", "pip", "install"]

    print("=== Installing dependencies ===\n")

    print("[1/4] Requirements (requirements.txt)...")
    run(pip + ["-r", str(REPO_ROOT / "requirements.txt")], "requirements.txt")

    print("\n[2/4] vggt-omega Python package (from GitHub)...")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "vggt-omega"
        run(["git", "clone", "--depth=1", OMEGA_GH, str(src)], "git clone vggt-omega")
        run(pip + [str(src), "--no-deps"], "pip install vggt-omega")

    print("\n[3/4] Grounding DINO (open-vocabulary detection)...")
    # ninja makes CUDA extension compilation much faster
    run(pip + ["ninja"], "ninja")
    _ensure_cuda_home()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "GroundingDINO"
        run(["git", "clone", "--depth=1", GDINO_GH, str(src)], "git clone GroundingDINO")
        run(pip + ["--no-build-isolation", str(src)], "pip install groundingdino")

    print("\n[4/4] SAM 2 (segmentation for fallback)...")
    # --no-deps prevents sam2 from upgrading PyTorch to an incompatible CUDA build
    run(pip + ["sam2", "--no-deps"], "pip install sam2")

    print("\n=== All dependencies installed. ===")


if __name__ == "__main__":
    main()
