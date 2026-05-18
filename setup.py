"""One-shot setup for scene-query.

Downloads model weights from OneDrive and installs all Python dependencies.
Run this once before using reconstruct.py or query.py.

Usage:
    python setup.py

Requirements:
  - Python 3.10+
  - pip, git
  - Internet connection (~4.3 GB download for VGGT-Omega)
  - CUDA GPU recommended for reconstruct.py (query.py runs fine on CPU)
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"


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

    run(SCRIPTS / "download_models.py")
    run(SCRIPTS / "install_deps.py")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("  Reconstruct a scene (GPU):")
    print("    python reconstruct.py --video examples/test_scenery1.mp4 --out outputs/scene1 --save-rrd outputs/scene1.rrd")
    print()
    print("  Query a scene (CPU/laptop):")
    print('    python query.py outputs/scene1 "find the chair"')
    print("=" * 60)


if __name__ == "__main__":
    main()
