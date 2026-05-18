"""Install all Python dependencies for scene-query.

Installs:
  - requirements.txt (rerun, numpy, scipy, opencv, pillow, etc.)
  - ultralytics (YOLOv8)
  - vggt-omega Python package (cloned from GitHub, no-deps install)

Usage:
    python scripts/install_deps.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OMEGA_GH  = "https://github.com/facebookresearch/vggt-omega.git"


def run(cmd: list, label: str) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: '{label}' failed (exit {result.returncode}).")
        sys.exit(result.returncode)


def main() -> None:
    pip = [sys.executable, "-m", "pip", "install"]

    print("=== Installing dependencies ===\n")

    print("[1/3] Core requirements (requirements.txt)...")
    run(pip + ["-r", str(REPO_ROOT / "requirements.txt")], "requirements.txt")

    print("\n[2/3] ultralytics (YOLOv8)...")
    run(pip + ["ultralytics"], "ultralytics")

    print("\n[3/3] vggt-omega Python package (from GitHub)...")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "vggt-omega"
        run(["git", "clone", "--depth=1", OMEGA_GH, str(src)], "git clone vggt-omega")
        run(pip + ["-e", str(src), "--no-deps"], "pip install vggt-omega")

    print("\n=== All dependencies installed. ===")


if __name__ == "__main__":
    main()
