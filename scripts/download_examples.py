"""
Download example tabletop images into examples/tabletop/.

The images are already included in the repo — only run this if you
deleted them or need to restore them.

Usage:
    python scripts/download_examples.py
"""

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEST_DIR = REPO_ROOT / "examples" / "tabletop"

BASE_URL = "https://raw.githubusercontent.com/facebookresearch/vggt/main/examples/kitchen/images"
IMAGE_NAMES = [f"{i:02d}.png" for i in range(0, 25)]  # 00.png through 24.png


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(IMAGE_NAMES)} tabletop images -> {DEST_DIR}/\n")

    for name in IMAGE_NAMES:
        dest = DEST_DIR / name
        if dest.exists():
            print(f"  {name} already exists, skipping")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    print(f"\nDone. {len(list(DEST_DIR.iterdir()))} images in {DEST_DIR}/")
    print("\nRun the pipeline with:")
    print("  python reconstruct.py --images examples/tabletop/ --out outputs/tabletop/")


if __name__ == "__main__":
    main()
