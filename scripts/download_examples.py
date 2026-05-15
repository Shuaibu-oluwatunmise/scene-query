"""
Download VGGT example room images into examples/room/.

These are the images used for development and demo purposes.
Run this once — no GPU needed.

Usage:
    python scripts/download_examples.py
"""

import urllib.request
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEST_DIR = REPO_ROOT / "examples" / "room"

BASE_URL = "https://raw.githubusercontent.com/facebookresearch/vggt/main/examples/room/images"
IMAGE_NAMES = ["no_overlap_1.png"] + [f"no_overlap_{i}.jpg" for i in range(2, 9)]


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(IMAGE_NAMES)} room images -> {DEST_DIR}/\n")

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

    print(f"\nDone. {len(IMAGE_NAMES)} images in {DEST_DIR}/")
    print("\nRun the pipeline with:")
    print("  python reconstruct.py --images examples/room/ --out outputs/room/")


if __name__ == "__main__":
    main()
