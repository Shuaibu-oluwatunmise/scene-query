"""Download model weights into checkpoints/.

Downloads:
  - VGGT-Omega weights (~4.3 GB) from Google Drive
  - Grounding DINO weights + config (~700 MB) from GitHub
  - SAM 2.1 large weights (~900 MB) from Meta CDN

Usage:
    python scripts/download_models.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

VGGT_GDRIVE_ID  = "12AVRrnZ86Yn6v6GhL5jbbJkV_RuKXO-9"

# Grounding DINO + SAM 2.1
GDINO_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GDINO_CONFIG_URL  = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
SAM21_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

REPO_ROOT     = Path(__file__).parent.parent
CKPT_ROOT     = REPO_ROOT / "checkpoints"
VGGT_PT       = CKPT_ROOT / "vggt_omega" / "vggt_omega_1b_512.pt"
GDINO_DIR     = CKPT_ROOT / "grounding_dino"
GDINO_WEIGHTS = GDINO_DIR / "groundingdino_swint_ogc.pth"
GDINO_CONFIG  = GDINO_DIR / "GroundingDINO_SwinT_OGC.py"
SAM2_DIR      = CKPT_ROOT / "sam2"
SAM2_WEIGHTS  = SAM2_DIR  / "sam2.1_hiera_large.pt"


def _download_file(url: str, dest: Path, label: str) -> None:
    """Download a URL to dest with a simple progress hook."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label}...")

    def _hook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
    print(f"\r  Done -> {dest}  ({dest.stat().st_size / 1e6:.0f} MB)")


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    # If zip extracted into a single subdirectory, flatten it into dest_dir
    children = [p for p in dest_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        sub = children[0]
        for item in sub.iterdir():
            item.rename(dest_dir / item.name)
        sub.rmdir()


def download_vggt_omega() -> None:
    import gdown

    if VGGT_PT.exists():
        print(f"  VGGT-Omega already present ({VGGT_PT.stat().st_size/1e9:.2f} GB) — skipping.")
        return

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = CKPT_ROOT / "vggt_omega.zip"

    print("  Downloading VGGT-Omega weights from Google Drive (~4.3 GB)...")
    gdown.download(id=VGGT_GDRIVE_ID, output=str(zip_path), fuzzy=True, quiet=False)

    if not zip_path.exists():
        print("ERROR: Download failed — file not found after gdown.")
        sys.exit(1)

    # If it downloaded as the .pt directly (not zipped), move it into place
    if not zipfile.is_zipfile(zip_path):
        VGGT_PT.parent.mkdir(parents=True, exist_ok=True)
        zip_path.rename(VGGT_PT)
    else:
        _unzip(zip_path, CKPT_ROOT)

    if not VGGT_PT.exists():
        print(f"ERROR: Expected {VGGT_PT} after download — check the file on Google Drive.")
        sys.exit(1)

    print(f"  VGGT-Omega ready: {VGGT_PT}  ({VGGT_PT.stat().st_size/1e9:.2f} GB)")


def download_gsam2() -> None:
    if GDINO_WEIGHTS.exists():
        print(f"  Grounding DINO weights already present — skipping.")
    else:
        _download_file(GDINO_WEIGHTS_URL, GDINO_WEIGHTS, "Grounding DINO weights (~700 MB)")

    if GDINO_CONFIG.exists():
        print(f"  Grounding DINO config already present — skipping.")
    else:
        _download_file(GDINO_CONFIG_URL, GDINO_CONFIG, "Grounding DINO config")

    if SAM2_WEIGHTS.exists():
        print(f"  SAM 2.1 weights already present — skipping.")
    else:
        _download_file(SAM21_WEIGHTS_URL, SAM2_WEIGHTS, "SAM 2.1 large weights (~900 MB)")


def main() -> None:
    print("=== Downloading model weights ===\n")
    _ensure_gdown()

    print("[1/2] VGGT-Omega")
    download_vggt_omega()

    print("\n[2/2] Grounding DINO + SAM 2.1")
    download_gsam2()

    print("\nAll weights ready.")


if __name__ == "__main__":
    main()
