"""Download model weights from Google Drive into checkpoints/.

VGGT-Omega weights are always downloaded (~4.3 GB).
YOLO weights are skipped if checkpoints/yolo_office/best.pt already exists
(it ships with the repo, so this only runs when git clone is not used).

Usage:
    python scripts/download_models.py
"""
import sys
import zipfile
from pathlib import Path

# Google Drive file/folder IDs
VGGT_GDRIVE_ID  = "1K2kK6ZiL93V06uZOH4D5oE-TzQAdLDHM"   # vggt_omega zip (~4.3 GB)
YOLO_GDRIVE_ID  = "1aeM2GtvtdwIXH-pCcIqJgCu-TE8fbOfJ"   # yolo_office folder

REPO_ROOT = Path(__file__).parent.parent
CKPT_ROOT = REPO_ROOT / "checkpoints"
YOLO_PT   = CKPT_ROOT / "yolo_office" / "best.pt"
VGGT_PT   = CKPT_ROOT / "vggt_omega" / "vggt_omega_1b_512.pt"


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


def download_yolo() -> None:
    import gdown

    if YOLO_PT.exists():
        print(f"  YOLO weights already present ({YOLO_PT}) — skipping.")
        return

    print("  Downloading YOLO office weights from Google Drive...")
    dest = CKPT_ROOT / "yolo_office"
    dest.mkdir(parents=True, exist_ok=True)

    # Drive link is a folder — download all contents into dest
    gdown.download_folder(id=YOLO_GDRIVE_ID, output=str(dest), quiet=False)

    # Unzip if a zip landed in dest
    for z in dest.glob("*.zip"):
        _unzip(z, dest)

    if not YOLO_PT.exists():
        print(f"ERROR: Expected {YOLO_PT} after download — check the folder on Google Drive.")
        sys.exit(1)

    print(f"  YOLO weights ready: {YOLO_PT}")


def main() -> None:
    print("=== Downloading model weights ===\n")
    _ensure_gdown()

    print("[1/2] VGGT-Omega")
    download_vggt_omega()

    print("\n[2/2] YOLO office detector")
    download_yolo()

    print("\nAll weights ready.")


if __name__ == "__main__":
    main()
