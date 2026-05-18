"""Download model weights from OneDrive into checkpoints/.

Always downloads VGGT-Omega (too large for git).
Only downloads YOLO weights if checkpoints/yolo_office/best.pt is missing.

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import base64
import sys
import zipfile
from pathlib import Path

# ── OneDrive share URLs ───────────────────────────────────────────────────────
# After uploading, paste each file's share link here (Anyone with link → View).
VGGT_OMEGA_SHARE_URL = "PASTE_ONEDRIVE_SHARE_URL_FOR_vggt_omega.zip"
YOLO_SHARE_URL       = "PASTE_ONEDRIVE_SHARE_URL_FOR_yolo_office.zip"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent
CKPT_ROOT  = REPO_ROOT / "checkpoints"
YOLO_PT    = CKPT_ROOT / "yolo_office" / "best.pt"
VGGT_DIR   = CKPT_ROOT / "vggt_omega"


def _onedrive_direct(share_url: str) -> str:
    """Convert a OneDrive share link to a direct-download URL."""
    encoded = base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{encoded}/root/content"


def _download(url: str, dest: Path, label: str) -> None:
    try:
        import requests
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests"], check=True)
        import requests

    print(f"  Downloading {label}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.1f}%  ({downloaded/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print(f"\r    Done — {downloaded/1e6:.0f} MB saved to {dest}")


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print(f"  Extracted to {dest_dir}/")


def download_vggt_omega() -> None:
    if VGGT_OMEGA_SHARE_URL.startswith("PASTE"):
        print("ERROR: Set VGGT_OMEGA_SHARE_URL in scripts/download_models.py first.")
        sys.exit(1)

    zip_path = CKPT_ROOT / "vggt_omega.zip"
    _download(_onedrive_direct(VGGT_OMEGA_SHARE_URL), zip_path, "VGGT-Omega (~4.3 GB)")
    _unzip(zip_path, CKPT_ROOT)

    pt = VGGT_DIR / "vggt_omega_1b_512.pt"
    if not pt.exists():
        print(f"ERROR: Expected {pt} after unzip — check the zip structure.")
        sys.exit(1)
    print(f"  VGGT-Omega ready: {pt}  ({pt.stat().st_size/1e9:.2f} GB)")


def download_yolo() -> None:
    if YOLO_PT.exists():
        print(f"  YOLO weights already present ({YOLO_PT}) — skipping.")
        return

    if YOLO_SHARE_URL.startswith("PASTE"):
        print("ERROR: Set YOLO_SHARE_URL in scripts/download_models.py first.")
        sys.exit(1)

    zip_path = CKPT_ROOT / "yolo_office.zip"
    _download(_onedrive_direct(YOLO_SHARE_URL), zip_path, "YOLO office weights (~6 MB)")
    _unzip(zip_path, CKPT_ROOT)

    if not YOLO_PT.exists():
        print(f"ERROR: Expected {YOLO_PT} after unzip — check the zip structure.")
        sys.exit(1)
    print(f"  YOLO weights ready: {YOLO_PT}")


def main() -> None:
    print("=== Downloading model weights ===\n")
    print("[1/2] VGGT-Omega")
    download_vggt_omega()
    print("\n[2/2] YOLO office detector")
    download_yolo()
    print("\nAll weights ready.")


if __name__ == "__main__":
    main()
