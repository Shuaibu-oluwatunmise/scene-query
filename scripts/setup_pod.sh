#!/bin/bash
# Run this after every pod migration to restore the Python environment.
# Weights in /workspace/checkpoints/ are already there -- no re-download needed.
#
# Usage:
#   bash scripts/setup_pod.sh

set -e

echo "=== scene-query pod setup ==="
echo ""

# Pull latest code
echo "[1/4] Pulling latest code..."
cd /workspace/scene-query
git remote set-url origin https://github.com/Shuaibu-oluwatunmise/scene-query.git
git pull
echo ""

# Install Python deps
echo "[2/4] Installing Python packages..."
pip install -q git+https://github.com/facebookresearch/vggt.git
pip install -q -r requirements.txt
echo ""

# Install Grounded-SAM-2 (sam2 + groundingdino)
echo "[3/4] Installing Grounded-SAM-2..."
if [ ! -d /workspace/Grounded-SAM-2 ]; then
    git clone --depth=1 https://github.com/IDEA-Research/Grounded-SAM-2.git /workspace/Grounded-SAM-2
fi
pip install -q -e /workspace/Grounded-SAM-2
pip install -q --no-build-isolation -e /workspace/Grounded-SAM-2/grounding_dino
# Add repo root to Python path so 'grounding_dino.groundingdino' imports work
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "/workspace/Grounded-SAM-2" > "$SITE_PKGS/grounded_sam2.pth"
echo "Grounded-SAM-2 installed."
echo ""

# Verify everything
echo "[4/4] Checking environment..."
python3 -c "import torch; print('PyTorch ' + torch.__version__ + ', CUDA: ' + str(torch.cuda.is_available()))"
python3 -c "import vggt; print('vggt OK')"
python3 -c "import groundingdino; print('groundingdino OK')"
python3 -c "import sam2; print('sam2 OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Setup complete. Weights at /workspace/checkpoints/ ==="
echo ""
echo "Smoke tests:"
echo "  python3 scripts/test_vggt.py --weights /workspace/checkpoints/vggt --out /workspace/outputs/vggt_test.ply"
echo "  python3 reconstruct.py --images examples/tabletop/ --out /workspace/outputs/tabletop/ --weights-vggt /workspace/checkpoints/vggt"
