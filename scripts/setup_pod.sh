#!/bin/bash
# Run this after every pod migration to restore the Python environment.
# Weights in /workspace/checkpoints/ are already there — no re-download needed.
#
# Usage:
#   bash scripts/setup_pod.sh

set -e

echo "=== scene-query pod setup ==="
echo ""

# Pull latest code
echo "[1/3] Pulling latest code..."
cd /workspace/scene-query
git remote set-url origin https://github.com/Shuaibu-oluwatunmise/scene-query.git
git pull
echo ""

# Install Python deps
echo "[2/3] Installing Python packages..."
pip install -q git+https://github.com/facebookresearch/vggt.git
pip install -q -r requirements.txt
echo ""

# Verify GPU
echo "[3/3] Checking environment..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import vggt; print('vggt installed OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Setup complete. Weights are at /workspace/checkpoints/ ==="
echo ""
echo "Run the smoke test:"
echo "  python3 scripts/test_vggt.py --weights /workspace/checkpoints/vggt --out /workspace/outputs/vggt_test.ply"
