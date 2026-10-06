#!/usr/bin/env bash
# =============================================================================
# setup_vm.sh — one-time environment setup on a Linux machine with an NVIDIA GPU
#
# Usage:   bash scripts/setup_vm.sh
#
# What it does:
#   1. checks the GPU is visible
#   2. creates an isolated Python virtual environment (.venv)
#   3. installs PyTorch matching the machine's CUDA version
#   4. installs PyTorch Geometric and the remaining dependencies
#   5. verifies that torch can actually see the GPU
#
# It is safe to re-run: existing steps are skipped rather than repeated.
# =============================================================================

set -e   # stop immediately if any command fails, rather than continuing broken

echo "=============================================================="
echo " AML project — environment setup"
echo "=============================================================="

# ---------------------------------------------------------------- 1. GPU check
echo
echo "[1/5] Checking for an NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found."
    echo "  The NVIDIA driver is not installed, or this machine has no GPU."
    echo "  Ask your administrator to install the driver, then re-run this script."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "  Driver reports CUDA ${CUDA_VER}"

# ---------------------------------------------------------------- 2. Python
echo
echo "[2/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: python3 not found. Install it with:  sudo apt install python3 python3-venv"
    exit 1
fi
python3 --version

# ---------------------------------------------------------------- 3. venv
echo
echo "[3/5] Creating the virtual environment (.venv)..."
# A virtual environment keeps this project's packages separate from the system
# Python, so nothing here can break other users' work on a shared machine.
if [ -d ".venv" ]; then
    echo "  .venv already exists — reusing it."
else
    python3 -m venv .venv
    echo "  created."
fi
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
echo "  using: $(which python)"

# ---------------------------------------------------------------- 4. PyTorch
echo
echo "[4/5] Installing PyTorch (this can take several minutes)..."
if python -c "import torch" 2>/dev/null; then
    echo "  torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    # Pick the wheel index matching the driver's CUDA major version. Installing a
    # build newer than the driver supports is the usual cause of "torch installed
    # but torch.cuda.is_available() is False".
    CUDA_MAJOR=${CUDA_VER%%.*}
    if   [ "$CUDA_MAJOR" -ge 12 ]; then IDX="https://download.pytorch.org/whl/cu121"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then IDX="https://download.pytorch.org/whl/cu118"
    else                                IDX="https://download.pytorch.org/whl/cpu"
         echo "  WARNING: CUDA ${CUDA_VER} is old; falling back to a CPU build."
    fi
    echo "  index: ${IDX}"
    pip install torch --index-url "${IDX}"
fi

echo
echo "      Installing PyTorch Geometric and dependencies..."
pip install --quiet torch-geometric pandas numpy scikit-learn matplotlib jupyter ipykernel

# ---------------------------------------------------------------- 5. verify
echo
echo "[5/5] Verifying the installation..."
python - <<'PYCHECK'
import torch
print(f"  torch            : {torch.__version__}")
print(f"  CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU memory       : {total:.1f} GiB")
else:
    print("  WARNING: torch cannot see the GPU. Training will fall back to CPU,")
    print("  which is far too slow for this project. Check that the installed")
    print("  torch build matches the driver's CUDA version.")
import torch_geometric
print(f"  torch_geometric  : {torch_geometric.__version__}")
PYCHECK

echo
echo "=============================================================="
echo " Setup complete."
echo
echo " Activate the environment in every new terminal with:"
echo "     source .venv/bin/activate"
echo "=============================================================="
