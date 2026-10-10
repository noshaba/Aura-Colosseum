#!/usr/bin/env bash
set -Eeuo pipefail

# Aura bootstrap installer for a fresh Linux checkout.
# Prerequisites not installed by this script:
#   - NVIDIA driver (for live generation)
#   - Conda/Miniconda
#   - Node.js >= 18 and npm
#   - Hugging Face account/model access
#
# Default known-working PyTorch wheel: 2.5.1 + CUDA 12.1.
# Override AURA_TORCH_INDEX_URL / AURA_TORCH_VERSION if your driver requires a different build.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${AURA_CONDA_ENV:-aura}"
PYTHON_VERSION="${AURA_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${AURA_TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${AURA_TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${AURA_TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX_URL="${AURA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || fail "Conda is required. Install Miniconda/Anaconda first."
command -v node >/dev/null 2>&1 || fail "Node.js >= 18 is required."
command -v npm >/dev/null 2>&1 || fail "npm is required."

NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
(( NODE_MAJOR >= 18 )) || fail "Node.js 18+ required; found $(node --version)."

# Make conda activation work from non-interactive shells.
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
else
  echo "Using existing Conda environment '$ENV_NAME'."
fi

conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel
conda install -c conda-forge cffi libsndfile -y
python -m pip install -U huggingface_hub soundfile

if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA-enabled PyTorch is already working; keeping it."
else
  echo "Installing known-working PyTorch CUDA build: torch $TORCH_VERSION from $TORCH_INDEX_URL"
  echo "If this does not match your NVIDIA driver, stop now and follow INSTALL.md section 4."
  python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  python -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is unavailable. Judge Mode can run, but live generation will not be fast/usable.")
PY

echo "Installing NVIDIA Kimodo compatibility package..."
python -m pip install -e "$ROOT/text2motion-aura"

echo "Installing Aura web dependencies..."
cd "$ROOT/aura-web"
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build

cd "$ROOT"
chmod +x scripts/run_aura_live.sh scripts/test.sh 2>/dev/null || true

cat <<MSG

Installation completed.

Before first live run:
  1. Ensure your Hugging Face account has access to Meta-Llama-3-8B-Instruct.
  2. Run: conda activate $ENV_NAME
  3. Run: hf auth login
  4. Run: ./scripts/run_aura_live.sh
  5. Open: http://localhost:5173

For a GPU-free frontend demo:
  cd aura-web && npm run judge

See INSTALL.md for troubleshooting and GPU/CUDA compatibility notes.
MSG
