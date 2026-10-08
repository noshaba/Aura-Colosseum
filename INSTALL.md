# Aura / Text2Motion Aura — New Machine Installation

This guide installs the **live Aura stack** on a fresh Linux machine.

Aura's live stack consists of:

1. **Text2Motion Aura text encoder** — runs on CPU by default.
2. **Text2Motion Aura G1 generator** — runs on the NVIDIA GPU.
3. **Aura local API** — stores motions, comparisons, learned priors, and experiment data.
4. **Aura web app** — React/Vite frontend at `http://localhost:5173`.

The standalone Text2Motion Aura/Viser demo is **not required** for the current Aura UI.

---

## 1. Recommended machine

For the live generator:

- Ubuntu 22.04/24.04 or another recent Linux distribution
- NVIDIA GPU with **16 GB VRAM minimum recommended** for the G1 model
- 32 GB system RAM recommended
- NVIDIA driver capable of running the chosen CUDA-enabled PyTorch wheel
- 20+ GB free disk space for Python packages, npm packages, Hugging Face models, and generated motions

For **Judge Mode only**, an NVIDIA GPU is not required.

> On 16 GB GPUs, keep the 8B text encoder on CPU. Aura's launcher already does this by default.

---

## 2. Install system tools

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y \
  git curl build-essential ffmpeg \
  libgl1 libglib2.0-0 libsndfile1
```

Verify the NVIDIA driver:

```bash
nvidia-smi
```

If `nvidia-smi` does not work, install/fix the NVIDIA driver before continuing.

---

## 3. Install Miniconda

If Conda is already installed, skip this section.

Install Miniconda using the official installer for your machine, restart the shell, and verify:

```bash
conda --version
```

Create a dedicated Aura environment with Python 3.10:

```bash
conda create -n aura python=3.10 -y
conda activate aura
```

Keep Aura isolated from user-site Python packages:

```bash
export PYTHONNOUSERSITE=1
```

---

## 4. Install CUDA-enabled PyTorch

**Do not install a CUDA build newer than your NVIDIA driver supports.** A mismatch can make `torch.cuda.is_available()` return `False` even when the GPU is visible in `nvidia-smi`.

A known-working setup for NVIDIA driver 535 / CUDA 12.2-capable systems is PyTorch 2.5.1 with the CUDA 12.1 wheel:

```bash
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

For a different driver/GPU, use a PyTorch CUDA wheel compatible with that driver.

Verify:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Do not continue with live generation until `CUDA available: True`.

---

## 5. Install Python support packages

Inside the `aura` Conda environment:

```bash
conda install -c conda-forge cffi libsndfile -y
python -m pip install -U pip setuptools wheel
python -m pip install -U huggingface_hub soundfile
```

This also avoids the common `_cffi_backend` / `soundfile` import problem.

---

## 6. Hugging Face access

The text encoder relies on Llama/LLM2Vec model weights and the motion generator downloads NVIDIA model weights from Hugging Face.

Before installing/running Aura:

1. Sign in to Hugging Face.
2. Make sure the account has access to **Meta-Llama-3-8B-Instruct**.
3. Accept any model-license/access prompts required for the NVIDIA G1 model if Hugging Face asks you to.
4. Create a **read** access token.

Log in from the terminal:

```bash
hf auth login
```

Verify:

```bash
hf auth whoami
```

Never commit or share the Hugging Face token.

---

## 7. Install Text2Motion Aura

From the Aura repository root:

```bash
conda activate aura
export PYTHONNOUSERSITE=1

cd text2motion-aura
python -m pip install -e .
cd ..
```

Aura uses Text2Motion Aura directly as a Python generation library. You do **not** need to install the optional Viser demo dependencies for the normal Aura web interface.

Verify imports:

```bash
python - <<'PY'
import torch
import text2motion_aura
print("Text2Motion Aura import: OK")
print("CUDA available:", torch.cuda.is_available())
PY
```

---

## 8. Install Node.js

Aura requires Node.js 18+; Node.js 20 LTS is recommended.

If Node is already installed:

```bash
node --version
npm --version
```

Then install the frontend dependencies:

```bash
cd aura-web
npm install
npm run build
cd ..
```

Keep the generated `aura-web/package-lock.json` in the repository for reproducible installs.

---

## 9. Run Aura

From the repository root:

```bash
conda activate aura
export PYTHONNOUSERSITE=1
chmod +x scripts/run_aura_live.sh
./scripts/run_aura_live.sh
```

The launcher starts:

```text
Text2Motion Aura text encoder  -> CPU, port 9550
Aura API + G1 generator        -> GPU, port 8765
Aura web app                   -> port 5173
```

Open:

```text
http://localhost:5173
```

The first generation can take longer because model weights may need to download and the G1 model is loaded lazily. Later generations reuse the loaded model.

---

## 10. GPU-memory configuration

The supplied launcher is optimized for GPUs around 16 GB:

```bash
TEXT_ENCODER_DEVICE=cpu
TEXT_ENCODER_MODE=api
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The text encoder therefore uses CPU/RAM while the motion model uses the GPU.

Before launching, if you previously ran Aura/Text2Motion processes, check GPU memory:

```bash
nvidia-smi
```

If an old process is holding most of the GPU:

```bash
pkill -f text2motion_aura || true
pkill -f aura_library_server.py || true
```

Then check `nvidia-smi` again.

---

## 11. Optional environment variables

The launcher provides useful defaults, but you can override them before launch:

```bash
# Number of denoising steps. Lower = faster previews.
export AURA_TEXT2MOTION_STEPS=30

# Motion model alias.
export AURA_TEXT2MOTION_MODEL=g1-rp

# Custom motion-library path.
export AURA_MOTION_LIBRARY=/path/to/aura-motion-library

# Solana devnet RPC. Browser and Python must use the same network.
export AURA_SOLANA_RPC_URL=https://api.devnet.solana.com
export VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
```

Then:

```bash
./scripts/run_aura_live.sh
```

---

## 12. Run Judge Mode without a GPU

Judge Mode is a static frontend and does not require Python, CUDA, or downloaded model weights.

```bash
cd aura-web
npm install
npm run judge
```

Or create a production Judge Mode build:

```bash
npm run build:judge
```

---

## 13. Run the tests

From the repository root with the Aura Conda environment active:

```bash
./scripts/test.sh
```

For a clean frontend reproducibility check:

```bash
cd aura-web
rm -rf node_modules
npm ci
npm run build
npm run build:judge
```

---

## Troubleshooting

### `vite: not found`

Frontend dependencies are missing:

```bash
cd aura-web
npm install
```

Then restart Aura.

### `ModuleNotFoundError: No module named 'torch'`

Install a CUDA-enabled PyTorch build that is compatible with the installed NVIDIA driver. Do not blindly install the newest CUDA wheel.

### `The NVIDIA driver on your system is too old`

Your PyTorch wheel was compiled for a newer CUDA runtime than the driver supports. Either update the driver or install an older compatible PyTorch CUDA wheel.

### `CUDA available: False`

Run:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

If `torch.version.cuda` is `None`, CPU-only PyTorch is installed.

### CUDA out of memory

First verify that the text encoder is on CPU and that no old process is holding VRAM:

```bash
nvidia-smi
```

The normal Aura launcher sets the text encoder to CPU automatically. If memory is still tight, reduce generation steps:

```bash
export AURA_TEXT2MOTION_STEPS=20
./scripts/run_aura_live.sh
```

### `_cffi_backend` / SoundFile error

```bash
conda activate aura
conda install -c conda-forge cffi libsndfile -y
python -m pip install -U soundfile
export PYTHONNOUSERSITE=1
```

### `hf: command not found`

```bash
python -m pip install -U huggingface_hub
hf auth login
```

### Model download returns 401/403

Check:

```bash
hf auth whoami
```

Then confirm your Hugging Face account has access to the gated Llama model and has accepted any required model terms.

### Port already in use

Aura uses ports `9550`, `8765`, and `5173`.

Find the process:

```bash
ss -ltnp | grep -E ':(9550|8765|5173)'
```

Stop the stale process and rerun the launcher.

### Where are logs?

The launcher writes logs under:

```text
.aura-logs/
```

Useful commands:

```bash
tail -n 100 .aura-logs/textencoder.log
tail -n 100 .aura-logs/library.log
tail -n 100 .aura-logs/aura-web.log
```

---

## Licensing / attribution

Aura includes and modifies third-party software and assets. Before redistribution or commercial use, review:

```text
THIRD_PARTY.md
text2motion-aura/LICENSE
text2motion-aura/ATTRIBUTIONS.MD
```

The Text2Motion Aura fork retains required upstream NVIDIA/third-party attribution and model identifiers. Renaming the integration does not remove upstream license obligations.
