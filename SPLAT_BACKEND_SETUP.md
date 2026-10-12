# Aura Gaussian-splat backend setup

The message `Missing: ns-process-data, ns-train, ns-export` means the Aura web UI is running, but the **Nerfstudio command-line tools are not installed in the environment Aura can access**.

Keep Nerfstudio separate from the Kimodo/Aura Python environment.

## Recommended layout

- `aura` — existing Aura + Kimodo environment
- `aura-splat` — Nerfstudio / Splatfacto + COLMAP + ffmpeg

A minimal Linux setup is:

```bash
conda create -n aura-splat -y python=3.10
conda activate aura-splat
python -m pip install --upgrade pip

# Install a CUDA-compatible PyTorch build for your machine first if needed.
# Then install Nerfstudio + the Gaussian rasterizer.
pip install nerfstudio gsplat

# Camera-video processing needs these tools too.
conda install -y -c conda-forge colmap ffmpeg
```

Verify:

```bash
ns-process-data --help
ns-train splatfacto --help
ns-export gaussian-splat --help
ffmpeg -version
colmap -h
```

Then return to the Aura environment and tell Aura which environment to use:

```bash
conda activate aura
cd ~/Documents/Aura-Colosseum
export AURA_SPLAT_NERFSTUDIO_ENV=aura-splat
./scripts/run_aura_live.sh
```

Or run:

```bash
export AURA_SPLAT_NERFSTUDIO_ENV=aura-splat
./scripts/check_splat_backend.sh
```

Restart Aura after installing/changing the reconstruction environment because backend availability is cached for the server process.

## What this patch also fixes

- Camera Splatfacto exports are converted from **Nerfstudio Z-up** world coordinates to **Aura/Three.js Y-up** coordinates before writing `scene.splat`.
- Manually loaded `.splat` files receive an initial 180° upright correction, because standalone splat files do not encode a universal world-up convention.
- Scene Lab now exposes `Flip 180°`, `Rotate +90°`, `Rotate -90°`, and `Raw axes` controls.
- Camera upload now times out with a useful error instead of appearing stuck forever.
- Browser recording bitrate is reduced from 8 Mbps to 4 Mbps to make localhost uploads faster while still retaining useful reconstruction frames.
