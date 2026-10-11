# Aura Scene Lab

Scene Lab keeps NVIDIA Kimodo as Aura's body-motion generator and adds a narrow scene-aware locomotion layer.

## Camera-first pipeline

```text
browser camera / phone camera
          ↓
short static-scene video
          ↓
Nerfstudio camera-pose processing (COLMAP)
          ↓
Splatfacto Gaussian reconstruction
          ↓
web .splat preview loaded automatically
          +
explicit floor / obstacle geometry proxy
          ↓
A* root-path planner
          +
NVIDIA Kimodo G1 motion
          ↓
Aura root-path fitting
          ↓
saved G1 scene trajectory
          ↓
existing Aura preference loop
```

The key methodological boundary is intentional: **the Gaussian splat is visual context, not collision geometry**. Ordinary monocular camera reconstruction also has arbitrary global scale unless calibrated. Scene Lab therefore plans against an explicit 2D geometry proxy with an expanded robot radius.

## Camera capture workflow

The Scene Lab page now has **Capture room** beside **Demo scene** and **Load .splat**.

1. Open Aura on `localhost` (or HTTPS, because browser camera APIs require a secure context).
2. Click **Capture room**.
3. Prefer the rear phone/laptop camera when available.
4. Click **Start scan** and move slowly around a static room for roughly 20–40 seconds.
5. Revisit surfaces from overlapping angles, avoid motion blur, and avoid moving people/objects.
6. Click **Stop & reconstruct**.
7. Aura uploads the video to the local Aura API, starts Gaussian reconstruction in a background thread, polls its status, and automatically loads `scene.splat` when reconstruction finishes.

Camera videos and reconstruction artifacts live under `aura-scene-captures/`, which is ignored by Git.

## Reconstruction backend

Aura supports two reconstruction modes.

### Nerfstudio / Splatfacto

If `ns-process-data`, `ns-train`, and `ns-export` are on PATH, Aura uses them directly. You can also keep Nerfstudio in a separate Conda environment:

```bash
export AURA_SPLAT_NERFSTUDIO_ENV=aura-splat
```

Aura then executes the equivalent of:

```text
ns-process-data video ...
ns-train splatfacto ...
ns-export gaussian-splat ...
```

The exported Gaussian PLY is converted locally into the common 32-byte `.splat` format used by Aura's lightweight web preview.

Useful tuning variables:

```bash
export AURA_SPLAT_TRAIN_STEPS=7000
export AURA_SPLAT_FRAMES=120
export AURA_SPLAT_MAX_UPLOAD_MB=300
```

Because Splatfacto uses the GPU, Aura releases its resident Kimodo model before reconstruction starts. Kimodo is loaded again automatically the next time motion generation is requested.

### Custom reconstruction command

For another 3DGS pipeline, configure an executable or wrapper:

```bash
export AURA_SPLAT_RECON_COMMAND=/absolute/path/to/reconstruct_splat.sh
```

Aura calls it with exactly three arguments:

```text
reconstruct_splat.sh INPUT_VIDEO WORK_DIRECTORY OUTPUT_SPLAT
```

The command must write the final 32-byte-record `.splat` file to `OUTPUT_SPLAT`.

## Scene capture API

```text
GET  /scene/capture/status
POST /scene/captures                       raw camera video body
GET  /scene/captures/<capture_id>
POST /scene/captures/<capture_id>/reconstruct
GET  /scene/captures/<capture_id>/splat

GET  /scene/default
POST /scene/plan
POST /scene/generate
```

Reconstruction is asynchronous. Status responses expose coarse stages such as camera-pose estimation, Gaussian training, export, conversion, ready, or error. The main Aura UI remains usable while this runs.

## What Scene Lab does after reconstruction

- Shows the captured Gaussian scene in the browser.
- Keeps start/goal positions and obstacle geometry separate from the splat.
- Runs an 8-connected A* planner against the expanded obstacle proxy.
- Generates exactly two G1 motions with NVIDIA Kimodo.
- Fits each generated motion's global/root path onto the planned route while preserving the body animation as much as possible.
- Saves the fitted trajectories into Aura's normal motion library so A/B choices can enter preference learning.

## Gaussian-splat viewer limitation

The bundled browser viewer is deliberately lightweight for the hackathon. It previews standard 32-byte `.splat` records as camera-facing Gaussian billboards and caps the displayed splat count. It is not a production 3DGS covariance/sorting renderer.

Likewise, a phone video reconstructed through COLMAP has no guaranteed metric scale. A future version can add metric calibration, depth/LiDAR capture, automatic geometry extraction, and a full production splat renderer.

## Claims Aura should make

Safe wording:

> Aura captures a real environment as a Gaussian-splat visual digital twin, plans with an explicit geometry proxy, and curates scene-fitted G1 locomotion demonstrations using NVIDIA Kimodo and human preference learning.

Do not claim:

- that NVIDIA Kimodo is scene-conditioned,
- that Gaussian opacity is collision geometry,
- that uncalibrated monocular splats have metric scale,
- physical balance or safety,
- contact-rich interaction such as precise sitting, grasping, or reaching,
- physics simulation.
