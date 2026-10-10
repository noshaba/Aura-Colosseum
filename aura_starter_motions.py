"""Materialize bundled hero G1 clips as trainable Aura motion records.

The web hero ships several preloaded G1 preview JSON files so a visitor can rate
motions before generating anything.  Historically those votes stayed only in
localStorage.  This module converts the exact same preview trajectories into
robot-native NPZ records in the Aura motion library, allowing those votes to be
persisted and consumed by the Aura reward model.

Starter clips are tagged separately from generated same-prompt candidates.  They
teach a broad motion-preference signal; they must not be presented as evidence
of prompt compliance or as a same-prompt experiment cohort.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from aura_g1_metrics import report_for_npz

STARTER_PREFIX = "starter-"
STARTER_GROUP = "starter-general-motion-quality-v1"


def ensure_starter_motions(base: Path, repo_root: Path | None = None) -> list[dict]:
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root or Path(__file__).resolve().parent).resolve()
    hero_dir = root / "aura-web" / "public" / "demo" / "hero"
    manifest_path = hero_dir / "manifest.json"
    if not manifest_path.is_file():
        return []

    manifest = json.loads(manifest_path.read_text())
    clips = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(clips, list):
        return []

    out: list[dict] = []
    for clip in clips:
        if not isinstance(clip, dict) or not clip.get("id") or not clip.get("file"):
            continue
        source = hero_dir / str(clip["file"])
        if not source.is_file():
            continue
        payload = json.loads(source.read_text())
        positions = np.asarray(payload.get("positions"), dtype=np.float32)
        rotations = np.asarray(payload.get("global_rot_mats"), dtype=np.float32)
        fps = float(payload.get("fps") or clip.get("fps") or 30.0)
        if positions.ndim != 3 or positions.shape[1:] != (34, 3) or len(positions) < 3:
            continue
        if rotations.shape != (len(positions), 34, 3, 3):
            continue

        motion_id = f"{STARTER_PREFIX}{clip['id']}"
        native_name = f"{motion_id}.npz"
        preview_name = f"{motion_id}.g1.json"
        native_path = base / native_name
        preview_path = base / preview_name
        record_path = base / f"{motion_id}.json"

        # Deterministic materialization: same bundled clip -> same bytes on a given
        # NumPy version.  Never overwrite a user's file with a different source.
        if not native_path.is_file():
            np.savez_compressed(native_path, posed_joints=positions, global_rot_mats=rotations)
        if not preview_path.is_file():
            shutil.copyfile(source, preview_path)

        evidence = report_for_npz(native_path, fps)
        record = {
            "id": motion_id,
            "name": str(clip.get("label") or clip["id"]),
            "prompt": str(clip.get("prompt") or ""),
            "model": "Unitree G1 starter reference",
            "source": "starter_reference",
            "preference_group": STARTER_GROUP,
            "preference_scope": "general_motion_quality",
            "created_at": "2026-01-01T00:00:00Z",
            "frames": int(len(positions)),
            "fps": fps,
            "native_file": native_name,
            "preview_file": preview_name,
            "preview_error": None,
            "kinematic_evaluation": evidence,
            "starter_manifest_id": str(clip["id"]),
            "starter_prompt": str(clip.get("prompt") or ""),
            "training_note": (
                "Bundled starter/reference motion. Human comparisons involving this "
                "record train Aura's broad motion-preference reward, not prompt compliance."
            ),
        }
        record_path.write_text(json.dumps(record, indent=2))
        out.append(record)
    return out
