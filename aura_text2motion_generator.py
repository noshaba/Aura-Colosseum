"""Persistent Text2Motion Aura G1 generation service used by Aura's local API.

This module deliberately does not expose the upstream editor. It loads the
Text2Motion Aura G1 model lazily, reuses it across requests, and publishes generated NPZ
motions into Aura's existing motion library.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = os.environ.get("AURA_TEXT2MOTION_MODEL", "g1-rp")
DEFAULT_DURATION = float(os.environ.get("AURA_TEXT2MOTION_DURATION", "5.0"))
DEFAULT_STEPS = int(os.environ.get("AURA_TEXT2MOTION_STEPS", "30"))
DEFAULT_SEED = int(os.environ.get("AURA_TEXT2MOTION_SEED", "42"))


def _text2motion_root() -> Path:
    # Repository layout fallback; installed package path is preferred when available.
    try:
        import text2motion_aura
        return Path(text2motion_aura.__file__).resolve().parent
    except Exception:
        return Path(__file__).resolve().parent / "text2motion-aura" / "text2motion_aura"


def examples_dir() -> Path:
    return _text2motion_root() / "assets" / "demo" / "examples" / "text2motion-aura-g1-rp"


def list_examples() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base = examples_dir()
    if not base.is_dir():
        return out
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        texts = meta.get("texts") or ([meta.get("text")] if meta.get("text") else [])
        durations = meta.get("durations") or ([meta.get("duration")] if meta.get("duration") else [])
        label = folder.name
        if "_" in label:
            label = label.split("_", 1)[1]
        label = label.replace("_", " ").strip().title()
        out.append({
            "id": folder.name,
            "label": label,
            "prompt": " ".join(str(x).strip() for x in texts if x).strip(),
            "texts": texts,
            "durations": durations,
            "has_constraints": (folder / "constraints.json").is_file(),
        })
    return out


class AuraText2MotionGenerator:
    def __init__(self, library_dir: Path):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._model = None
        self._resolved_model = None
        self._device = None
        self._busy = False
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        cuda_available = False
        cuda_name = None
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return {
            "model": "text2motion-aura-g1",
            "model_loaded": self._model is not None,
            "busy": self._busy,
            "cuda_available": cuda_available,
            "cuda_name": cuda_name,
            "default_duration": DEFAULT_DURATION,
            "default_diffusion_steps": DEFAULT_STEPS,
            "last_error": self._last_error,
        }

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch
        from text2motion_aura import load_model

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable to the Aura generator. Run the server from the working Text2Motion Aura environment "
                "with a CUDA-enabled PyTorch build."
            )
        self._device = "cuda:0"
        # The separate CPU text-encoder server is selected through TEXT_ENCODER_MODE=api.
        from text2motion_aura.model.registry import MODEL_INFOS
        resolved_request = MODEL_NAME
        if MODEL_NAME.lower() in {"g1", "g1-rp", "text2motion-aura-g1", "text2motion-aura-g1-rp"}:
            match = next((i.short_key for i in MODEL_INFOS if i.skeleton.upper() == "G1" and i.dataset.upper() == "RP" and i.family.upper() != "TMR"), None)
            if match is None:
                raise RuntimeError("No G1 RP generation model is available in Text2Motion Aura.")
            resolved_request = match
        self._model, self._resolved_model = load_model(
            resolved_request,
            device=self._device,
            default_family=None,
            return_resolved_name=True,
        )
        return self._model

    def _example_inputs(self, example_id: str):
        base = examples_dir().resolve()
        if not example_id or Path(example_id).name != example_id:
            raise ValueError("Invalid example id")
        folder = (base / example_id).resolve()
        try:
            folder.relative_to(base)
        except ValueError as exc:
            raise ValueError("Invalid example id") from exc
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            raise ValueError("Unknown Text2Motion Aura example")
        meta = json.loads(meta_path.read_text())
        texts = meta.get("texts") or ([meta.get("text")] if meta.get("text") else [])
        durations = meta.get("durations") or ([meta.get("duration", DEFAULT_DURATION)] * len(texts))
        if not texts:
            raise ValueError("Selected example has no text prompt")
        constraints_path = folder / "constraints.json"
        return folder, meta, texts, durations, constraints_path if constraints_path.is_file() else None

    def _publish(
        self,
        output: dict[str, Any],
        fps: float,
        prompt: str,
        sample_idx: int = 0,
        *,
        batch_id: str | None = None,
        candidate_index: int | None = None,
        candidate_count: int | None = None,
        generation_seed: int | None = None,
    ) -> dict[str, Any]:
        from aura_g1_preview import g1_preview_bytes

        def sample_array(value):
            arr = np.asarray(value)
            # Text2Motion Aura model outputs include a leading sample dimension.
            if arr.ndim > 0 and arr.shape[0] > sample_idx:
                return arr[sample_idx]
            return arr

        posed = sample_array(output["posed_joints"])
        global_rot = sample_array(output["global_rot_mats"])
        local_rot = sample_array(output["local_rot_mats"]) if "local_rot_mats" in output else None
        uid = uuid.uuid4().hex[:12]
        native_path = self.library_dir / f"{uid}.npz"
        preview_path = self.library_dir / f"{uid}.g1.json"
        arrays: dict[str, Any] = {
            "posed_joints": posed,
            "global_rot_mats": global_rot,
            "root_positions": posed[:, 0, :],
        }
        if local_rot is not None:
            arrays["local_rot_mats"] = local_rot
        if "foot_contacts" in output:
            arrays["foot_contacts"] = sample_array(output["foot_contacts"])
        np.savez(native_path, **arrays)
        preview_path.write_bytes(g1_preview_bytes(posed, fps, global_rot))
        item = {
            "id": uid,
            "name": (prompt or "Generated G1 motion").strip()[:100],
            "model": "text2motion-aura-g1",
            "sample": "aura-direct-generator",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frames": int(len(posed)),
            "fps": float(fps),
            "native_file": native_path.name,
            "preview_file": preview_path.name,
            "preview_error": None,
            "evaluation_status": "awaiting_review",
        }
        if batch_id:
            item["batch_id"] = batch_id
        if candidate_index is not None:
            item["candidate_index"] = int(candidate_index)
        if candidate_count is not None:
            item["candidate_count"] = int(candidate_count)
        if generation_seed is not None:
            item["generation_seed"] = int(generation_seed)
        tmp = self.library_dir / f"{uid}.json.tmp"
        meta = self.library_dir / f"{uid}.json"
        tmp.write_text(json.dumps(item, indent=2))
        tmp.replace(meta)
        return item

    def generate(
        self,
        *,
        prompt: str | None = None,
        example_id: str | None = None,
        duration: float | None = None,
        diffusion_steps: int | None = None,
        seed: int | None = None,
        count: int | None = None,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt and not example_id:
            raise ValueError("Enter a motion prompt or choose a Text2Motion Aura example.")
        duration = float(duration if duration is not None else DEFAULT_DURATION)
        diffusion_steps = int(diffusion_steps if diffusion_steps is not None else DEFAULT_STEPS)
        seed = int(seed if seed is not None else DEFAULT_SEED)
        count = int(count if count is not None else 2)
        if not 1.0 <= duration <= 12.0:
            raise ValueError("Duration must be between 1 and 12 seconds.")
        if not 2 <= diffusion_steps <= 300:
            raise ValueError("Diffusion steps must be between 2 and 300.")
        if not 2 <= count <= 6:
            raise ValueError("Generate between 2 and 6 motion candidates per batch.")

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Text2Motion Aura is already generating a motion. Wait for the current generation to finish.")
        self._busy = True
        self._last_error = None
        try:
            model = self._ensure_model()
            import torch
            from text2motion_aura.constraints import load_constraints_lst
            from text2motion_aura.tools import seed_everything

            constraints = []
            cfg_type = "separated"
            cfg_weight: Any = [2.0, 2.0]
            if example_id:
                _, meta, texts, durations, constraints_path = self._example_inputs(example_id)
                num_frames = [max(1, int(float(d) * model.fps)) for d in durations]
                if constraints_path:
                    constraints = load_constraints_lst(str(constraints_path), model.skeleton)
                seed = int(meta.get("seed", seed))
                # Keep the user's quicker preview step count unless explicitly requested in the POST.
                cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else None
                if cfg and not cfg.get("enabled", True):
                    cfg_type = "nocfg"
                    cfg_weight = None
                elif cfg:
                    cfg_weight = [float(cfg.get("text_weight", 2.0)), float(cfg.get("constraint_weight", 2.0))]
                prompt_label = " ".join(str(t).strip() for t in texts if t)
            else:
                texts = [prompt]
                num_frames = [max(1, int(duration * model.fps))]
                prompt_label = prompt

            # Generate candidates sequentially. On 16 GB GPUs this is more reliable than
            # batching several diffusion samples into one large CUDA allocation, while the
            # already-loaded Text2Motion Aura model is reused across the whole request.
            batch_id = uuid.uuid4().hex[:12]
            items: list[dict[str, Any]] = []
            for candidate_index in range(count):
                candidate_seed = seed + candidate_index
                seed_everything(candidate_seed)
                kwargs: dict[str, Any] = {
                    "constraint_lst": constraints,
                    "num_denoising_steps": diffusion_steps,
                    "num_samples": 1,
                    "multi_prompt": True,
                    "num_transition_frames": 5,
                    "post_processing": False,
                    "return_numpy": True,
                    "cfg_type": cfg_type,
                }
                if cfg_weight is not None:
                    kwargs["cfg_weight"] = cfg_weight
                with torch.inference_mode():
                    output = model(texts, num_frames, **kwargs)
                items.append(self._publish(
                    output,
                    float(model.fps),
                    prompt_label,
                    batch_id=batch_id,
                    candidate_index=candidate_index + 1,
                    candidate_count=count,
                    generation_seed=candidate_seed,
                ))
            return {
                "ok": True,
                # Keep the first motion for backward compatibility with older clients.
                "motion": items[0],
                "motions": items,
                "generation": {
                    "prompt": prompt_label,
                    "example_id": example_id,
                    "duration_s": sum(num_frames) / float(model.fps),
                    "diffusion_steps": diffusion_steps,
                    "seed": seed,
                    "count": count,
                    "batch_id": batch_id,
                    "device": self._device,
                },
            }
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._busy = False
            self._lock.release()
