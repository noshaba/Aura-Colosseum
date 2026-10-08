"""Publish generated motions to Aura's local, file-backed motion library."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def library_dir() -> Path:
    path = Path(os.environ.get("AURA_MOTION_LIBRARY", str(Path(__file__).resolve().parents[3] / "aura-motion-library")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish_motion(session, motion, *, prompt: str = "", sample: str = "generated") -> dict:
    """Save native NPZ with an appropriate skeleton preview for its model."""
    from text2motion_aura.exports.bvh import save_motion_bvh
    from text2motion_aura.exports.motion_io import save_kimodo_npz
    base = library_dir()
    uid = uuid.uuid4().hex[:12]
    native_path = base / f"{uid}.npz"
    bvh_path = base / f"{uid}.bvh"
    joints_pos = motion.joints_pos.detach().cpu().numpy()
    data = {
        "posed_joints": joints_pos,
        "global_rot_mats": motion.joints_rot.detach().cpu().numpy(),
        "local_rot_mats": motion.joints_local_rot.detach().cpu().numpy(),
        "root_positions": joints_pos[:, session.skeleton.root_idx, :],
    }
    if motion.foot_contacts is not None:
        data["foot_contacts"] = motion.foot_contacts.detach().cpu().numpy()
    save_kimodo_npz(str(native_path), data)
    preview_error = None
    preview_path = bvh_path
    if session.skeleton.name == "g1skel34":
        # G1 has no SOMA global_rot_offsets or BVH rest-pose data.
        # Use actual generated world joint positions instead of an invalid BVH.
        import numpy as np
        from text2motion_aura.demo.aura_g1_preview import g1_preview_bytes
        preview_path = base / f"{uid}.g1.json"
        try:
            preview_path.write_bytes(g1_preview_bytes(joints_pos, session.model_fps, motion.joints_rot.detach().cpu().numpy()))
        except Exception as exc:
            preview_error = str(exc)
            preview_path.unlink(missing_ok=True)
    else:
      try:
        save_motion_bvh(
            str(bvh_path), motion.joints_local_rot,
            motion.joints_pos[:, session.skeleton.root_idx, :],
            skeleton=session.skeleton, fps=float(session.model_fps), standard_tpose=False,
        )
      except Exception as exc:
        preview_error = str(exc)
        bvh_path.unlink(missing_ok=True)
    item = {
        "id": uid, "name": (prompt or "Generated motion").strip()[:100],
        "model": session.model_name, "sample": sample,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(joints_pos), "fps": float(session.model_fps),
        "native_file": native_path.name,
        "preview_file": preview_path.name if preview_path.exists() else None,
        "preview_error": preview_error,
        "evaluation_status": "awaiting_review",
    }
    metadata = base / f"{uid}.json"
    tmp = base / f"{uid}.json.tmp"
    tmp.write_text(json.dumps(item, indent=2))
    tmp.replace(metadata)
    return item
