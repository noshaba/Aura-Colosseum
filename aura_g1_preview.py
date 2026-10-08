"""G1 native browser preview payload from Text2Motion Aura world-space poses."""
import json
import numpy as np

G1_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32]

def g1_preview_bytes(positions, fps, global_rot_mats=None):
    arr = np.asarray(positions, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (34, 3):
        raise ValueError(f"Expected G1 joints with shape (frames,34,3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Motion contains non-finite joint positions")
    payload = {
        "format": "g1-joints-v1",
        "fps": float(fps),
        "parents": G1_PARENTS,
        "positions": np.round(arr, 5).tolist(),
    }
    if global_rot_mats is not None:
        rot = np.asarray(global_rot_mats, dtype=np.float32)
        if rot.shape != (arr.shape[0], 34, 3, 3):
            raise ValueError(f"Expected G1 global rotations with shape (frames,34,3,3), got {rot.shape}")
        if not np.isfinite(rot).all():
            raise ValueError("Motion contains non-finite joint rotations")
        payload["format"] = "g1-joints-v2"
        payload["global_rot_mats"] = np.round(rot, 6).tolist()
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
