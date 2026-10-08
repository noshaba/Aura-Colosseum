"""Aura AMP-inspired learned motion prior.

This is intentionally NOT full Adversarial Motion Priors (AMP): no policy is
trained in a physics simulator and no physical-validity claim is made.

Instead, Aura learns a small discriminator over short G1 state-transition
windows. Human pairwise preferences provide weak acceptance targets. The model
therefore acts as a learned trajectory prior / reward-model signal alongside
hard kinematic constraints and the pairwise preference ranker.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from aura_g1_metrics import canonical_bytes, sha256
from aura_preference import get_motion, list_preferences

VERSION = "aura-amp-inspired-prior-v1"
MODEL_BASENAME = "aura-motion-prior-v1"
MAX_WINDOWS_PER_MOTION = 96
HIDDEN_1 = 64
HIDDEN_2 = 24


def _paths(base: Path):
    return base / f"{MODEL_BASENAME}.npz", base / f"{MODEL_BASENAME}.json"


def _trajectory(base: Path, motion_id: str):
    motion = get_motion(base, motion_id)
    native = (base / motion["native_file"]).resolve()
    if native.parent != base.resolve() or native.suffix != ".npz" or not native.is_file():
        raise ValueError("No usable G1 NPZ")
    with np.load(native, allow_pickle=False) as data:
        p = np.asarray(data["posed_joints"], dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or p.shape[0] < 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite G1 trajectory shaped (T,34,3)")
    return motion, p, float(motion.get("fps") or 30.0)


def transition_windows(positions: np.ndarray, fps: float, max_windows: int = MAX_WINDOWS_PER_MOTION):
    """AMP-style local state-transition descriptors from world-space G1 joints.

    Features deliberately avoid the existing handcrafted screening metrics:
    root-relative pose, joint velocities, root velocity and root height are used
    directly. One row represents one local transition.
    """
    p = np.asarray(positions, dtype=np.float64)
    fps = float(fps)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or p.shape[0] < 3 or fps <= 0:
        raise ValueError("Expected at least three G1 frames and fps > 0")
    root = p[:, 0]
    relative = p - root[:, None, :]
    velocity = np.diff(p, axis=0) * fps
    root_velocity = np.diff(root, axis=0) * fps
    # Descriptor for state_t -> state_t+1.
    x = np.concatenate([
        relative[:-1].reshape(len(p) - 1, -1),
        velocity.reshape(len(p) - 1, -1),
        root_velocity,
        root[:-1, 1:2],
    ], axis=1)
    if len(x) > max_windows:
        idx = np.linspace(0, len(x) - 1, max_windows).round().astype(int)
        x = x[idx]
    return x.astype(np.float64, copy=False)


def _labels(preferences: list[dict[str, Any]]):
    stats: dict[str, list[int]] = {}
    for r in preferences:
        left, right, winner = r["left_id"], r["right_id"], r["winner_id"]
        stats.setdefault(left, [0, 0])
        stats.setdefault(right, [0, 0])
        loser = right if winner == left else left
        stats[winner][0] += 1
        stats[loser][1] += 1
    # Beta(1,1) smoothing: one vote never becomes a 0/1 target.
    return {mid: (wins + 1.0) / (wins + losses + 2.0) for mid, (wins, losses) in stats.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _bce(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return float(np.mean(-(target * np.log(pred) + (1 - target) * np.log(1 - pred))))


def _forward(x, w1, b1, w2, b2, w3, b3):
    z1 = x @ w1 + b1
    h1 = np.maximum(z1, 0)
    z2 = h1 @ w2 + b2
    h2 = np.maximum(z2, 0)
    logits = (h2 @ w3 + b3).reshape(-1)
    return z1, h1, z2, h2, logits, _sigmoid(logits)


def train_prior(conn, base: Path, epochs: int = 260):
    prefs = list(reversed(list_preferences(conn)))
    if len(prefs) < 6:
        raise ValueError("Collect at least six generated-motion comparisons before training the motion prior")
    targets = _labels(prefs)
    if len(targets) < 4:
        raise ValueError("The motion prior needs preferences covering at least four distinct generated motions")

    windows: dict[str, np.ndarray] = {}
    usable_ids = []
    for mid in sorted(targets):
        try:
            motion, p, fps = _trajectory(base, mid)
            windows[mid] = transition_windows(p, fps)
            usable_ids.append(mid)
        except (ValueError, KeyError, OSError):
            continue
    if len(usable_ids) < 4:
        raise ValueError("At least four unchanged G1 NPZ motions are required")

    # Motion-disjoint validation when the data is large enough. This avoids
    # reporting frame-level leakage as generalization.
    rng = np.random.default_rng(73)
    ids = usable_ids[:]
    rng.shuffle(ids)
    heldout = len(ids) >= 6 and len(prefs) >= 10
    if heldout:
        n_hold = max(1, len(ids) // 4)
        val_ids = sorted(ids[-n_hold:])
        train_ids = sorted(ids[:-n_hold])
    else:
        train_ids = sorted(ids)
        val_ids = []

    x_train = np.concatenate([windows[mid] for mid in train_ids], axis=0)
    y_train = np.concatenate([np.full(len(windows[mid]), targets[mid]) for mid in train_ids]).astype(np.float64)
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-3)
    x = np.clip((x_train - mean) / std, -8, 8)

    in_dim = x.shape[1]
    rng = np.random.default_rng(17)
    w1 = rng.normal(0, math.sqrt(2 / in_dim), (in_dim, HIDDEN_1))
    b1 = np.zeros(HIDDEN_1)
    w2 = rng.normal(0, math.sqrt(2 / HIDDEN_1), (HIDDEN_1, HIDDEN_2))
    b2 = np.zeros(HIDDEN_2)
    w3 = rng.normal(0, 0.08, (HIDDEN_2, 1))
    b3 = np.zeros(1)

    lr = 0.018
    reg = 2e-4
    for step in range(max(40, int(epochs))):
        z1, h1, z2, h2, logits, pred = _forward(x, w1, b1, w2, b2, w3, b3)
        n = len(x)
        dlogit = ((pred - y_train) / n)[:, None]
        dw3 = h2.T @ dlogit + reg * w3
        db3 = dlogit.sum(axis=0)
        dh2 = dlogit @ w3.T
        dz2 = dh2 * (z2 > 0)
        dw2 = h1.T @ dz2 + reg * w2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ w2.T
        dz1 = dh1 * (z1 > 0)
        dw1 = x.T @ dz1 + reg * w1
        db1 = dz1.sum(axis=0)
        # Mild decay improves stability on tiny preference datasets.
        eta = lr * (0.35 + 0.65 * (1 - step / max(1, epochs)))
        w3 -= eta * dw3; b3 -= eta * db3
        w2 -= eta * dw2; b2 -= eta * db2
        w1 -= eta * dw1; b1 -= eta * db1

    _, _, _, _, _, train_pred = _forward(x, w1, b1, w2, b2, w3, b3)
    train_loss = _bce(train_pred, y_train)

    val_loss = None
    val_accuracy = None
    val_motion_predictions = []
    if val_ids:
        for mid in val_ids:
            xv = np.clip((windows[mid] - mean) / std, -8, 8)
            *_, pv = _forward(xv, w1, b1, w2, b2, w3, b3)
            score = float(pv.mean())
            val_motion_predictions.append((mid, score, float(targets[mid])))
        val_loss = _bce(np.array([x[1] for x in val_motion_predictions]), np.array([x[2] for x in val_motion_predictions]))
        val_accuracy = float(np.mean([(score >= .5) == (target >= .5) for _, score, target in val_motion_predictions]))

    model_path, meta_path = _paths(base)
    np.savez_compressed(model_path, mean=mean, std=std, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    model_sha = sha256(model_path.read_bytes())
    result = {
        "version": VERSION,
        "kind": "AMP-inspired trajectory-window discriminator",
        "trained_at": int(time.time()),
        "vote_count": len(prefs),
        "motion_count": len(usable_ids),
        "training_motion_ids": train_ids,
        "validation_motion_ids": val_ids,
        "train_window_count": int(len(x_train)),
        "train_log_loss": round(train_loss, 5),
        "heldout_motion_accuracy": round(val_accuracy, 4) if val_accuracy is not None else None,
        "heldout_motion_log_loss": round(val_loss, 5) if val_loss is not None else None,
        "dataset_sha256": sha256(canonical_bytes(prefs)),
        "model_sha256": model_sha,
        "label_method": "smoothed per-motion win rate from real generated-motion pairwise preferences",
        "feature_method": "root-relative G1 joint pose + joint velocity + root velocity + root height over local transitions",
        "caveat": "Learned preference-derived motion prior only. It is not full AMP, physics simulation, dynamic stability, task success, or real-robot safety validation.",
    }
    meta_path.write_text(json.dumps(result, indent=2))
    return {"model": result, "scores": score_all(base, metadata=result)}


def _load(base: Path):
    model_path, meta_path = _paths(base)
    if not model_path.is_file() or not meta_path.is_file():
        return None, None
    meta = json.loads(meta_path.read_text())
    if meta.get("version") != VERSION or meta.get("model_sha256") != sha256(model_path.read_bytes()):
        raise ValueError("Motion-prior model metadata/hash mismatch")
    with np.load(model_path, allow_pickle=False) as d:
        arrays = {k: np.asarray(d[k], dtype=np.float64) for k in ("mean", "std", "w1", "b1", "w2", "b2", "w3", "b3")}
    return meta, arrays


def score_motion(base: Path, motion_id: str, arrays=None):
    if arrays is None:
        _, arrays = _load(base)
    if arrays is None:
        return None
    motion, p, fps = _trajectory(base, motion_id)
    x = transition_windows(p, fps)
    x = np.clip((x - arrays["mean"]) / arrays["std"], -8, 8)
    *_, pred = _forward(x, arrays["w1"], arrays["b1"], arrays["w2"], arrays["b2"], arrays["w3"], arrays["b3"])
    return float(pred.mean())


def score_all(base: Path, metadata=None):
    meta, arrays = _load(base)
    if arrays is None:
        return []
    out = []
    for path in sorted(base.glob("*.json")):
        if path.name.endswith(".g1.json") or path.name == f"{MODEL_BASENAME}.json":
            continue
        try:
            item = json.loads(path.read_text())
            if not isinstance(item, dict) or "g1" not in str(item.get("model", "")).lower() or not item.get("native_file"):
                continue
            score = score_motion(base, str(item["id"]), arrays)
            out.append({"id": item["id"], "name": item.get("name", item["id"]), "prior_score": round(float(score), 4)})
        except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError):
            continue
    out.sort(key=lambda x: x["prior_score"], reverse=True)
    return out


def status(base: Path):
    try:
        meta, _ = _load(base)
        return {"model": meta, "scores": score_all(base, metadata=meta) if meta else []}
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return {"model": None, "scores": [], "error": str(exc)}
