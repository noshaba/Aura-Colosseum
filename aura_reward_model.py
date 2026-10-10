"""Aura Motion Reward Model.

A small trajectory reward model trained directly from human pairwise preferences.
This is the reward-modeling component commonly used inside RLHF systems, but it
is deliberately *not* described as full RLHF because Aura does not currently
optimize the Text2Motion generator or a robot policy against this reward with
reinforcement learning.

The model consumes robot-native G1 trajectories rather than handcrafted summary
metrics. It encodes a fixed-length sequence of root-relative joint positions,
joint velocities, root velocity/height and (when available) 6D global joint
rotations with a compact temporal Transformer. A scalar reward is trained using
Bradley-Terry pairwise preference loss: P(A > B) = sigmoid(r(A) - r(B)).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from aura_g1_metrics import canonical_bytes, sha256
from aura_preference import get_motion, validated_preferences

VERSION = "aura-motion-reward-transformer-v1"
MODEL_BASENAME = "aura-motion-reward-v1"
SEQ_LEN = 64
D_MODEL = 96
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 192
DROPOUT = 0.10
DEFAULT_EPOCHS = 40
MIN_COMPARISONS = 6
MIN_MOTIONS = 4


def _torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on install state
        raise RuntimeError("Aura reward model requires PyTorch. Install the project's documented PyTorch build first.") from exc
    return torch, nn


def _paths(base: Path):
    return base / f"{MODEL_BASENAME}.pt", base / f"{MODEL_BASENAME}.json"


def _load_native(base: Path, motion_id: str):
    motion = get_motion(base, motion_id)
    native = (base / motion["native_file"]).resolve()
    if native.parent != base.resolve() or native.suffix != ".npz" or not native.is_file():
        raise ValueError("No usable G1 NPZ")
    with np.load(native, allow_pickle=False) as data:
        positions = np.asarray(data["posed_joints"], dtype=np.float32)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float32) if "global_rot_mats" in data else None
    if positions.ndim != 3 or positions.shape[1:] != (34, 3) or positions.shape[0] < 3 or not np.isfinite(positions).all():
        raise ValueError("Expected finite G1 posed_joints shaped (T,34,3)")
    if rotations is not None:
        if rotations.shape != (positions.shape[0], 34, 3, 3) or not np.isfinite(rotations).all():
            rotations = None
    return motion, positions, rotations, float(motion.get("fps") or 30.0)


def _resample_frames(x: np.ndarray, length: int = SEQ_LEN):
    if len(x) == length:
        return x
    idx = np.linspace(0, len(x) - 1, length).round().astype(np.int64)
    return x[idx]


def trajectory_sequence(positions: np.ndarray, fps: float, rotations: np.ndarray | None = None, seq_len: int = SEQ_LEN):
    """Build per-frame neural features directly from a G1 trajectory.

    The feature vector intentionally does not reuse Aura's handcrafted screening
    metrics. It contains local pose/dynamics information from the full motion.
    """
    p = np.asarray(positions, dtype=np.float32)
    fps = float(fps)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or p.shape[0] < 3 or fps <= 0:
        raise ValueError("Expected at least three G1 frames and fps > 0")
    root = p[:, 0]
    relative = p - root[:, None, :]
    velocity = np.zeros_like(p)
    velocity[1:] = np.diff(p, axis=0) * fps
    root_velocity = np.zeros_like(root)
    root_velocity[1:] = np.diff(root, axis=0) * fps
    root_height = root[:, 1:2]

    # Continuous 6D rotation representation: first two columns of each matrix.
    # If older NPZ files do not have rotations we keep a presence bit and zeros.
    if rotations is not None:
        r = np.asarray(rotations, dtype=np.float32)
        rot6d = r[:, :, :, :2].reshape(len(p), -1)
        rotation_present = np.ones((len(p), 1), dtype=np.float32)
    else:
        rot6d = np.zeros((len(p), 34 * 6), dtype=np.float32)
        rotation_present = np.zeros((len(p), 1), dtype=np.float32)

    x = np.concatenate([
        relative.reshape(len(p), -1),
        velocity.reshape(len(p), -1),
        root_velocity,
        root_height,
        rot6d,
        rotation_present,
    ], axis=1)
    x = _resample_frames(x, seq_len)
    if not np.isfinite(x).all():
        raise ValueError("Motion produced non-finite reward-model features")
    return x.astype(np.float32, copy=False)


def _motion_sequence(base: Path, motion_id: str):
    _motion, p, r, fps = _load_native(base, motion_id)
    return trajectory_sequence(p, fps, r)


def _make_model(input_dim: int):
    torch, nn = _torch()

    class AuraMotionRewardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, D_MODEL),
                nn.LayerNorm(D_MODEL),
                nn.GELU(),
            )
            self.position = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS, norm=nn.LayerNorm(D_MODEL))
            self.reward_head = nn.Sequential(
                nn.Linear(D_MODEL, 64),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(64, 1),
            )
            nn.init.normal_(self.position, std=0.01)

        def forward(self, x):
            h = self.input_proj(x) + self.position[:, :x.shape[1]]
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.reward_head(pooled).squeeze(-1)

    return AuraMotionRewardNet()


def _pair_split(preferences: list[dict[str, Any]]):
    """Deterministic unordered-pair split; repeated A/B votes never leak across sets."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(preferences):
        key = r.get("pair_key") or tuple(sorted((r["left_id"], r["right_id"])))
        groups.setdefault(key, []).append(i)
    can_holdout = len(preferences) >= 10 and len(groups) >= 4
    if not can_holdout:
        return list(range(len(preferences))), []
    keys = sorted(groups)
    rng = np.random.default_rng(117)
    rng.shuffle(keys)
    hold_count = max(1, len(keys) // 5)
    val_keys = set(keys[-hold_count:])
    val = [i for k in keys if k in val_keys for i in groups[k]]
    train = [i for k in keys if k not in val_keys for i in groups[k]]
    if len(train) < MIN_COMPARISONS:
        return list(range(len(preferences))), []
    return train, val


def _valid_preferences(conn, base: Path):
    valid, diagnostics = validated_preferences(conn, base)
    sequences: dict[str, np.ndarray] = {}
    usable = []
    sequence_rejections = 0
    for r in valid:
        try:
            for mid in (r["left_id"], r["right_id"]):
                if mid not in sequences:
                    sequences[mid] = _motion_sequence(base, mid)
            usable.append(r)
        except (ValueError, KeyError, OSError, TypeError):
            sequence_rejections += 1
    if sequence_rejections:
        diagnostics = dict(diagnostics)
        diagnostics["rejected_comparison_count"] += sequence_rejections
        diagnostics["valid_comparison_count"] = len(usable)
        diagnostics["reward_trainable"] = len(usable) >= MIN_COMPARISONS and diagnostics["unique_motion_count"] >= MIN_MOTIONS
        reasons = dict(diagnostics.get("rejection_reasons") or {})
        reasons["trajectory_encoding_failed"] = reasons.get("trajectory_encoding_failed", 0) + sequence_rejections
        diagnostics["rejection_reasons"] = reasons
    return usable, sequences, diagnostics


def _normalization(sequences: dict[str, np.ndarray], motion_ids: set[str]):
    frames = np.concatenate([sequences[mid] for mid in sorted(motion_ids)], axis=0)
    mean = frames.mean(axis=0).astype(np.float32)
    std = np.maximum(frames.std(axis=0), 1e-3).astype(np.float32)
    return mean, std


def _tensor_batch(torch, sequences, rows, indices, mean, std, device):
    left = np.stack([sequences[rows[i]["left_id"]] for i in indices])
    right = np.stack([sequences[rows[i]["right_id"]] for i in indices])
    y = np.asarray([1.0 if rows[i]["winner_id"] == rows[i]["left_id"] else 0.0 for i in indices], dtype=np.float32)
    left = np.clip((left - mean) / std, -8.0, 8.0)
    right = np.clip((right - mean) / std, -8.0, 8.0)
    return (
        torch.from_numpy(left).to(device),
        torch.from_numpy(right).to(device),
        torch.from_numpy(y).to(device),
    )


def _evaluate(model, torch, sequences, rows, indices, mean, std, device):
    if not indices:
        return None, None
    model.eval()
    with torch.no_grad():
        left, right, y = _tensor_batch(torch, sequences, rows, indices, mean, std, device)
        logits = model(left) - model(right)
        probs = torch.sigmoid(logits)
        eps = 1e-7
        loss = -(y * torch.log(probs.clamp(eps, 1 - eps)) + (1 - y) * torch.log((1 - probs).clamp(eps, 1 - eps))).mean()
        acc = ((probs >= 0.5) == (y >= 0.5)).float().mean()
    return float(acc.cpu()), float(loss.cpu())


def train_reward_model(conn, base: Path, epochs: int = DEFAULT_EPOCHS):
    torch, nn = _torch()
    valid, sequences, diagnostics = _valid_preferences(conn, base)
    if len(valid) < MIN_COMPARISONS:
        raise ValueError(
            f"Aura has {len(valid)} valid unique comparisons of {MIN_COMPARISONS} required "
            f"({diagnostics['stored_comparison_count']} stored, {diagnostics['duplicate_comparison_count']} duplicates, "
            f"{diagnostics['rejected_comparison_count']} rejected). Generate fresh candidates if duplicates are being rejected."
        )
    motion_ids = sorted({mid for r in valid for mid in (r["left_id"], r["right_id"])})
    if diagnostics["unique_motion_count"] < MIN_MOTIONS:
        raise ValueError(
            f"Aura has {diagnostics['unique_motion_count']} distinct valid generated motion hashes of {MIN_MOTIONS} required. "
            f"Generate a fresh batch; repeated file hashes do not count twice."
        )

    train_idx, val_idx = _pair_split(valid)
    train_motion_ids = {mid for i in train_idx for mid in (valid[i]["left_id"], valid[i]["right_id"])}
    mean, std = _normalization(sequences, train_motion_ids)
    input_dim = int(next(iter(sequences.values())).shape[1])

    # Reward-model training is intentionally CPU-first so it does not compete
    # with NVIDIA Kimodo for VRAM on 16 GB cards.
    device = torch.device("cpu")
    torch.manual_seed(29)
    np.random.seed(29)
    try:
        torch.set_num_threads(max(1, min(4, int(torch.get_num_threads()))))
    except Exception:
        pass
    model = _make_model(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(29)
    epochs = max(4, min(400, int(epochs)))
    batch_size = min(8, len(train_idx))

    model.train()
    for epoch in range(epochs):
        order = np.asarray(train_idx, dtype=np.int64).copy()
        rng.shuffle(order)
        # Gentle LR decay for tiny preference datasets.
        lr_scale = 0.25 + 0.75 * (1.0 - epoch / max(1, epochs - 1))
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * lr_scale
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size].tolist()
            left, right, y = _tensor_batch(torch, sequences, valid, batch, mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(left) - model(right)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_acc, train_loss = _evaluate(model, torch, sequences, valid, train_idx, mean, std, device)
    val_acc, val_loss = _evaluate(model, torch, sequences, valid, val_idx, mean, std, device)

    model_path, meta_path = _paths(base)
    state = {
        "version": VERSION,
        "input_dim": input_dim,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(state, model_path)
    model_sha = sha256(model_path.read_bytes())
    param_count = int(sum(p.numel() for p in model.parameters()))
    result = {
        "version": VERSION,
        "kind": "pairwise temporal Transformer motion reward model",
        "training_paradigm": "human-preference reward modeling (RLHF component; not full RLHF)",
        "trained_at": int(time.time()),
        "training_count": len(train_idx),
        "total_comparison_count": len(valid),
        "stored_comparison_count": diagnostics["stored_comparison_count"],
        "unique_pair_count": diagnostics["unique_pair_count"],
        "duplicate_comparison_count": diagnostics["duplicate_comparison_count"],
        "rejected_comparison_count": diagnostics["rejected_comparison_count"],
        "motion_count": diagnostics["unique_motion_count"],
        "source_motion_id_count": len(motion_ids),
        "parameter_count": param_count,
        "sequence_length": SEQ_LEN,
        "input_dim": input_dim,
        "architecture": {"d_model": D_MODEL, "heads": NHEAD, "layers": NUM_LAYERS, "feedforward": DIM_FEEDFORWARD},
        "loss": "Bradley-Terry pairwise preference loss: BCEWithLogits(r(left)-r(right), human_choice)",
        "train_pair_accuracy": round(float(train_acc), 4) if train_acc is not None else None,
        "train_pair_log_loss": round(float(train_loss), 5) if train_loss is not None else None,
        "heldout_pair_accuracy": round(float(val_acc), 4) if val_acc is not None else None,
        "heldout_pair_log_loss": round(float(val_loss), 5) if val_loss is not None else None,
        "heldout_pair_count": len(val_idx),
        "source_motion_ids": motion_ids,
        "dataset_sha256": sha256(canonical_bytes(valid)),
        "model_sha256": model_sha,
        "feature_method": "root-relative G1 joint positions + joint velocities + root velocity + root height + optional global joint rotations in 6D representation",
        "caveat": "Aura learns a human-preference reward over saved G1 trajectories. It is not physical safety validation, simulator task success, or full RLHF because no generator/policy is optimized with reinforcement learning against this reward.",
    }
    meta_path.write_text(json.dumps(result, indent=2))
    return {"model": result, "scores": score_all(base)}


def _load(base: Path):
    torch, _ = _torch()
    model_path, meta_path = _paths(base)
    if not model_path.is_file() or not meta_path.is_file():
        return None, None, None, None
    meta = json.loads(meta_path.read_text())
    if meta.get("version") != VERSION or meta.get("model_sha256") != sha256(model_path.read_bytes()):
        raise ValueError("Aura reward-model metadata/hash mismatch")
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older supported torch builds.
        payload = torch.load(model_path, map_location="cpu")
    if payload.get("version") != VERSION:
        raise ValueError("Aura reward-model checkpoint version mismatch")
    input_dim = int(payload["input_dim"])
    model = _make_model(input_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = payload["mean"].detach().cpu().numpy().astype(np.float32)
    std = payload["std"].detach().cpu().numpy().astype(np.float32)
    return meta, model, mean, std


def score_motion(base: Path, motion_id: str, loaded=None):
    torch, _ = _torch()
    if loaded is None:
        loaded = _load(base)
    meta, model, mean, std = loaded
    if model is None:
        return None
    seq = _motion_sequence(base, motion_id)
    x = np.clip((seq - mean) / std, -8.0, 8.0)[None, ...]
    with torch.no_grad():
        reward = float(model(torch.from_numpy(x)).item())
    return reward


def score_all(base: Path):
    loaded = _load(base)
    meta, model, _mean, _std = loaded
    if model is None:
        return []
    rows = []
    for path in sorted(base.glob("*.json")):
        if path.name.endswith(".g1.json") or path.name == f"{MODEL_BASENAME}.json" or path.name.startswith("aura-motion-prior"):
            continue
        try:
            item = json.loads(path.read_text())
            if not isinstance(item, dict) or not item.get("id") or "g1" not in str(item.get("model", "")).lower() or not item.get("native_file"):
                continue
            if item.get("source") == "starter_reference":
                continue
            reward = score_motion(base, str(item["id"]), loaded)
            if reward is not None:
                rows.append({"id": item["id"], "name": item.get("name", item["id"]), "reward": round(float(reward), 5)})
        except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda x: x["reward"], reverse=True)
    n = len(rows)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        # Rank percentile is display-only; it is explicitly not a calibrated probability.
        row["rank_percentile"] = round(1.0 if n <= 1 else 1.0 - (rank - 1) / (n - 1), 4)
    return rows


def pair_probability(base: Path, left_id: str, right_id: str):
    loaded = _load(base)
    if loaded[1] is None:
        return None
    a = score_motion(base, left_id, loaded)
    b = score_motion(base, right_id, loaded)
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, a - b)))))


def status(base: Path):
    try:
        meta, model, _mean, _std = _load(base)
        return {"model": meta, "scores": score_all(base) if model is not None else []}
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {"model": None, "scores": [], "error": str(exc)}
