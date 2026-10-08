"""Aura downstream imitation benchmark.

Compares a preference-ranked subset of saved G1 motions against equally sized random
subsets under the same fixed kinematic behavior-cloning learner and a disjoint held-out
cohort. This is an *offline kinematic imitation proxy*, not physics simulation and not a
real-robot task-success benchmark.

The learner predicts the next frame's root displacement and root-relative joint motion
from the current root-relative pose and previous frame velocity. It is deliberately
small/deterministic (ridge regression) so the result is reproducible and cheap to run.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from aura_g1_metrics import canonical_bytes, sha256
from aura_preference import connect, latest, rank

VERSION = "aura-offline-imitation-v1"
METRIC_KEYS = (
    "one_step_mpjpe_cm",
    "rollout_mpjpe_cm",
    "rollout_root_error_cm",
)


@dataclass(frozen=True)
class MotionSequence:
    motion_id: str
    name: str
    fps: float
    positions: np.ndarray  # (T, 34, 3), Text2Motion Aura Y-up / Z-forward
    native_sha256: str


def _safe_record(base: Path, file: Path) -> dict | None:
    try:
        obj = json.loads(file.read_text())
        if not isinstance(obj, dict) or not obj.get("id") or not obj.get("native_file"):
            return None
        if "g1" not in str(obj.get("model", "")).lower():
            return None
        native = (base / obj["native_file"]).resolve()
        if native.parent != base.resolve() or native.suffix != ".npz" or not native.is_file():
            return None
        return obj
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def motion_records(base: Path) -> list[dict]:
    records: list[dict] = []
    for file in base.glob("*.json"):
        if file.name.endswith(".g1.json"):
            continue
        obj = _safe_record(base, file)
        if obj:
            records.append(obj)
    return records


def list_cohorts(base: Path, *, minimum: int = 2) -> list[dict]:
    """Group G1 generations by exact prompt/name so the benchmark compares one task."""
    groups: dict[str, list[dict]] = {}
    for r in motion_records(base):
        name = str(r.get("name") or "Generated motion").strip()
        groups.setdefault(name, []).append(r)
    out = []
    for name, rows in groups.items():
        if len(rows) < minimum:
            continue
        out.append(
            {
                "name": name,
                "count": len(rows),
                "motion_ids": sorted(str(r["id"]) for r in rows),
                "recommended": len(rows) >= 8,
            }
        )
    out.sort(key=lambda x: (-x["count"], x["name"].lower()))
    return out


def load_sequence(base: Path, record: dict) -> MotionSequence:
    native = (base / record["native_file"]).resolve()
    if native.parent != base.resolve() or native.suffix != ".npz" or not native.is_file():
        raise ValueError(f"Invalid native file for {record.get('id')}")
    with np.load(native, allow_pickle=False) as data:
        if "posed_joints" not in data:
            raise ValueError(f"{record.get('id')} has no posed_joints")
        p = np.asarray(data["posed_joints"], dtype=np.float64)
    fps = float(record.get("fps", 0))
    if p.ndim != 3 or p.shape[1:] != (34, 3) or len(p) < 12 or fps <= 0 or not np.isfinite(p).all():
        raise ValueError(f"{record.get('id')} is not a usable G1 trajectory")
    return MotionSequence(
        motion_id=str(record["id"]),
        name=str(record.get("name") or record["id"]),
        fps=fps,
        positions=p,
        native_sha256=sha256(native.read_bytes()),
    )


def _transition_arrays(seq: MotionSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y and frame indices for one sequence.

    X[t] = [root-relative pose at t, previous root-relative delta, previous root delta]
    Y[t] = [next root-relative delta, next root delta]
    for t=1..T-2.
    """
    p = seq.positions
    root = p[:, 0]
    rel = p[:, 1:] - root[:, None, :]  # (T,33,3) => 99 dims
    drel = np.diff(rel, axis=0)
    droot = np.diff(root, axis=0)
    # index t maps state at frame t to delta t->t+1; previous delta is t-1->t.
    x = np.concatenate(
        [rel[1:-1].reshape(len(p) - 2, -1), drel[:-1].reshape(len(p) - 2, -1), droot[:-1]], axis=1
    )
    y = np.concatenate([drel[1:].reshape(len(p) - 2, -1), droot[1:]], axis=1)
    idx = np.arange(1, len(p) - 1, dtype=int)
    return x, y, idx


def _even_indices(n: int, count: int) -> np.ndarray:
    if count >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.rint(np.linspace(0, n - 1, count)).astype(int))


class RidgeMotionPolicy:
    """Small deterministic behavior-cloning policy over kinematic motion states."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.clip_lo: np.ndarray | None = None
        self.clip_hi: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 4:
            raise ValueError("Not enough transitions to train imitation policy")
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), 1e-5)
        self.y_mean = y.mean(axis=0)
        self.y_scale = np.maximum(y.std(axis=0), 1e-5)
        zx = (x - self.x_mean) / self.x_scale
        zy = (y - self.y_mean) / self.y_scale
        # Add intercept after standardization; regularize weights but not intercept.
        design = np.concatenate([zx, np.ones((len(zx), 1))], axis=1)
        reg = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(design.T @ design + reg, design.T @ zy)
        self.clip_lo = np.quantile(y, 0.005, axis=0)
        self.clip_hi = np.quantile(y, 0.995, axis=0)
        span = np.maximum(self.clip_hi - self.clip_lo, 1e-6)
        self.clip_lo = self.clip_lo - 0.25 * span
        self.clip_hi = self.clip_hi + 0.25 * span

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.x_mean is None or self.x_scale is None or self.y_mean is None or self.y_scale is None:
            raise RuntimeError("Policy is not fitted")
        one = x.ndim == 1
        xx = x[None] if one else x
        zx = (xx - self.x_mean) / self.x_scale
        design = np.concatenate([zx, np.ones((len(zx), 1))], axis=1)
        y = (design @ self.weights) * self.y_scale + self.y_mean
        if self.clip_lo is not None and self.clip_hi is not None:
            y = np.clip(y, self.clip_lo, self.clip_hi)
        return y[0] if one else y


def _training_data(seqs: Iterable[MotionSequence], samples_per_motion: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seq in seqs:
        x, y, _ = _transition_arrays(seq)
        use = _even_indices(len(x), samples_per_motion)
        xs.append(x[use])
        ys.append(y[use])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _next_positions_from_delta(seq: MotionSequence, t: int, delta: np.ndarray) -> np.ndarray:
    root = seq.positions[t, 0]
    rel = seq.positions[t, 1:] - root[None, :]
    drel = delta[:99].reshape(33, 3)
    droot = delta[99:102]
    next_root = root + droot
    next_rel = rel + drel
    return np.concatenate([next_root[None, :], next_root[None, :] + next_rel], axis=0)


def evaluate_policy(policy: RidgeMotionPolicy, test: list[MotionSequence], horizon_s: float = 0.5) -> dict:
    one_step_errors: list[float] = []
    rollout_errors: list[float] = []
    root_errors: list[float] = []
    rollout_windows = 0
    for seq in test:
        x, y, idx = _transition_arrays(seq)
        pred = policy.predict(x)
        for row, t in zip(pred, idx):
            pp = _next_positions_from_delta(seq, int(t), row)
            one_step_errors.append(float(np.linalg.norm(pp - seq.positions[t + 1], axis=1).mean()))

        horizon = max(2, int(round(float(horizon_s) * seq.fps)))
        candidates = np.arange(1, max(1, len(seq.positions) - horizon - 1), dtype=int)
        if len(candidates) == 0:
            continue
        starts = candidates[_even_indices(len(candidates), min(5, len(candidates)))]
        for start in starts:
            root = seq.positions[start, 0].copy()
            rel = (seq.positions[start, 1:] - root[None, :]).copy()
            prev_rel = (seq.positions[start, 1:] - seq.positions[start, 0]) - (
                seq.positions[start - 1, 1:] - seq.positions[start - 1, 0]
            )
            prev_root = seq.positions[start, 0] - seq.positions[start - 1, 0]
            target_t = start + horizon
            for _ in range(horizon):
                state = np.concatenate([rel.reshape(-1), prev_rel.reshape(-1), prev_root])
                dy = policy.predict(state)
                drel = dy[:99].reshape(33, 3)
                droot = dy[99:102]
                root = root + droot
                rel = rel + drel
                prev_rel, prev_root = drel, droot
            pred_pos = np.concatenate([root[None, :], root[None, :] + rel], axis=0)
            gt = seq.positions[target_t]
            rollout_errors.append(float(np.linalg.norm(pred_pos - gt, axis=1).mean()))
            root_errors.append(float(np.linalg.norm(root - gt[0])))
            rollout_windows += 1
    if not one_step_errors or not rollout_errors:
        raise ValueError("Held-out motions are too short for the requested horizon")
    return {
        "one_step_mpjpe_cm": round(float(np.mean(one_step_errors) * 100.0), 4),
        "rollout_mpjpe_cm": round(float(np.mean(rollout_errors) * 100.0), 4),
        "rollout_root_error_cm": round(float(np.mean(root_errors) * 100.0), 4),
        "test_transitions": int(len(one_step_errors)),
        "rollout_windows": int(rollout_windows),
    }


def _fit_and_eval(train: list[MotionSequence], test: list[MotionSequence], samples_per_motion: int, horizon_s: float) -> dict:
    x, y = _training_data(train, samples_per_motion)
    policy = RidgeMotionPolicy(alpha=1.0)
    policy.fit(x, y)
    out = evaluate_policy(policy, test, horizon_s=horizon_s)
    out["training_transitions"] = int(len(x))
    return out


def _summary(values: list[dict]) -> dict:
    result = {}
    for key in METRIC_KEYS:
        arr = np.asarray([v[key] for v in values], dtype=float)
        result[key] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=0)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }
    return result


def _heldout_ids(ids: list[str], count: int, seed: int) -> list[str]:
    ordered = sorted(ids, key=lambda mid: hashlib.sha256(f"{seed}:{mid}".encode()).hexdigest())
    return ordered[:count]


def run_benchmark(
    base: Path,
    db: Path,
    *,
    cohort_name: str,
    train_count: int = 4,
    holdout_count: int = 2,
    random_trials: int = 20,
    samples_per_motion: int = 100,
    horizon_s: float = 0.5,
    seed: int = 37,
) -> dict:
    if not isinstance(cohort_name, str) or not cohort_name.strip():
        raise ValueError("Choose a same-prompt motion cohort")
    cohort_name = cohort_name.strip()
    train_count = int(train_count)
    holdout_count = int(holdout_count)
    random_trials = int(random_trials)
    samples_per_motion = int(samples_per_motion)
    seed = int(seed)
    horizon_s = float(horizon_s)
    if not 2 <= train_count <= 12:
        raise ValueError("train_count must be between 2 and 12")
    if not 1 <= holdout_count <= 6:
        raise ValueError("holdout_count must be between 1 and 6")
    if not 3 <= random_trials <= 50:
        raise ValueError("random_trials must be between 3 and 50")
    if not 20 <= samples_per_motion <= 200:
        raise ValueError("samples_per_motion must be between 20 and 200")
    if not 0.2 <= horizon_s <= 1.5:
        raise ValueError("horizon_s must be between 0.2 and 1.5 seconds")

    records = [r for r in motion_records(base) if str(r.get("name") or "").strip() == cohort_name]
    if len(records) < train_count + holdout_count + 1:
        raise ValueError(
            f"Need at least {train_count + holdout_count + 1} G1 motions with exactly this prompt "
            "so Aura and random selection are not forced to use the same training set"
        )
    seqs = {str(r["id"]): load_sequence(base, r) for r in records}
    usable_ids = sorted(seqs)

    with connect(db) as conn:
        model = latest(conn)
        if not model:
            raise ValueError("Train the human preference model before running the downstream benchmark")
        ranked = rank(conn, base)["rankings"]
    rank_signal = {r["id"]: float(r["ranking_signal"]) for r in ranked}
    missing = [mid for mid in usable_ids if mid not in rank_signal]
    if missing:
        raise ValueError("Some cohort motions are missing from the learned ranking; refresh screening and retrain")

    seen_ids = set(model.get('source_motion_ids') or [])
    if not seen_ids:
        raise ValueError('Retrain the preference model with Aura pairwise model v2 before benchmarking unseen motions')
    unseen_ids = [mid for mid in usable_ids if mid not in seen_ids]
    if len(unseen_ids) < holdout_count:
        raise ValueError(
            f'Need {holdout_count} fresh same-prompt motions that were generated after preference training and never compared. '
            f'Only {len(unseen_ids)} are currently unseen by the selector.'
        )
    holdout = _heldout_ids(unseen_ids, holdout_count, seed)
    train_pool = [mid for mid in usable_ids if mid not in holdout]
    aura_ids = sorted(train_pool, key=lambda mid: rank_signal[mid], reverse=True)[:train_count]
    if len(train_pool) <= train_count:
        raise ValueError("Need at least one extra training-pool motion for a meaningful random baseline")

    # Equalize frame budget across every possible training motion in this cohort.
    available = [len(_transition_arrays(seqs[mid])[0]) for mid in train_pool]
    effective_samples = min(samples_per_motion, min(available))
    if effective_samples < 20:
        raise ValueError("Motions are too short for the downstream benchmark")
    test_seqs = [seqs[mid] for mid in holdout]
    aura_result = _fit_and_eval([seqs[mid] for mid in aura_ids], test_seqs, effective_samples, horizon_s)

    combinations = list(itertools.combinations(train_pool, train_count))
    rng = np.random.default_rng(seed)
    if len(combinations) > random_trials:
        choice = rng.choice(len(combinations), size=random_trials, replace=False)
        subsets = [combinations[int(i)] for i in choice]
    else:
        subsets = combinations
    random_results = [
        _fit_and_eval([seqs[mid] for mid in subset], test_seqs, effective_samples, horizon_s) for subset in subsets
    ]
    random_summary = _summary(random_results)
    comparison = {}
    for key in METRIC_KEYS:
        baseline = float(random_summary[key]["mean"])
        aura = float(aura_result[key])
        improvement = ((baseline - aura) / baseline * 100.0) if baseline > 1e-12 else 0.0
        percentile = float(np.mean([r[key] >= aura for r in random_results]) * 100.0)
        comparison[key] = {
            "aura": round(aura, 4),
            "random_mean": round(baseline, 4),
            "relative_improvement_pct": round(improvement, 2),
            "aura_beats_random_trials_pct": round(percentile, 1),
            "lower_is_better": True,
        }

    sources = {mid: seqs[mid].native_sha256 for mid in usable_ids}
    core = {
        "version": VERSION,
        "selection_model_version": model.get("version"),
        "selection_model_dataset_sha256": model.get("dataset_sha256"),
        "cohort_name": cohort_name,
        "cohort_count": len(usable_ids),
        "train_count": train_count,
        "holdout_count": holdout_count,
        "random_trials": len(random_results),
        "samples_per_motion": effective_samples,
        "horizon_s": horizon_s,
        "seed": seed,
        "aura_selected_ids": aura_ids,
        "heldout_ids": holdout,
        "selector_seen_motion_ids": sorted(seen_ids),
        "downstream_holdout_is_selector_unseen": True,
        "aura_result": aura_result,
        "random_summary": random_summary,
        "comparison": comparison,
        "source_motion_sha256": sources,
        "metric_definition": {
            "one_step_mpjpe_cm": "Mean per-joint position error for one-step held-out motion prediction",
            "rollout_mpjpe_cm": f"Mean per-joint position error after an autoregressive {horizon_s:.2f}s rollout",
            "rollout_root_error_cm": f"Pelvis/root position error after an autoregressive {horizon_s:.2f}s rollout",
        },
        "scope": "Offline kinematic behavior-cloning proxy on held-out G1 trajectories; not physics simulation, balance validation, or real-robot success.",
        "interpretation": "If Aura-selected training motions outperform the repeated random baseline under the same learner and transition budget, that is evidence that learned selection helps this specific imitation proxy. A negative result is retained as-is.",
    }
    result = {**core, "created_at": int(time.time())}
    result["report_sha256"] = sha256(canonical_bytes(core))
    outdir = base / "benchmarks"
    outdir.mkdir(exist_ok=True)
    filename = f"{result['created_at']}-{result['report_sha256'][:12]}.json"
    (outdir / filename).write_text(json.dumps(result, indent=2))
    (outdir / "latest.json").write_text(json.dumps(result, indent=2))
    return result


def latest_benchmark(base: Path) -> dict | None:
    path = base / "benchmarks" / "latest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
