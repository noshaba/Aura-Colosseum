"""Aura native G1 constraint screening and candidate selection.

No BioIK / AI4Animation dependency. This module evaluates explicit, user-selected
kinematic requirements on NVIDIA Kimodo G1 trajectories and uses Aura's learned human
preference signal only as a tie-breaker. It does NOT solve inverse kinematics,
run physics, or claim real-robot safety.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aura_g1_metrics import canonical_bytes, report_for_npz, sha256
from aura_preference import connect, latest, rank

VERSION = "aura-native-g1-constraints-v1"

# NVIDIA Kimodo G1Skeleton34 indices.
PELVIS = 0
LEFT_TOE = 7
RIGHT_TOE = 14
WAIST_PITCH = 17
LEFT_HAND = 25
RIGHT_HAND = 33

PRESETS: dict[str, dict[str, Any]] = {
    "balanced_forward_walk": {
        "label": "Balanced forward walk",
        "description": "Forward progress with limited lateral drift, torso lean, foot slide and vertical bobbing.",
        "constraints": {
            "min_forward_progress_m": 0.75,
            "max_lateral_drift_m": 0.65,
            "min_path_efficiency": 0.65,
            "max_root_height_range_m": 0.22,
            "max_near_floor_toe_speed_m_s": 0.35,
            "max_below_floor_fraction": 0.08,
            "max_torso_lean_deg": 28.0,
            "max_end_root_speed_m_s": 0.35,
        },
    },
    "careful_forward_walk": {
        "label": "Careful forward walk",
        "description": "Tighter kinematic screen for controlled walking and stopping.",
        "constraints": {
            "min_forward_progress_m": 0.5,
            "max_lateral_drift_m": 0.4,
            "min_path_efficiency": 0.72,
            "max_root_height_range_m": 0.16,
            "max_near_floor_toe_speed_m_s": 0.22,
            "max_below_floor_fraction": 0.05,
            "max_torso_lean_deg": 20.0,
            "max_end_root_speed_m_s": 0.22,
        },
    },
    "stationary_reach": {
        "label": "Stationary reach",
        "description": "Keep the base mostly planted while allowing upper-body reaching motion.",
        "constraints": {
            "max_root_displacement_m": 0.35,
            "max_root_height_range_m": 0.18,
            "max_near_floor_toe_speed_m_s": 0.18,
            "max_below_floor_fraction": 0.05,
            "max_torso_lean_deg": 32.0,
            "max_end_root_speed_m_s": 0.18,
        },
    },
    "static_balance": {
        "label": "Static balance screen",
        "description": "Minimal root drift, foot slide and torso lean. Kinematic only; not a dynamics test.",
        "constraints": {
            "max_root_displacement_m": 0.2,
            "max_root_height_range_m": 0.12,
            "max_near_floor_toe_speed_m_s": 0.12,
            "max_below_floor_fraction": 0.04,
            "max_torso_lean_deg": 16.0,
            "max_end_root_speed_m_s": 0.12,
        },
    },
}

ALLOWED = {
    "min_forward_progress_m": (0.0, 20.0, ">="),
    "max_lateral_drift_m": (0.0, 20.0, "<="),
    "min_path_efficiency": (0.0, 1.0, ">="),
    "max_root_displacement_m": (0.0, 20.0, "<="),
    "max_root_height_range_m": (0.0, 5.0, "<="),
    "max_near_floor_toe_speed_m_s": (0.0, 20.0, "<="),
    "max_below_floor_fraction": (0.0, 1.0, "<="),
    "max_torso_lean_deg": (0.0, 90.0, "<="),
    "max_end_root_speed_m_s": (0.0, 20.0, "<="),
}

LABELS = {
    "min_forward_progress_m": "Forward progress",
    "max_lateral_drift_m": "Lateral drift",
    "min_path_efficiency": "Path efficiency",
    "max_root_displacement_m": "Root displacement",
    "max_root_height_range_m": "Root height range",
    "max_near_floor_toe_speed_m_s": "Near-floor toe speed",
    "max_below_floor_fraction": "Below estimated floor fraction",
    "max_torso_lean_deg": "Torso lean (95th percentile)",
    "max_end_root_speed_m_s": "Ending root speed",
}

UNITS = {
    "min_forward_progress_m": "m",
    "max_lateral_drift_m": "m",
    "min_path_efficiency": "ratio",
    "max_root_displacement_m": "m",
    "max_root_height_range_m": "m",
    "max_near_floor_toe_speed_m_s": "m/s",
    "max_below_floor_fraction": "fraction",
    "max_torso_lean_deg": "deg",
    "max_end_root_speed_m_s": "m/s",
}


def sanitize_constraints(raw: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("constraints must be an object")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key not in ALLOWED:
            raise ValueError(f"Unsupported constraint: {key}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Constraint {key} must be numeric") from None
        lo, hi, _ = ALLOWED[key]
        if not math.isfinite(number) or not lo <= number <= hi:
            raise ValueError(f"Constraint {key} must be between {lo} and {hi}")
        out[key] = number
    if not out:
        raise ValueError("Choose at least one constraint")
    return out


def preset_payload() -> list[dict[str, Any]]:
    return [
        {"id": key, "label": val["label"], "description": val["description"], "constraints": val["constraints"]}
        for key, val in PRESETS.items()
    ]


def _safe_motion(base: Path, motion_id: str) -> dict[str, Any]:
    if not isinstance(motion_id, str) or not motion_id or len(motion_id) > 120 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in motion_id
    ):
        raise ValueError("Invalid motion ID")
    record_path = base / f"{motion_id}.json"
    if not record_path.is_file():
        raise ValueError(f"Motion not found: {motion_id}")
    record = json.loads(record_path.read_text())
    if "g1" not in str(record.get("model", "")).lower():
        raise ValueError(f"Only G1 motions are supported: {motion_id}")
    native = (base / str(record.get("native_file", ""))).resolve()
    if native.parent != base.resolve() or native.suffix != ".npz" or not native.is_file():
        raise ValueError(f"No usable G1 NPZ for {motion_id}")
    record["_native"] = native
    return record


def _motion_measurements(record: dict[str, Any]) -> dict[str, float | None]:
    native: Path = record["_native"]
    fps = float(record.get("fps", 0))
    if fps <= 0:
        raise ValueError("Motion fps must be positive")
    with np.load(native, allow_pickle=False) as data:
        p = np.asarray(data["posed_joints"], dtype=np.float64)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float64) if "global_rot_mats" in data else None
    if p.ndim != 3 or p.shape[1:] != (34, 3) or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite posed_joints shaped (T,34,3)")

    report = report_for_npz(native, fps)["report"]
    root = p[:, PELVIS]
    horizontal_delta = root[-1, [0, 2]] - root[0, [0, 2]]
    root_disp = float(np.linalg.norm(horizontal_delta))
    root_path = float(report["root_horizontal_path_m"])
    path_efficiency = root_disp / root_path if root_path > 1e-8 else 1.0

    # NVIDIA Kimodo G1 uses Y-up. Local +Z is the robot's forward direction in bundled G1 examples.
    if rotations is not None and rotations.shape[:2] == p.shape[:2]:
        forward3 = rotations[0, PELVIS] @ np.array([0.0, 0.0, 1.0])
        right3 = rotations[0, PELVIS] @ np.array([1.0, 0.0, 0.0])
        forward = forward3[[0, 2]]
        right = right3[[0, 2]]
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        right /= max(float(np.linalg.norm(right)), 1e-8)
    else:
        forward = np.array([0.0, 1.0])
        right = np.array([1.0, 0.0])
    forward_progress = float(horizontal_delta @ forward)
    lateral_drift = float(abs(horizontal_delta @ right))

    torso = p[:, WAIST_PITCH] - p[:, PELVIS]
    torso_norm = np.linalg.norm(torso, axis=1)
    cos = np.divide(torso[:, 1], torso_norm, out=np.ones_like(torso_norm), where=torso_norm > 1e-8)
    torso_lean = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    velocities = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1) * fps
    tail_frames = max(1, min(len(velocities), int(round(0.35 * fps))))
    end_speed = float(np.mean(velocities[-tail_frames:]))

    return {
        "forward_progress_m": round(forward_progress, 4),
        "lateral_drift_m": round(lateral_drift, 4),
        "path_efficiency": round(float(path_efficiency), 4),
        "root_displacement_m": round(root_disp, 4),
        "root_height_range_m": float(report["root_height_range_m"]),
        "near_floor_toe_speed_m_s": report.get("near_floor_toe_speed_m_s"),
        "below_floor_fraction": float(report["estimated_below_floor_fraction"]),
        "torso_lean_p95_deg": round(float(np.percentile(torso_lean, 95)), 3),
        "end_root_speed_m_s": round(end_speed, 4),
    }


def _observed_key(constraint_key: str) -> str:
    return {
        "min_forward_progress_m": "forward_progress_m",
        "max_lateral_drift_m": "lateral_drift_m",
        "min_path_efficiency": "path_efficiency",
        "max_root_displacement_m": "root_displacement_m",
        "max_root_height_range_m": "root_height_range_m",
        "max_near_floor_toe_speed_m_s": "near_floor_toe_speed_m_s",
        "max_below_floor_fraction": "below_floor_fraction",
        "max_torso_lean_deg": "torso_lean_p95_deg",
        "max_end_root_speed_m_s": "end_root_speed_m_s",
    }[constraint_key]


def evaluate_motion(base: Path, motion_id: str, constraints: dict[str, float]) -> dict[str, Any]:
    record = _safe_motion(base, motion_id)
    observed = _motion_measurements(record)
    checks = []
    passed = 0
    for key, limit in constraints.items():
        lo, hi, op = ALLOWED[key]
        value = observed[_observed_key(key)]
        # Missing toe-contact proxy is not silently treated as a pass.
        ok = False if value is None else (float(value) >= limit if op == ">=" else float(value) <= limit)
        if ok:
            passed += 1
        checks.append(
            {
                "key": key,
                "label": LABELS[key],
                "observed": value,
                "operator": op,
                "limit": limit,
                "unit": UNITS[key],
                "pass": ok,
            }
        )
    total = len(checks)
    return {
        "id": record["id"],
        "name": record.get("name") or record["id"],
        "native_sha256": sha256(record["_native"].read_bytes()),
        "passed": passed,
        "total": total,
        "all_pass": passed == total,
        "pass_fraction": round(passed / total, 4),
        "measurements": observed,
        "checks": checks,
    }


def evaluate_candidates(base: Path, db: Path, motion_ids: list[str], raw_constraints: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(motion_ids, list) or not 1 <= len(motion_ids) <= 100:
        raise ValueError("Choose between 1 and 100 motion IDs")
    motion_ids = list(dict.fromkeys(str(x) for x in motion_ids))
    constraints = sanitize_constraints(raw_constraints)

    ranking_signal: dict[str, float] = {}
    model = None
    try:
        with connect(db) as conn:
            model = latest(conn)
            if model:
                ranking_signal = {row["id"]: float(row["ranking_signal"]) for row in rank(conn, base)["rankings"]}
    except Exception:
        model = None
        ranking_signal = {}

    results = []
    for motion_id in motion_ids:
        item = evaluate_motion(base, motion_id, constraints)
        item["preference_signal"] = ranking_signal.get(motion_id)
        results.append(item)

    # Hard constraints dominate. Human preference is only a tie-breaker within equally satisfying candidates.
    results.sort(
        key=lambda x: (
            1 if x["all_pass"] else 0,
            x["pass_fraction"],
            x["preference_signal"] if x["preference_signal"] is not None else float("-inf"),
            x["id"],
        ),
        reverse=True,
    )
    report = {
        "version": VERSION,
        "scope": "Explicit kinematic filtering of saved NVIDIA Kimodo G1 trajectories; no IK solve, physics, balance proof, collision simulation or hardware validation.",
        "constraints": constraints,
        "selection_rule": "Hard constraint satisfaction first; Aura learned human preference signal only breaks ties. No composite quality score.",
        "selector_model_version": (model or {}).get("version"),
        "candidate_count": len(results),
        "all_pass_count": sum(1 for x in results if x["all_pass"]),
        "ordered_ids": [x["id"] for x in results],
        "results": results,
    }
    report["report_sha256"] = sha256(canonical_bytes(report))
    return report
