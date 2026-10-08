"""Reproducible screening metrics for G1 world-space joint trajectories.
These are kinematic proxies, NOT dynamic stability or hardware safety guarantees.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

# Matches Text2Motion Aura G1Skeleton34.bone_order_names_with_parents (see text2motion-aura/text2motion_aura/skeleton/definitions.py).
FEET = {'left': (6, 7), 'right': (13, 14)}  # ankle roll and toe endpoint
REPORT_VERSION = 'g1-kinematic-screen-v1'


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def evaluate_positions(positions, fps):
    p = np.asarray(positions, dtype=np.float64)
    fps = float(fps)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or p.shape[0] < 2 or fps <= 0 or not np.isfinite(p).all():
        raise ValueError('Expected at least 2 finite G1 frames shaped (T,34,3) and fps > 0')
    # Kinodo's saved posed_joints is Y-up in this integration. World origin is not
    # necessarily the physical floor; use low quantile of toe endpoints as proxy.
    toes = np.stack([p[:, 7], p[:, 14]], axis=1)
    floor_y = float(np.percentile(toes[..., 1], 5))
    # On-frame contact heuristic; 3 cm tolerance, not calibrated to hardware.
    tolerance_m = 0.03
    intervals = p.shape[0] - 1
    contacts = []
    contact_displacements = []
    for name, (_, toe_idx) in FEET.items():
        toe = p[:, toe_idx]
        near_floor = toe[:, 1] <= floor_y + tolerance_m
        both_contact = near_floor[:-1] & near_floor[1:]
        delta = np.linalg.norm(np.diff(toe[:, [0, 2]], axis=0), axis=1)
        contacts.append({'foot': name, 'contact_frame_fraction': round(float(near_floor.mean()), 4),
                         'contact_intervals': int(both_contact.sum()),
                         'contact_horizontal_speed_m_s': round(float(delta[both_contact].mean()*fps), 4) if both_contact.any() else None})
        if both_contact.any():
            contact_displacements.extend((delta[both_contact]*fps).tolist())
    # Never call this physical penetration: we don't know the actual ground height.
    below_floor_fraction = float((toes[..., 1] < floor_y - .02).mean())
    root = p[:, 0]
    path = np.diff(root[:, [0, 2]], axis=0)
    out = {
        'method': REPORT_VERSION,
        'scope': 'World-space G1 joints only; screening proxies, not simulator or hardware validation',
        'frames': int(p.shape[0]), 'fps': fps, 'duration_s': round(float(intervals/fps), 3),
        'estimated_floor_y_m': round(floor_y, 4),
        'floor_estimation': '5th percentile of toe Y positions; depends on motion and coordinate calibration',
        'root_horizontal_displacement_m': round(float(np.linalg.norm(root[-1, [0, 2]]-root[0, [0, 2]])), 4),
        'root_horizontal_path_m': round(float(np.linalg.norm(path, axis=1).sum()), 4),
        'root_height_range_m': round(float(np.ptp(root[:, 1])), 4),
        'estimated_below_floor_fraction': round(below_floor_fraction, 4),
        'near_floor_toe_speed_m_s': round(float(np.mean(contact_displacements)), 4) if contact_displacements else None,
        'feet': contacts,
        'notes': [
            'Lower near-floor horizontal toe speed can indicate less apparent foot sliding, but contact is inferred from toe height alone.',
            'Estimated floor penetration is not physical collision detection.',
            'Root displacement/path describe the generated trajectory; neither is a motion-quality score.',
            'Balance, task completion, prompt adherence and real-robot safety are NOT measured.'
        ],
    }
    return out


def report_for_npz(npz_path: Path, fps: float):
    raw_hash = sha256(npz_path.read_bytes())
    with np.load(npz_path, allow_pickle=False) as data:
        report = evaluate_positions(data['posed_joints'], fps)
    # The report hash never includes self-referential hash, timestamp, or local path.
    return {'native_sha256': raw_hash, 'report_sha256': sha256(canonical_bytes(report)), 'report': report}
