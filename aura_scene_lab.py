"""Scene-aware locomotion layer for Aura.

This module intentionally keeps NVIDIA Kimodo as the motion generator.  It adds a
splat-derived occupancy map or explicit geometry proxy, plans a collision-aware
2D root path, and bends the generated G1 root trajectory onto that path.
Occupancy uses Aura bounds ordering: [xmin, xmax, zmin, zmax].
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from aura_io import atomic_write_bytes, atomic_write_text

SCENE_METHOD = "aura-scene-root-path-fit-v1"
DEFAULT_SCENE = {
    "name": "Demo room",
    "bounds": [-3.0, 3.0, -2.4, 2.4],  # xmin, xmax, zmin, zmax
    "start": [-2.2, -1.2],
    "goal": [2.2, 1.1],
    "floor_y": 0.0,
    "robot_radius": 0.34,
    "grid_resolution": 0.18,
    "obstacles": [
        {"name": "table", "x": 0.0, "z": 0.0, "width": 1.35, "depth": 0.82},
    ],
}


def _f(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def normalize_scene(scene: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(DEFAULT_SCENE if not scene else scene)
    bounds = src.get("bounds", DEFAULT_SCENE["bounds"])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("scene.bounds must be [xmin, xmax, zmin, zmax]")
    xmin, xmax, zmin, zmax = [_f(v, "scene.bounds") for v in bounds]
    if xmax - xmin < 1.0 or zmax - zmin < 1.0:
        raise ValueError("scene bounds are too small")
    start = src.get("start", DEFAULT_SCENE["start"])
    goal = src.get("goal", DEFAULT_SCENE["goal"])
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError("scene.start must be [x, z]")
    if not isinstance(goal, (list, tuple)) or len(goal) != 2:
        raise ValueError("scene.goal must be [x, z]")
    start = [_f(start[0], "scene.start.x"), _f(start[1], "scene.start.z")]
    goal = [_f(goal[0], "scene.goal.x"), _f(goal[1], "scene.goal.z")]
    radius = max(0.12, min(0.8, _f(src.get("robot_radius", 0.34), "scene.robot_radius")))
    resolution = max(0.08, min(0.5, _f(src.get("grid_resolution", 0.18), "scene.grid_resolution")))
    floor_y = _f(src.get("floor_y", 0.0), "scene.floor_y")
    for label, point in (("start", start), ("goal", goal)):
        if not (xmin <= point[0] <= xmax and zmin <= point[1] <= zmax):
            raise ValueError(f"scene {label} is outside scene bounds")

    obstacles = []
    raw_obstacles = src.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        raise ValueError("scene.obstacles must be a list")
    if len(raw_obstacles) > 64:
        raise ValueError("scene supports at most 64 geometry-proxy obstacles")
    for index, obstacle in enumerate(raw_obstacles):
        if not isinstance(obstacle, dict):
            raise ValueError("each obstacle must be an object")
        width = max(0.05, _f(obstacle.get("width", 1.0), f"obstacle[{index}].width"))
        depth = max(0.05, _f(obstacle.get("depth", 1.0), f"obstacle[{index}].depth"))
        obstacles.append({
            "name": str(obstacle.get("name") or f"obstacle-{index + 1}")[:80],
            "x": _f(obstacle.get("x", 0.0), f"obstacle[{index}].x"),
            "z": _f(obstacle.get("z", 0.0), f"obstacle[{index}].z"),
            "width": width,
            "depth": depth,
        })
    occupancy = src.get("occupancy")
    if occupancy is not None:
        if not isinstance(occupancy, dict):
            raise ValueError("scene.occupancy must be an object")
        cell = _f(occupancy.get("cell_size"), "occupancy.cell_size")
        if not 0.05 <= cell <= 0.5:
            raise ValueError("occupancy.cell_size must be between 0.05 and 0.5")
        if occupancy.get("bounds") != [xmin, xmax, zmin, zmax]:
            raise ValueError("occupancy bounds must match scene bounds [xmin, xmax, zmin, zmax]")
        nx, nz = math.ceil((xmax-xmin)/cell), math.ceil((zmax-zmin)/cell)
        raw = occupancy.get("occupied")
        if nx*nz > 250000 or not isinstance(raw, list) or len(raw) > 250000:
            raise ValueError("occupancy exceeds 250,000 cells")
        cells = set()
        for pair in raw:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) is not int for v in pair)
                    or not (0 <= pair[0] < nx and 0 <= pair[1] < nz)):
                raise ValueError("invalid occupied cell index")
            cells.add(tuple(pair))
        occupancy = {"cell_size": cell, "bounds": [xmin, xmax, zmin, zmax],
                     "floor_y": floor_y, "occupied": [list(p) for p in sorted(cells)]}
    return {
        "occupancy": occupancy,
        "safety_margin": max(0.0, min(0.5, _f(src.get("safety_margin", 0.08), "safety_margin"))),
        "name": str(src.get("name") or "Scene")[:100],
        "bounds": [xmin, xmax, zmin, zmax],
        "start": start,
        "goal": goal,
        "floor_y": floor_y,
        "robot_radius": radius,
        "grid_resolution": resolution,
        "obstacles": obstacles,
    }


def _collision_boxes(a, b, scene):
    r = scene["robot_radius"] + scene.get("safety_margin", 0.08)
    for o in scene["obstacles"]:
        yield (o["x"]-o["width"]/2-r, o["x"]+o["width"]/2+r,
               o["z"]-o["depth"]/2-r, o["z"]+o["depth"]/2+r)
    occ = scene.get("occupancy")
    if occ:
        cells = scene.get("_occupied")
        if cells is None:
            cells = {tuple(p) for p in occ["occupied"]}
        c = occ["cell_size"]
        xmin, xmax, zmin, zmax = occ["bounds"]
        for ix in range(max(0, math.floor((min(a[0], b[0])-r-xmin-1e-9)/c)),
                        min(math.ceil((xmax-xmin)/c)-1, math.floor((max(a[0], b[0])+r-xmin)/c))+1):
            for iz in range(max(0, math.floor((min(a[1], b[1])-r-zmin-1e-9)/c)),
                            min(math.ceil((zmax-zmin)/c)-1, math.floor((max(a[1], b[1])+r-zmin)/c))+1):
                if (ix, iz) in cells:
                    yield xmin+ix*c-r, xmin+(ix+1)*c+r, zmin+iz*c-r, zmin+(iz+1)*c+r


def _point_blocked(x: float, z: float, scene: dict[str, Any]) -> bool:
    return not _segment_clear((x, z), (x, z), scene)


def _grid(scene: dict[str, Any]):
    xmin, xmax, zmin, zmax = scene["bounds"]
    res = scene["grid_resolution"]
    nx = max(2, int(math.floor((xmax - xmin) / res)) + 1)
    nz = max(2, int(math.floor((zmax - zmin) / res)) + 1)

    def to_idx(point):
        ix = int(round((point[0] - xmin) / res))
        iz = int(round((point[1] - zmin) / res))
        return max(0, min(nx - 1, ix)), max(0, min(nz - 1, iz))

    def to_world(cell):
        return xmin + cell[0] * res, zmin + cell[1] * res

    return nx, nz, to_idx, to_world


def _segment_clear(a: tuple[float, float], b: tuple[float, float], scene: dict[str, Any]) -> bool:
    r = scene["robot_radius"] + scene.get("safety_margin", 0.08)
    xmin, xmax, zmin, zmax = scene["bounds"]
    if any(not (xmin+r <= p[0] <= xmax-r and zmin+r <= p[1] <= zmax-r) for p in (a, b)):
        return False
    # Exact slab intersection, including tangency, against inflated rectangles.
    for box in _collision_boxes(a, b, scene):
        lo, hi = 0.0, 1.0
        for axis, lower, upper in ((0, box[0], box[1]), (1, box[2], box[3])):
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-12:
                if not lower <= a[axis] <= upper:
                    lo, hi = 1.0, 0.0
                    break
            else:
                t0, t1 = (lower-a[axis])/delta, (upper-a[axis])/delta
                lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo <= hi:
            return False
    return True


def _simplify_path(path: list[tuple[float, float]], scene: dict[str, Any]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        far = len(path) - 1
        while far > anchor + 1 and not _segment_clear(path[anchor], path[far], scene):
            far -= 1
        out.append(path[far])
        anchor = far
    return out


def plan_path(scene_payload: dict[str, Any] | None) -> dict[str, Any]:
    scene = normalize_scene(scene_payload)
    if scene.get("occupancy"):
        scene["_occupied"] = {tuple(p) for p in scene["occupancy"]["occupied"]}
    nx, nz, to_idx, to_world = _grid(scene)
    if nx * nz > 250000:
        raise ValueError("Planning grid exceeds 250,000 cells")
    for label in ("start", "goal"):
        if not _segment_clear(tuple(scene[label]), to_world(to_idx(scene[label])), scene):
            raise ValueError(f"scene {label} or its grid connection lacks clearance; move it to free space")
    start = to_idx(scene["start"])
    goal = to_idx(scene["goal"])
    if _point_blocked(*to_world(start), scene):
        raise ValueError("scene start is inside the expanded collision proxy")
    if _point_blocked(*to_world(goal), scene):
        raise ValueError("scene goal is inside the expanded collision proxy")

    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dz, step_cost in moves:
            nxt = (current[0] + dx, current[1] + dz)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < nz):
                continue
            wx, wz = to_world(nxt)
            if not _segment_clear(to_world(current), (wx, wz), scene):
                continue
            # Avoid diagonal corner cutting.
            if dx and dz:
                if _point_blocked(*to_world((current[0] + dx, current[1])), scene) or _point_blocked(*to_world((current[0], current[1] + dz)), scene):
                    continue
            new_cost = cost[current] + step_cost
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came[nxt] = current
    if goal not in came:
        raise ValueError("No collision-clear path exists for the current geometry proxy")

    cells = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = came[cursor]
    cells.reverse()
    raw = [to_world(cell) for cell in cells]
    raw = [tuple(scene["start"]), *raw, tuple(scene["goal"])]
    smooth = _simplify_path(raw, scene)
    scene.pop("_occupied", None)
    distance = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(smooth, smooth[1:]))
    return {
        "method": "aura-a-star-occupancy-v2",
        "scene": scene,
        "path": [[x, z] for x, z in smooth],
        "path_length_m": round(distance, 5),
        "grid_cells": len(cells),
        "note": "Path checked against scene occupancy and explicit obstacles, inflated by robot radius plus safety margin. Unobserved scan geometry is not guaranteed free.",
    }


def _resample_polyline(path: list[list[float]], count: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) < 2:
        raise ValueError("A planned path needs at least two points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total < 1e-5:
        raise ValueError("The planned path is too short")
    targets = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    headings = np.empty(count, dtype=np.float32)
    for i, distance in enumerate(targets):
        j = int(np.searchsorted(cumulative, distance, side="right") - 1)
        j = max(0, min(len(seg) - 1, j))
        local = 0.0 if seg[j] < 1e-8 else float((distance - cumulative[j]) / seg[j])
        out[i] = pts[j] * (1.0 - local) + pts[j + 1] * local
        direction = pts[j + 1] - pts[j]
        headings[i] = math.atan2(float(direction[0]), float(direction[1]))
    return out, headings


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _native_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_motion_to_scene(base: Path, item: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a freshly generated G1 motion in place so its root follows the plan."""
    from aura_g1_preview import g1_preview_bytes

    native = (base / str(item.get("native_file", ""))).resolve()
    if native.parent != base.resolve() or not native.is_file() or native.suffix != ".npz":
        raise ValueError("Generated scene candidate has no valid G1 NPZ")
    preview = (base / str(item.get("preview_file", ""))).resolve()
    if preview.parent != base.resolve():
        raise ValueError("Generated scene candidate has an invalid preview path")

    with np.load(native, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    posed = np.asarray(arrays["posed_joints"], dtype=np.float32)
    global_rot = np.asarray(arrays["global_rot_mats"], dtype=np.float32)
    frames = len(posed)
    if frames < 2:
        raise ValueError("Generated scene candidate is too short")
    target, headings = _resample_polyline(plan["path"], frames)

    original_root = posed[:, 0, :].copy()
    horizontal = original_root[-1, [0, 2]] - original_root[0, [0, 2]]
    if float(np.linalg.norm(horizontal)) < 0.05:
        # Fall back to the pelvis-to-neck-ish forward estimate when root translation is tiny.
        horizontal = np.asarray([0.0, 1.0], dtype=np.float32)
    source_heading = math.atan2(float(horizontal[0]), float(horizontal[1]))
    floor_y = float(plan["scene"].get("floor_y", 0.0))
    y_shift = floor_y - float(np.percentile(posed[:, :, 1], 2.0))

    fitted = posed.copy()
    fitted_global = global_rot.copy()
    root_local = np.asarray(arrays.get("local_rot_mats"), dtype=np.float32).copy() if "local_rot_mats" in arrays else None
    for i in range(frames):
        yaw = float(headings[i] - source_heading)
        rot = _yaw_matrix(yaw)
        offsets = posed[i] - original_root[i]
        rotated = offsets @ rot.T
        fitted[i, :, 0] = rotated[:, 0] + target[i, 0]
        fitted[i, :, 1] = rotated[:, 1] + original_root[i, 1] + y_shift
        fitted[i, :, 2] = rotated[:, 2] + target[i, 1]
        fitted_global[i] = rot[None, :, :] @ global_rot[i]
        if root_local is not None:
            root_local[i, 0] = rot @ root_local[i, 0]

    arrays["posed_joints"] = fitted.astype(np.float32)
    arrays["global_rot_mats"] = fitted_global.astype(np.float32)
    arrays["root_positions"] = fitted[:, 0, :].astype(np.float32)
    if root_local is not None:
        arrays["local_rot_mats"] = root_local.astype(np.float32)

    original_hash = str(item.get("native_sha256") or _native_sha256(native))
    np.savez(native, **arrays)
    atomic_write_bytes(preview, g1_preview_bytes(fitted, float(item.get("fps", 30.0)), fitted_global))
    fitted_hash = _native_sha256(native)
    item = dict(item)
    item["pre_scene_native_sha256"] = original_hash
    item["native_sha256"] = fitted_hash
    item["model"] = "nvidia-kimodo-g1 + aura-scene-fit"
    item["scene_fit"] = {
        "method": SCENE_METHOD,
        "scene_name": plan["scene"]["name"],
        "path": plan["path"],
        "path_length_m": plan["path_length_m"],
        "geometry_proxy": plan["scene"],
        "claim": "Root-path adaptation against an explicit kinematic geometry proxy; not physics simulation or contact-rich scene interaction.",
    }
    atomic_write_text(base / f"{item['id']}.json", json.dumps(item, indent=2))
    return item


def generate_scene_candidates(generator, base: Path, *, prompt: str, scene: dict[str, Any], count: int = 2) -> dict[str, Any]:
    if int(count) != 2:
        raise ValueError("Aura Scene Lab currently generates exactly two candidates")
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a locomotion/navigation prompt")
    plan = plan_path(scene)
    generated = generator.generate(prompt=prompt, count=2)
    fitted = [fit_motion_to_scene(base, dict(item), plan) for item in generated.get("motions", [])]
    if len(fitted) != 2:
        raise RuntimeError("NVIDIA Kimodo did not return two scene candidates")
    return {
        "ok": True,
        "motions": fitted,
        "plan": plan,
        "generation": generated.get("generation", {}),
        "scope": "locomotion/navigation root-path fitting; not scene-conditioned pose generation",
    }
