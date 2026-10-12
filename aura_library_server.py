"""Aura local development library API. Only binds to localhost; not for public deployment."""
import json
from aura_preference import connect, add_preference, list_preferences, latest, train, rank, next_pair, preference_diagnostics
from aura_g1_metrics import REPORT_VERSION, report_for_npz, sha256
from aura_io import atomic_write_bytes, atomic_write_text
from aura_downstream_benchmark import MIN_HOLDOUT, list_cohorts, run_benchmark, latest_benchmark, selector_status
from aura_constraints import preset_payload, evaluate_candidates
from aura_bounties import (create_bounty, list_bounties, list_curations, curate,
    attach_post_signature, record_payment, close_bounty, create_wallet_challenge)
from aura_text2motion_generator import AuraText2MotionGenerator, list_examples
from aura_motion_prior import train_prior, status as prior_status
from aura_reward_model import train_reward_model, status as reward_status
from aura_scene_lab import DEFAULT_SCENE, generate_scene_candidates, plan_path
from aura_scene_capture import (MAX_UPLOAD_BYTES, create_capture, get_capture, reconstruction_backend,
    splat_path as scene_splat_path, start_reconstruction)
from aura_starter_motions import ensure_starter_motions
import os
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE = Path(os.environ.get("AURA_MOTION_LIBRARY", str(Path(__file__).resolve().parent / "aura-motion-library"))).resolve()
BASE.mkdir(exist_ok=True, parents=True)
ensure_starter_motions(BASE)

DB = BASE / "aura-preferences.sqlite3"
PORT = int(os.environ.get("AURA_LIBRARY_PORT", "8765"))
GENERATOR = AuraText2MotionGenerator(BASE)

# Browser origins allowed to POST. Browsers always send Origin on cross-site requests, so a
# page on another site can't drive this API (generate, train, close bounties) through your
# browser; requests without Origin come from non-browser clients like curl and the tests.
ALLOWED_ORIGINS = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1", "[::1]") for port in (5173, 4173)}
ALLOWED_ORIGINS |= {o.strip() for o in os.environ.get("AURA_ALLOWED_ORIGINS", "").split(",") if o.strip()}

# ---- Library maintenance --------------------------------------------------------------
# Preview-format upgrades and metric caching write files, so they run in one background
# refresh (never in GET /motions) and write atomically. Failures are remembered per file
# version so a broken record isn't retried on every poll.
_REFRESH_LOCK = threading.Lock()
_REFRESH_FAILURES: dict[str, tuple[tuple, str]] = {}


def _is_g1(item):
    return 'g1' in str(item.get('model', '')).lower()


def _preview_needs_upgrade(item):
    preview = BASE / f"{item['id']}.g1.json"
    try:
        # Preview payloads can be several MB; the header is enough to tell the format.
        with preview.open("rb") as f:
            return b'"format":"g1-joints-v2"' not in f.read(96)
    except OSError:
        return True


def _metrics_current(item, digest=None):
    e = item.get('kinematic_evaluation')
    if not e or e.get('report', {}).get('method') != REPORT_VERSION:
        return False
    return digest is None or e.get('native_sha256') == digest


def _native_path(item):
    native = (BASE / str(item.get('native_file', ''))).resolve()
    if native.parent != BASE or native.suffix != '.npz' or not native.is_file():
        raise ValueError('Invalid or missing motion file')
    return native


def _file_signature(file, item):
    try:
        return (file.stat().st_mtime_ns, _native_path(item).stat().st_mtime_ns)
    except (OSError, ValueError):
        return (file.stat().st_mtime_ns, None)


def _refresh_record(file):
    item = json.loads(file.read_text())
    if not isinstance(item, dict) or not item.get('id') or not _is_g1(item) or item.get('source') == 'starter_reference':
        return
    signature = _file_signature(file, item)
    failed = _REFRESH_FAILURES.get(file.name)
    if failed and failed[0] == signature:
        return
    try:
        import numpy as np
        from aura_g1_preview import g1_preview_bytes
        native = _native_path(item)
        changed = False
        preview = BASE / f"{item['id']}.g1.json"
        if _preview_needs_upgrade(item):
            with np.load(native, allow_pickle=False) as data:
                pos = np.asarray(data["posed_joints"])
                rot = np.asarray(data["global_rot_mats"])
            atomic_write_bytes(preview, g1_preview_bytes(pos, item.get("fps", 30), rot))
            changed = True
        if item.get('preview_file') != preview.name or item.get('preview_error'):
            item['preview_file'], item['preview_error'] = preview.name, None
            changed = True
        # Cache metrics by source-file digest and algorithm version.
        digest = sha256(native.read_bytes())
        if not _metrics_current(item, digest):
            item['kinematic_evaluation'] = report_for_npz(native, item['fps'])
            changed = True
        if changed:
            atomic_write_text(file, json.dumps(item, indent=2))
        _REFRESH_FAILURES.pop(file.name, None)
    except Exception as exc:
        _REFRESH_FAILURES[file.name] = (signature, f'{type(exc).__name__}: {exc}')


def refresh_library():
    """Bring every motion record's preview and metrics up to date. One run at a time."""
    if not _REFRESH_LOCK.acquire(blocking=False):
        return
    try:
        for file in sorted(BASE.glob("*.json")):
            if file.name.endswith('.g1.json') or file.name.startswith('.'):
                continue
            try:
                _refresh_record(file)
            except (OSError, ValueError):
                continue
    finally:
        _REFRESH_LOCK.release()


def schedule_library_refresh():
    if not _REFRESH_LOCK.locked():
        threading.Thread(target=refresh_library, name='aura-library-refresh', daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def respond_json(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_binary(self, status, data, content_type="application/octet-stream", filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def _reject_unsafe_post(self, path):
        """Origin must be an allowed local origin (when present); bodies must be JSON, except
        the video upload. Both force cross-site browser requests through a CORS preflight,
        which this server never approves."""
        origin = self.headers.get('Origin')
        if origin is not None and origin not in ALLOWED_ORIGINS:
            self.respond_json(403, {'error': 'Cross-origin request rejected. Set AURA_ALLOWED_ORIGINS to allow another local UI origin.'})
            return True
        content_type = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        if path == '/scene/captures':
            if not content_type.startswith('video/'):
                self.respond_json(415, {'error': 'Camera captures must be uploaded with a video/* Content-Type'})
                return True
        elif content_type != 'application/json':
            self.respond_json(415, {'error': 'Requests must use Content-Type: application/json'})
            return True
        return False

    def do_POST(self):
        path = urlparse(self.path).path
        if self._reject_unsafe_post(path):
            return
        try:
            if path == '/scene/captures':
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > MAX_UPLOAD_BYTES:
                    raise ValueError(f'Camera capture must be between 1 byte and {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')
                content_type = self.headers.get('Content-Type')
                name = self.headers.get('X-Aura-Capture-Name', 'Camera scene')
                duration = self.headers.get('X-Aura-Capture-Duration')
                out = create_capture(
                    self.rfile.read(size), content_type=content_type, name=name,
                    duration_s=float(duration) if duration not in (None, '') else None,
                )
                return self.respond_json(201, out)
            capture_parts = path.strip('/').split('/')
            if len(capture_parts) == 4 and capture_parts[0] == 'scene' and capture_parts[1] == 'captures' and capture_parts[3] == 'reconstruct':
                out = start_reconstruction(capture_parts[2], before_start=GENERATOR.release_model)
                return self.respond_json(202, out)
            if path not in ('/preferences', '/train', '/prior/train', '/reward/train', '/benchmark', '/constraints/evaluate', '/generator/generate', '/scene/plan', '/scene/generate', '/bounties', '/bounties/post-signature', '/bounties/curate', '/bounties/pay', '/bounties/close', '/bounties/challenge'):
                return self.send_error(404)
            size = int(self.headers.get('Content-Length', '0'))
            if size > 16384 or size < 0: raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size)) if size else {}
            if path == '/scene/plan':
                out = plan_path(body.get('scene'))
            elif path == '/scene/generate':
                out = generate_scene_candidates(
                    GENERATOR, BASE, prompt=body.get('prompt', ''), scene=body.get('scene') or DEFAULT_SCENE, count=body.get('count', 2)
                )
            elif path == '/generator/generate':
                out = GENERATOR.generate(
                    prompt=body.get('prompt'),
                    example_id=body.get('example_id'),
                    duration=body.get('duration'),
                    diffusion_steps=body.get('diffusion_steps'),
                    seed=body.get('seed'),
                    count=body.get('count'),
                )
            elif path == '/benchmark':
                out = run_benchmark(
                    BASE, DB, cohort_name=body.get('cohort_name', ''),
                    selector=body.get('selector', 'reward_transformer'),
                    train_count=body.get('train_count', 6), holdout_count=body.get('holdout_count', 6),
                    random_trials=body.get('random_trials', 20), samples_per_motion=body.get('samples_per_motion', 100),
                    horizon_s=body.get('horizon_s', 0.5), seed=body.get('seed', 37),
                )
            elif path == '/constraints/evaluate':
                out = evaluate_candidates(BASE, DB, body.get('motion_ids', []), body.get('constraints', {}))
            else:
                with connect(DB) as conn:
                    if path == '/preferences':
                        out = add_preference(
                            conn, BASE, body.get('left_id'), body.get('right_id'),
                            body.get('winner_id'), body.get('context', ''), body.get('evaluator_id')
                        )
                        out['training_data'] = preference_diagnostics(conn, BASE)
                    elif path == '/train':
                        out = train(conn, BASE)
                    elif path == '/prior/train':
                        out = train_prior(conn, BASE)
                    elif path == '/reward/train':
                        out = train_reward_model(conn, BASE, epochs=body.get('epochs', 40))
                    elif path == '/bounties':
                        out = create_bounty(conn, body)
                    elif path == '/bounties/post-signature':
                        out = attach_post_signature(conn, body.get('bounty_id', ''), body.get('signature'))
                    elif path == '/bounties/curate':
                        out = curate(conn, BASE, body)
                    elif path == '/bounties/pay':
                        out = record_payment(conn, body)
                    elif path == '/bounties/close':
                        out = close_bounty(conn, body)
                    elif path == '/bounties/challenge':
                        out = create_wallet_challenge(conn, body)
            self.respond_json(200, out)
        except (ValueError, TypeError, KeyError) as e:
            self.respond_json(400, {'error': str(e)})
        except RuntimeError as e:
            self.respond_json(503, {'error': str(e)})
        except Exception:
            ref = uuid.uuid4().hex[:8]
            print(f'[aura-api] error {ref} on POST {path}', file=sys.stderr)
            traceback.print_exc()
            self.respond_json(500, {'error': f'Internal server error (ref {ref}). Details are in the Aura API log.'})


    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == '/generator/status':
            return self.respond_json(200, GENERATOR.status())
        if path == '/scene/default':
            return self.respond_json(200, {'scene': DEFAULT_SCENE})
        if path == '/scene/capture/status':
            return self.respond_json(200, {
                'backend': reconstruction_backend(),
                'max_upload_mb': MAX_UPLOAD_BYTES // (1024 * 1024),
                'capture_note': 'Browser camera capture requires localhost or HTTPS. Reconstruction is visual context; scene planning still uses the explicit geometry proxy.',
            })
        capture_parts = path.strip('/').split('/')
        if len(capture_parts) in (3, 4) and capture_parts[0] == 'scene' and capture_parts[1] == 'captures':
            try:
                capture_id = capture_parts[2]
                if len(capture_parts) == 3:
                    return self.respond_json(200, get_capture(capture_id))
                if capture_parts[3] == 'splat':
                    file = scene_splat_path(capture_id)
                    return self.respond_binary(200, file.read_bytes(), 'application/octet-stream', f'{capture_id}.splat')
            except (ValueError, OSError) as exc:
                return self.respond_json(404, {'error': str(exc)})
        if path == '/generator/examples':
            return self.respond_json(200, {'examples': list_examples()})
        if path == '/bounties':
            try:
                with connect(DB) as conn:
                    return self.respond_json(200, {'bounties': list_bounties(conn)})
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__})
        if path == '/bounties/curations':
            try:
                bounty_id = (query.get('bounty_id') or [None])[0]
                with connect(DB) as conn:
                    return self.respond_json(200, {'curations': list_curations(conn, bounty_id)})
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__})
        if path == '/constraints/presets':
            return self.respond_json(200, {'version': 'aura-native-g1-constraints-v1', 'presets': preset_payload()})
        if path == '/benchmark/cohorts':
            try:
                cohorts = list_cohorts(BASE)
                status = selector_status(BASE, DB)
                for cohort in cohorts:
                    cohort['selectors'] = {
                        name: {'label': s['label'], 'trained': s['trained'], 'version': s['version'],
                               'unseen_count': sum(1 for mid in cohort['motion_ids'] if mid not in set(s['seen_motion_ids'])) if s['trained'] else 0}
                        for name, s in status.items()
                    }
                return self.respond_json(200, {'cohorts': cohorts, 'min_holdout': MIN_HOLDOUT})
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__})
        if path == '/benchmark':
            try:
                return self.respond_json(200, {'result': latest_benchmark(BASE)})
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__})
        if path == '/prior':
            try:
                return self.respond_json(200, prior_status(BASE))
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__ + ': ' + str(exc)[:300]})
        if path == '/reward':
            try:
                return self.respond_json(200, reward_status(BASE))
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__ + ': ' + str(exc)[:300]})
        if path == '/preferences/diagnostics':
            try:
                with connect(DB) as conn:
                    return self.respond_json(200, preference_diagnostics(conn, BASE))
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__ + ': ' + str(exc)[:300]})
        if path in ('/preferences', '/critic', '/rankings', '/next-pair'):
            try:
                with connect(DB) as conn:
                    data = {'preferences':list_preferences(conn)} if path == '/preferences' else (
                        {'model':latest(conn), 'vote_count':len(list_preferences(conn))} if path == '/critic' else (
                            rank(conn, BASE) if path == '/rankings' else next_pair(conn, BASE)))
                return self.respond_json(200, data)
            except Exception as exc:
                return self.respond_json(500, {'error': type(exc).__name__})
        if path == "/motions":
            # Read-only: stale previews/metrics are fixed by the background refresh, which
            # writes atomically, so a poll never sees (or produces) a half-written file.
            entries = []
            stale = False
            for file in BASE.glob("*.json"):
                if file.name.endswith('.g1.json') or file.name.startswith('.'):
                    continue  # Preview JSON and temp files are not motion-library records.
                try:
                    item = json.loads(file.read_text())
                    if not isinstance(item, dict) or not item.get('id') or not item.get('native_file'):
                        continue
                    # Starter/reference records exist in the same library so their
                    # human votes can train Aura, but they are already rendered by
                    # the top hero arena and should not appear as generated library items.
                    if item.get('source') == 'starter_reference':
                        continue
                    if _is_g1(item):
                        failure = _REFRESH_FAILURES.get(file.name)
                        if failure:
                            item['preview_error'] = f'G1 preview recovery: {failure[1]}'
                        elif _preview_needs_upgrade(item):
                            stale = True
                        if not _metrics_current(item):
                            stale = stale or not failure
                            # Serve fresh metrics now; the refresh persists them.
                            try:
                                item['kinematic_evaluation'] = report_for_npz(_native_path(item), item['fps'])
                            except Exception as exc:
                                item['evaluation_error'] = str(exc)
                    entries.append(item)
                except (ValueError, OSError):
                    continue
            if stale:
                schedule_library_refresh()
            entries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
            payload = json.dumps(entries).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/files/"):
            name = path[len("/files/"):]
            if not name or "/" in name or "\\" in name or not name.endswith((".bvh", ".npz", ".g1.json")):
                return self.send_error(400)
            file = BASE / name
            if not file.is_file():
                return self.send_error(404)
            data = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if name.endswith(".g1.json") else "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

if __name__ == "__main__":
    print(f"Aura library: {BASE} at http://127.0.0.1:{PORT}")
    schedule_library_refresh()  # upgrade any legacy previews/metrics in the background
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
