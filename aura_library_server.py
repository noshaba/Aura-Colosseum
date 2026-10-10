"""Aura local development library API. Only binds to localhost; not for public deployment."""
import json
from aura_preference import connect, add_preference, list_preferences, latest, train, rank, next_pair, preference_diagnostics
from aura_g1_metrics import report_for_npz
from aura_downstream_benchmark import list_cohorts, run_benchmark, latest_benchmark
from aura_constraints import preset_payload, evaluate_candidates
from aura_bounties import (create_bounty, list_bounties, list_curations, curate,
    attach_post_signature, record_payment, close_bounty)
from aura_text2motion_generator import AuraText2MotionGenerator, list_examples
from aura_motion_prior import train_prior, status as prior_status
from aura_reward_model import train_reward_model, status as reward_status
from aura_starter_motions import ensure_starter_motions
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE = Path(os.environ.get("AURA_MOTION_LIBRARY", str(Path(__file__).resolve().parent / "aura-motion-library"))).resolve()
BASE.mkdir(exist_ok=True, parents=True)
ensure_starter_motions(BASE)

DB = BASE / "aura-preferences.sqlite3"
PORT = int(os.environ.get("AURA_LIBRARY_PORT", "8765"))
GENERATOR = AuraText2MotionGenerator(BASE)

class Handler(BaseHTTPRequestHandler):
    def respond_json(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/preferences', '/train', '/prior/train', '/reward/train', '/benchmark', '/constraints/evaluate', '/generator/generate', '/bounties', '/bounties/post-signature', '/bounties/curate', '/bounties/pay', '/bounties/close'):
            return self.send_error(404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size > 16384 or size < 0: raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size)) if size else {}
            if path == '/generator/generate':
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
                    train_count=body.get('train_count', 4), holdout_count=body.get('holdout_count', 2),
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
                        out = close_bounty(conn, body.get('bounty_id', ''))
            self.respond_json(200, out)
        except (ValueError, TypeError, KeyError) as e:
            self.respond_json(400, {'error': str(e)})
        except RuntimeError as e:
            self.respond_json(503, {'error': str(e)})
        except Exception as e:
            self.respond_json(500, {'error': 'Server error: ' + type(e).__name__ + ': ' + str(e)[:500]})


    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == '/generator/status':
            return self.respond_json(200, GENERATOR.status())
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
                with connect(DB) as conn:
                    model = latest(conn)
                seen = set((model or {}).get('source_motion_ids') or [])
                for cohort in cohorts:
                    cohort['selector_unseen_count'] = sum(1 for mid in cohort['motion_ids'] if mid not in seen) if model else 0
                    cohort['selector_model_version'] = (model or {}).get('version')
                return self.respond_json(200, {'cohorts': cohorts})
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
            entries = []
            for file in BASE.glob("*.json"):
                if file.name.endswith('.g1.json'):
                    continue  # Preview JSON is not a motion-library record.
                try:
                    item = json.loads(file.read_text())
                    if not isinstance(item, dict) or not item.get('id') or not item.get('native_file'):
                        continue
                    # Starter/reference records exist in the same library so their
                    # human votes can train Aura, but they are already rendered by
                    # the top hero arena and should not appear as generated library items.
                    if item.get('source') == 'starter_reference':
                        continue
                    if item.get("model", "").lower().find("g1") >= 0:
                        try:
                            import numpy as np
                            from aura_g1_preview import g1_preview_bytes
                            native = BASE / item["native_file"]
                            preview = BASE / f"{item['id']}.g1.json"
                            needs_upgrade = not preview.is_file()
                            if preview.is_file():
                                try:
                                    # Preview payloads can be several MB; inspect only the header on polling.
                                    header = preview.open("rb").read(96)
                                    needs_upgrade = b'"format":"g1-joints-v2"' not in header
                                except Exception:
                                    needs_upgrade = True
                            if needs_upgrade:
                                with np.load(native, allow_pickle=False) as data:
                                    pos = np.asarray(data["posed_joints"])
                                    rot = np.asarray(data["global_rot_mats"])
                                preview.write_bytes(g1_preview_bytes(pos, item.get("fps", 30), rot))
                            item["preview_file"] = preview.name
                            item["preview_error"] = None
                            file.write_text(json.dumps(item, indent=2))
                        except Exception as exc:
                            item["preview_error"] = f"G1 preview recovery: {exc}"
                    if 'g1' in item.get('model', '').lower() and (BASE / item.get('native_file', '')).is_file():
                        try:
                            native = BASE / item['native_file']
                            if native.parent != BASE or native.suffix != '.npz':
                                raise ValueError('Invalid motion filename')
                            # Cache results by source-file digest and algorithm version.
                            from aura_g1_metrics import REPORT_VERSION, sha256
                            digest = sha256(native.read_bytes())
                            existing = item.get('kinematic_evaluation')
                            if (not existing or existing.get('native_sha256') != digest
                                    or existing.get('report', {}).get('method') != REPORT_VERSION):
                                item['kinematic_evaluation'] = report_for_npz(native, item['fps'])
                                file.write_text(json.dumps(item, indent=2))
                        except Exception as exc:
                            item['evaluation_error'] = str(exc)
                    entries.append(item)
                except (ValueError, OSError):
                    continue
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
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
