"""Server and storage hardening: request guards, sanitized 500s, read-only GET /motions,
atomic writes, and all-or-nothing curations."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from aura_bounties import attach_post_signature, create_bounty, curate, list_curations
from aura_g1_metrics import report_for_npz
from aura_io import atomic_write_bytes
from aura_preference import connect, list_preferences
from wallet_test_utils import TestWallet, signed

ROOT = Path(__file__).resolve().parent
REQUESTER = '11111111111111111111111111111111'
CURATORS = (TestWallet(11), TestWallet(12))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def write_motion(base, mid, offset, *, with_metrics=True):
    p = np.zeros((40, 34, 3)); p[:, :, 1] = 1.; p[:, 7, 1] = p[:, 14, 1] = .1; p[:, 0, 2] = np.linspace(0, .5 + offset, 40)
    rot = np.tile(np.eye(3), (40, 34, 1, 1))
    npz = base / f'{mid}.npz'
    np.savez_compressed(npz, posed_joints=p, global_rot_mats=rot)
    item = {'id': mid, 'name': 'Walk forward', 'model': 'g1', 'fps': 30, 'native_file': npz.name}
    if with_metrics:
        item['kinematic_evaluation'] = report_for_npz(npz, 30)
    (base / f'{mid}.json').write_text(json.dumps(item))


class ServerProcess:
    """The real server in a subprocess, on its own port and library."""

    def __init__(self, base):
        self.port = free_port()
        env = {**os.environ, 'AURA_MOTION_LIBRARY': str(base), 'AURA_LIBRARY_PORT': str(self.port)}
        self.proc = subprocess.Popen([sys.executable, 'aura_library_server.py'], env=env, cwd=ROOT,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(60):
            try:
                urlopen(f'http://127.0.0.1:{self.port}/generator/examples', timeout=1).read()
                return
            except Exception:
                time.sleep(.1)
        self.stop()
        raise RuntimeError('server did not start')

    def post(self, path, body=b'{}', headers=None):
        req = Request(f'http://127.0.0.1:{self.port}{path}', data=body, headers=headers or {}, method='POST')
        try:
            with urlopen(req, timeout=5) as r:
                return r.status, json.load(r)
        except HTTPError as e:
            return e.code, json.load(e)

    def get(self, path):
        with urlopen(f'http://127.0.0.1:{self.port}{path}', timeout=10) as r:
            return json.load(r)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)
        return self.proc.stderr.read().decode(errors='replace')


class RequestGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        for i in range(2):
            write_motion(cls.base, f'm{i}', i * .1)
        # A directory where the SQLite file should be: any DB access raises an internal error.
        (cls.base / 'aura-preferences.sqlite3').mkdir()
        cls.server = ServerProcess(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.tmp.cleanup()

    JSON = {'Content-Type': 'application/json'}

    def test_foreign_origin_is_rejected(self):
        status, body = self.server.post('/train', headers={**self.JSON, 'Origin': 'https://evil.example'})
        self.assertEqual(status, 403)
        self.assertIn('Cross-origin', body['error'])

    def test_simple_cross_site_bodies_are_rejected(self):
        # text/plain and form posts skip the browser's CORS preflight, so they must be refused.
        for ctype in ('text/plain', 'application/x-www-form-urlencoded', ''):
            status, _ = self.server.post('/bounties/close', b'{"bounty_id":"x"}', {'Content-Type': ctype} if ctype else {})
            self.assertEqual(status, 415, ctype)

    def test_video_upload_requires_a_video_content_type(self):
        status, _ = self.server.post('/scene/captures', b'not a video', {'Content-Type': 'text/plain'})
        self.assertEqual(status, 415)

    def test_local_ui_origin_and_plain_clients_pass_the_guards(self):
        for headers in ({**self.JSON, 'Origin': 'http://localhost:5173'}, self.JSON):
            status, body = self.server.post('/scene/plan', b'{}', headers)
            self.assertNotIn(status, (403, 415), body)

    def test_internal_errors_do_not_leak_details(self):
        status, body = self.server.post('/preferences', json.dumps({
            'left_id': 'm0', 'right_id': 'm1', 'winner_id': 'm0', 'context': 'walk'}).encode(), self.JSON)
        self.assertEqual(status, 500)
        self.assertRegex(body['error'], r'^Internal server error \(ref [0-9a-f]{8}\)')
        self.assertNotIn('sqlite', body['error'].lower())
        self.assertNotIn('directory', body['error'].lower())


class MotionsReadOnlyTests(unittest.TestCase):
    def test_get_motions_never_writes_and_the_refresh_persists_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_motion(base, 'stale', .2, with_metrics=False)  # no metrics, no preview
            os.environ['AURA_MOTION_LIBRARY'] = tmp
            try:
                sys.modules.pop('aura_library_server', None)
                import aura_library_server as server
            finally:
                os.environ.pop('AURA_MOTION_LIBRARY')
            refreshes = []
            with patch.object(server, 'schedule_library_refresh', lambda: refreshes.append(1)):
                port = free_port()
                httpd = server.ThreadingHTTPServer(('127.0.0.1', port), server.Handler)
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                try:
                    before = {p.name: p.stat().st_mtime_ns for p in base.iterdir()}
                    motions = json.load(urlopen(f'http://127.0.0.1:{port}/motions', timeout=10))
                    after = {p.name: p.stat().st_mtime_ns for p in base.iterdir()}
                    self.assertEqual(before, after, 'GET /motions must not write files')
                    stale = next(m for m in motions if m['id'] == 'stale')
                    self.assertIn('kinematic_evaluation', stale)  # computed in memory for the response
                    self.assertEqual(refreshes, [1])

                    server.refresh_library()
                    record = json.loads((base / 'stale.json').read_text())
                    self.assertIn('kinematic_evaluation', record)
                    self.assertEqual(record['preview_file'], 'stale.g1.json')
                    self.assertTrue((base / 'stale.g1.json').is_file())
                    self.assertFalse([p for p in base.iterdir() if p.name.endswith('.tmp')])

                    urlopen(f'http://127.0.0.1:{port}/motions', timeout=10).read()
                    self.assertEqual(refreshes, [1], 'an up-to-date library needs no refresh')
                finally:
                    httpd.shutdown()
                    httpd.server_close()


class AtomicWriteTests(unittest.TestCase):
    def test_concurrent_readers_never_see_a_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            a, b = b'a' * 2_000_000, b'b' * 3_000_000
            atomic_write_bytes(path, a)
            stop, bad = threading.Event(), []

            def reader():
                while not stop.is_set():
                    data = path.read_bytes()
                    if data not in (a, b):
                        bad.append(len(data))
            t = threading.Thread(target=reader)
            t.start()
            for i in range(60):
                atomic_write_bytes(path, b if i % 2 == 0 else a)
            stop.set()
            t.join()
            self.assertEqual(bad, [], f'torn reads of sizes {bad[:5]}')
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['record.json'])


class CurationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.db = self.base / 'db.sqlite3'
        for i in range(2):
            write_motion(self.base, f'm{i}', i * .1)
        with connect(self.db) as conn:
            b = create_bounty(conn, {'title': 'Walk', 'prompt': 'Walk forward', 'criterion': 'Controlled walking',
                                     'requester_wallet': REQUESTER, 'reward_lamports': 1_000_000, 'target_curations': 1})
            with patch('aura_bounties.verify_bounty_post', return_value={'slot': 1}):
                attach_post_signature(conn, b['id'], '2' * 64)
        self.bounty_id = b['id']

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, curator):
        vote = {'bounty_id': self.bounty_id, 'curator_wallet': curator.address, 'left_id': 'm0', 'right_id': 'm1', 'winner_id': 'm1'}
        with connect(self.db) as conn:
            return signed(conn, curator, 'curate', vote)

    def test_failed_curation_leaves_no_orphan_vote(self):
        with connect(self.db) as conn, patch('aura_bounties._insert_curation', side_effect=RuntimeError('disk full')):
            with self.assertRaises(RuntimeError):
                curate(conn, self.base, self.payload(CURATORS[0]))
        with connect(self.db) as conn:
            self.assertEqual(list_preferences(conn), [])
            self.assertEqual(list_curations(conn, self.bounty_id), [])

    def test_concurrent_curations_cannot_exceed_the_target(self):
        # Both requests pass the unlocked pre-checks together, then race to insert.
        barrier = threading.Barrier(2, timeout=10)
        import aura_bounties
        real_get_motion = aura_bounties.get_motion

        def synced_get_motion(base, mid):
            if mid == 'm1':
                barrier.wait()
            return real_get_motion(base, mid)
        results = []

        payloads = {c.address: self.payload(c) for c in CURATORS}  # signed before the race

        def run(curator):
            conn = connect(self.db)
            try:
                curate(conn, self.base, payloads[curator.address])
                results.append('ok')
            except ValueError as e:
                results.append(str(e))
            finally:
                conn.close()
        with patch('aura_bounties.get_motion', synced_get_motion):
            threads = [threading.Thread(target=run, args=(c,)) for c in CURATORS]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(results.count('ok'), 1, results)
        self.assertIn('reached its target', next(r for r in results if r != 'ok'))
        with connect(self.db) as conn:
            self.assertEqual(len(list_curations(conn, self.bounty_id)), 1)
            self.assertEqual(len(list_preferences(conn)), 1)


if __name__ == '__main__':
    unittest.main()
