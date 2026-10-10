import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


class RewardApiTest(unittest.TestCase):
    def test_reward_http(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for i in range(4):
                t = 34
                p = np.zeros((t, 34, 3), dtype=np.float32)
                p[:, :, 1] = np.linspace(.05, 1.0, 34)[None, :]
                p[:, :, 2] = np.linspace(0, .20 + .07 * i, t)[:, None]
                p[:, 7, 1] = .02
                p[:, 14, 1] = .02
                rot = np.tile(np.eye(3, dtype=np.float32), (t, 34, 1, 1))
                np.savez_compressed(base / f'm{i}.npz', posed_joints=p, global_rot_mats=rot)
                (base / f'm{i}.json').write_text(json.dumps({
                    'id': f'm{i}', 'name': 'same prompt', 'model': 'G1', 'fps': 30,
                    'frames': t, 'native_file': f'm{i}.npz', 'preview_file': None,
                }))
            env = {**os.environ, 'AURA_MOTION_LIBRARY': td, 'AURA_LIBRARY_PORT': '8878'}
            proc = subprocess.Popen(
                [sys.executable, 'aura_library_server.py'], cwd=Path(__file__).parent,
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            try:
                for _ in range(80):
                    try:
                        urlopen('http://127.0.0.1:8878/reward', timeout=.5).read()
                        break
                    except Exception:
                        time.sleep(.1)
                else:
                    self.fail('server failed to start')

                def post(path, obj, timeout=30):
                    req = Request(
                        'http://127.0.0.1:8878' + path,
                        data=json.dumps(obj).encode(),
                        headers={'Content-Type': 'application/json'},
                    )
                    with urlopen(req, timeout=timeout) as r:
                        return json.load(r)

                votes = [
                    ('m0', 'm1', 'm1'), ('m0', 'm2', 'm2'), ('m0', 'm3', 'm3'),
                    ('m1', 'm2', 'm2'), ('m1', 'm3', 'm3'), ('m2', 'm3', 'm3'),
                ]
                for a, b, w in votes:
                    post('/preferences', {'left_id': a, 'right_id': b, 'winner_id': w, 'context': 'same prompt'})
                trained = post('/reward/train', {'epochs': 4})
                self.assertEqual(trained['model']['motion_count'], 4)
                self.assertEqual(len(trained['scores']), 4)
                with urlopen('http://127.0.0.1:8878/reward') as r:
                    st = json.load(r)
                self.assertEqual(st['model']['version'], 'aura-motion-reward-transformer-v1')
            finally:
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
