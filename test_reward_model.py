import json
import tempfile
from pathlib import Path

import numpy as np

from aura_preference import connect, add_preference
from aura_reward_model import (
    VERSION, pair_probability, score_all, status, train_reward_model,
    trajectory_sequence,
)


def make_motion(base: Path, mid: str, quality: float):
    t = 42
    fps = 30.0
    p = np.zeros((t, 34, 3), dtype=np.float32)
    p[:, :, 1] = np.linspace(0.05, 1.05, 34)[None, :]
    phase = np.linspace(0, 2 * np.pi, t, dtype=np.float32)
    p[:, :, 0] = (0.01 + 0.002 * quality) * np.sin(phase)[:, None]
    p[:, :, 2] = np.linspace(0, 0.30 + 0.07 * quality, t)[:, None]
    p[:, 7, 1] = 0.025 + 0.003 * np.sin(phase)
    p[:, 14, 1] = 0.025 + 0.003 * np.cos(phase)
    rot = np.tile(np.eye(3, dtype=np.float32), (t, 34, 1, 1))
    np.savez_compressed(base / f'{mid}.npz', posed_joints=p, global_rot_mats=rot)
    (base / f'{mid}.json').write_text(json.dumps({
        'id': mid, 'name': 'same prompt', 'model': 'G1', 'fps': fps,
        'frames': t, 'native_file': f'{mid}.npz', 'preview_file': None,
    }))
    return p, rot


def main():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        made = {}
        for i in range(4):
            made[f'm{i}'] = make_motion(base, f'm{i}', float(i))
        seq = trajectory_sequence(made['m0'][0], 30.0, made['m0'][1])
        assert seq.shape[0] == 64
        assert seq.shape[1] > 400

        votes = [
            ('m0', 'm1', 'm1'), ('m0', 'm2', 'm2'), ('m0', 'm3', 'm3'),
            ('m1', 'm2', 'm2'), ('m1', 'm3', 'm3'), ('m2', 'm3', 'm3'),
        ]
        with connect(base / 'prefs.sqlite3') as conn:
            for a, b, w in votes:
                add_preference(conn, base, a, b, w, 'same prompt')
            result = train_reward_model(conn, base, epochs=6)

        assert result['model']['version'] == VERSION
        assert result['model']['training_paradigm'].startswith('human-preference reward modeling')
        assert result['model']['parameter_count'] > 10000
        assert len(result['scores']) == 4
        assert sorted(x['rank'] for x in result['scores']) == [1, 2, 3, 4]
        prob = pair_probability(base, 'm0', 'm3')
        assert prob is not None and 0 <= prob <= 1
        st = status(base)
        assert st['model']['model_sha256'] == result['model']['model_sha256']
        assert len(score_all(base)) == 4
        print('reward model test passed', [(x['id'], x['reward']) for x in result['scores']])


if __name__ == '__main__':
    main()
