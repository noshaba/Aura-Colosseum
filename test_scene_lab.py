import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aura_scene_lab import DEFAULT_SCENE, fit_motion_to_scene, plan_path
from aura_g1_preview import g1_preview_bytes


class SceneLabTests(unittest.TestCase):
    def test_path_avoids_obstacle(self):
        plan = plan_path(DEFAULT_SCENE)
        self.assertGreater(plan['path_length_m'], 0.1)
        self.assertGreaterEqual(len(plan['path']), 3)
        obstacle = plan['scene']['obstacles'][0]
        radius = plan['scene']['robot_radius']
        for x, z in plan['path']:
            self.assertFalse(
                abs(x - obstacle['x']) <= obstacle['width'] / 2 + radius and
                abs(z - obstacle['z']) <= obstacle['depth'] / 2 + radius
            )

    def test_fit_rewrites_root_to_scene_path(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            frames, joints = 20, 34
            posed = np.zeros((frames, joints, 3), dtype=np.float32)
            for i in range(frames):
                posed[i, :, 2] = i / (frames - 1)
                posed[i, :, 1] = 0.7
                posed[i, 1:, 1] += np.linspace(-0.65, 0.65, joints - 1, dtype=np.float32)
            global_rot = np.tile(np.eye(3, dtype=np.float32), (frames, joints, 1, 1))
            local_rot = global_rot.copy()
            uid = 'abc123'
            np.savez(base / f'{uid}.npz', posed_joints=posed, global_rot_mats=global_rot,
                     root_positions=posed[:, 0], local_rot_mats=local_rot)
            (base / f'{uid}.g1.json').write_bytes(g1_preview_bytes(posed, 30, global_rot))
            item = {
                'id': uid, 'name': 'walk', 'model': 'nvidia-kimodo-g1', 'fps': 30,
                'native_file': f'{uid}.npz', 'preview_file': f'{uid}.g1.json',
                'native_sha256': 'before',
            }
            (base / f'{uid}.json').write_text(json.dumps(item))
            plan = plan_path(DEFAULT_SCENE)
            fitted = fit_motion_to_scene(base, item, plan)
            with np.load(base / f'{uid}.npz', allow_pickle=False) as d:
                root = d['root_positions']
            self.assertAlmostEqual(float(root[0, 0]), DEFAULT_SCENE['start'][0], places=3)
            self.assertAlmostEqual(float(root[0, 2]), DEFAULT_SCENE['start'][1], places=3)
            self.assertAlmostEqual(float(root[-1, 0]), DEFAULT_SCENE['goal'][0], places=3)
            self.assertAlmostEqual(float(root[-1, 2]), DEFAULT_SCENE['goal'][1], places=3)
            self.assertEqual(fitted['scene_fit']['method'], 'aura-scene-root-path-fit-v1')
            self.assertNotEqual(fitted['native_sha256'], 'before')


if __name__ == '__main__':
    unittest.main()
