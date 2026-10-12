"""Seed policy and duplicate-output handling in AuraText2MotionGenerator.generate().

Runs without PyTorch or a GPU: the Kimodo model, torch and the text2motion_aura
helpers are replaced by stubs, and the fake model's output depends only on the seed
it was given, so the tests control exactly when two candidates collide.
"""
import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from aura_text2motion_generator import AuraText2MotionGenerator


class FakeKimodo:
    """Returns a motion determined by the current seed; `collide` maps seeds to a shared output."""
    fps = 30

    def __init__(self):
        self.seed = None
        self.calls = []
        self.collide = lambda seed: seed

    def __call__(self, texts, num_frames, **kwargs):
        key = self.collide(self.seed)
        self.calls.append(self.seed)
        frames = sum(num_frames)
        pos = np.zeros((1, frames, 34, 3), dtype=np.float32)
        pos[..., 1] = 1.0
        pos[0, :, 0, 2] = np.linspace(0, 1 + (key % 997) / 997, frames)
        rot = np.tile(np.eye(3, dtype=np.float32), (1, frames, 34, 1, 1))
        return {'posed_joints': pos, 'global_rot_mats': rot}


class GeneratorUniquenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = Path(self.tmp.name)
        self.gen = AuraText2MotionGenerator(self.library)
        self.model = FakeKimodo()

        def seed_everything(seed):
            self.model.seed = seed

        stubs = {
            'torch': types.SimpleNamespace(inference_mode=contextlib.nullcontext),
            'text2motion_aura': types.ModuleType('text2motion_aura'),
            'text2motion_aura.constraints': types.SimpleNamespace(load_constraints_lst=lambda *a, **k: []),
            'text2motion_aura.tools': types.SimpleNamespace(seed_everything=seed_everything),
        }
        self.patches = [
            mock.patch.dict(sys.modules, stubs),
            mock.patch.object(self.gen, '_ensure_model', return_value=self.model),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def generate(self, **kwargs):
        return self.gen.generate(prompt='walk forward', duration=1.0, diffusion_steps=2, **kwargs)

    def library_files(self, suffix):
        return sorted(p.name for p in self.library.iterdir() if p.name.endswith(suffix) and not p.name.endswith('.g1.json'))

    def test_each_batch_gets_a_fresh_random_seed(self):
        first = self.generate(count=3)['generation']
        second = self.generate(count=3)['generation']
        self.assertEqual(first['seed_policy'], 'fresh_random_per_batch')
        self.assertNotEqual(first['seed'], second['seed'])
        self.assertFalse(set(first['candidate_seeds']) & set(second['candidate_seeds']))
        hashes = first['candidate_sha256'] + second['candidate_sha256']
        self.assertEqual(len(set(hashes)), 6)

    def test_explicit_seed_is_reproducible(self):
        out = self.generate(count=3, seed=1000)['generation']
        self.assertEqual(out['seed_policy'], 'explicit')
        self.assertEqual(out['candidate_seeds'], [1000, 1001, 1002])

    def test_duplicate_candidate_is_discarded_and_regenerated(self):
        # Seeds 1000 and 1001 produce the same motion: candidate 2 must be retried.
        self.model.collide = lambda seed: 1000 if seed in (1000, 1001) else seed
        out = self.generate(count=2, seed=1000)
        seeds = out['generation']['candidate_seeds']
        self.assertEqual(seeds[0], 1000)
        self.assertNotIn(seeds[1], (1000, 1001))  # walked away from the colliding seed
        self.assertEqual(len(set(out['generation']['candidate_sha256'])), 2)
        self.assertEqual(self.model.calls[:2], [1000, 1001])
        # The rejected attempt left no files behind: exactly one record per accepted candidate.
        self.assertEqual(len(self.library_files('.npz')), 2)
        self.assertEqual(len(self.library_files('.json')), 2)

    def test_motion_already_in_the_library_is_not_published_again(self):
        self.generate(count=2, seed=1000)
        out = self.generate(count=2, seed=1000)  # same seeds would reproduce the same bytes
        self.assertEqual(len(self.library_files('.npz')), 4)
        all_bytes = {p.read_bytes() for p in self.library.glob('*.npz')}
        self.assertEqual(len(all_bytes), 4)
        self.assertNotIn(1000, out['generation']['candidate_seeds'])

    def test_gives_up_after_max_attempts_without_leaking_files_or_the_lock(self):
        self.model.collide = lambda seed: 7  # every seed produces the same motion
        with self.assertRaisesRegex(RuntimeError, 'unique candidate 2/2'):
            self.generate(count=2)
        self.assertEqual(len(self.library_files('.npz')), 1)  # only the accepted first candidate
        self.assertFalse(self.gen.status()['busy'])
        self.assertTrue(self.gen._lock.acquire(blocking=False))  # lock was released
        self.gen._lock.release()


if __name__ == '__main__':
    unittest.main()
