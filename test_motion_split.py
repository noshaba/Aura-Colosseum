"""Motion-level train/validation split (aura_preference.motion_level_split) and the
same-prompt filter in next_pair.

The split must never put one motion on both sides: in winner-vs-next tournaments a
motion appears in many pairs, so a pair-level split leaks it into validation.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aura_g1_metrics import report_for_npz
from aura_preference import add_preference, connect, motion_level_split, next_pair, train


def tournament_rows(batches, per_batch, rng):
    """Winner-vs-next comparisons inside each batch, like the hero arena produces."""
    rows = []
    for b in range(batches):
        ids = [f'b{b}m{k}' for k in range(per_batch)]
        winner = ids[0]
        for challenger in ids[1:]:
            rows.append({'left_id': winner, 'right_id': challenger, 'left_hash': f'h-{winner}', 'right_hash': f'h-{challenger}'})
            if rng.random() < .5:
                winner = challenger
    return rows


def batch_of(motion_id):
    return 'batch:' + motion_id.split('m')[0]


def motions_of(rows, idx):
    return {m for i in idx for m in (rows[i]['left_id'], rows[i]['right_id'])}


class MotionLevelSplitTests(unittest.TestCase):
    def test_no_motion_is_on_both_sides(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            rows = tournament_rows(int(rng.integers(1, 8)), int(rng.integers(2, 7)), rng)
            unit = batch_of if trial % 2 else (lambda m: 'motion:' + m)
            train_idx, val_idx, info = motion_level_split(rows, unit, seed=trial)
            self.assertFalse(motions_of(rows, train_idx) & motions_of(rows, val_idx))
            self.assertFalse(set(train_idx) & set(val_idx))
            self.assertEqual(len(train_idx) + len(val_idx) + info['dropped_mixed_count'], len(rows))

    def test_holds_out_whole_batches(self):
        rows = tournament_rows(6, 4, np.random.default_rng(1))  # 18 comparisons, 6 batches
        train_idx, val_idx, info = motion_level_split(rows, batch_of, seed=3)
        self.assertTrue(info['heldout'])
        self.assertEqual(info['split_unit'], 'generation batch')
        self.assertEqual(info['dropped_mixed_count'], 0)  # pairs never cross batches
        held = {batch_of(m) for m in motions_of(rows, val_idx)}
        self.assertFalse(held & {batch_of(m) for m in motions_of(rows, train_idx)})

    def test_falls_back_to_single_motions_when_there_are_too_few_batches(self):
        rows = tournament_rows(1, 20, np.random.default_rng(2))
        _, _, info = motion_level_split(rows, batch_of, seed=0)
        self.assertEqual(info['split_unit'], 'motion')

    def test_mixed_pairs_are_dropped_and_counted(self):
        # All pairs among 8 motions: holding out any motion makes its pairs with the rest "mixed".
        ids = [f'm{k}' for k in range(8)]
        rows = [{'left_id': a, 'right_id': b, 'left_hash': a, 'right_hash': b}
                for i, a in enumerate(ids) for b in ids[i + 1:]]
        train_idx, val_idx, info = motion_level_split(rows, lambda m: 'motion:' + m, seed=5)
        self.assertTrue(info['heldout'])
        self.assertGreater(info['dropped_mixed_count'], 0)
        self.assertFalse(motions_of(rows, train_idx) & motions_of(rows, val_idx))

    def test_small_data_withholds_validation_instead_of_leaking(self):
        rows = tournament_rows(1, 6, np.random.default_rng(3))  # 5 comparisons
        train_idx, val_idx, info = motion_level_split(rows, batch_of, seed=0)
        self.assertEqual((len(train_idx), val_idx, info['heldout']), (5, [], False))
        self.assertIn('fewer than 10', info['reason'])


def write_motion(base, motion_id, offset, prompt):
    pos = np.zeros((40, 34, 3))
    pos[:, :, 1] = 1.0
    pos[:, 7, 1] = pos[:, 14, 1] = .15
    pos[:, 0, 2] = np.linspace(0, .2 + offset, 40)
    pos[:, 7, 0] = offset * .05 * np.arange(40) / 40
    npz = base / f'{motion_id}.npz'
    np.savez_compressed(npz, posed_joints=pos)
    meta = {'id': motion_id, 'name': prompt, 'model': 'g1', 'fps': 30, 'native_file': npz.name,
            'kinematic_evaluation': report_for_npz(npz, 30)}
    (base / f'{motion_id}.json').write_text(json.dumps(meta))
    return motion_id


class NextPairTests(unittest.TestCase):
    def test_suggested_pair_is_always_same_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            conn = connect(base / 'p.db')
            walk = [write_motion(base, f'walk{k}', k * .1, 'walk forward') for k in range(5)]
            # Cartwheel motions sit right between the walks in feature space, so the
            # closest-signal pair would cross prompts without the filter.
            [write_motion(base, f'cart{k}', k * .1 + .05, 'do a cartwheel') for k in range(5)]
            for i in range(4):
                for j in range(i + 1, 4):
                    add_preference(conn, base, walk[i], walk[j], walk[i], 'walk forward')
            train(conn, base)
            pair = next_pair(conn, base)['pair']
            self.assertEqual(len(pair), 2)
            names = {json.loads((base / f'{m}.json').read_text())['name'] for m in pair}
            self.assertEqual(len(names), 1, f'next_pair crossed prompts: {pair}')
            conn.close()


if __name__ == '__main__':
    unittest.main()
