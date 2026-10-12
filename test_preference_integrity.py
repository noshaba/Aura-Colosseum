"""Integrity rules for the human-preference dataset (aura_preference.py).

Covers what keeps training data honest: one vote per evaluator per content pair
(order-independent), same-prompt pairs only, no identical-bytes pairs, and rows whose
source motion changed after the vote are rejected at training time.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aura_g1_metrics import report_for_npz
from aura_preference import add_preference, connect, list_preferences, validated_preferences


def _write_motion(base: Path, motion_id: str, offset: float, prompt: str = 'walk forward') -> str:
    """A tiny G1 motion whose bytes depend on `offset`, labelled with `prompt`."""
    pos = np.zeros((40, 34, 3))
    pos[:, :, 1] = 1.0
    pos[:, 7, 1] = pos[:, 14, 1] = .15
    pos[:, 0, 2] = np.linspace(0, .2 + offset, 40)
    npz = base / f'{motion_id}.npz'
    np.savez_compressed(npz, posed_joints=pos)
    meta = {'id': motion_id, 'name': prompt, 'model': 'g1', 'fps': 30, 'native_file': npz.name,
            'kinematic_evaluation': report_for_npz(npz, 30)}
    (base / f'{motion_id}.json').write_text(json.dumps(meta))
    return motion_id


class PreferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.conn = connect(self.base / 'preferences.db')
        self.a = _write_motion(self.base, 'motionA', .0)
        self.b = _write_motion(self.base, 'motionB', .1)
        self.c = _write_motion(self.base, 'motionC', .2)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def vote(self, left, right, winner, evaluator='eval-1'):
        return add_preference(self.conn, self.base, left, right, winner, 'walk forward', evaluator)

    def test_same_evaluator_cannot_vote_twice_on_a_pair(self):
        self.vote(self.a, self.b, self.a)
        with self.assertRaisesRegex(ValueError, 'already rated this exact motion pair'):
            self.vote(self.a, self.b, self.b)
        self.assertEqual(len(list_preferences(self.conn)), 1)

    def test_pair_key_ignores_left_right_order(self):
        first = self.vote(self.a, self.b, self.a)
        with self.assertRaisesRegex(ValueError, 'already rated'):
            self.vote(self.b, self.a, self.a)  # same pair, sides swapped
        self.assertEqual(first['pair_key'], '|'.join(sorted((first['left_hash'], first['right_hash']))))

    def test_distinct_evaluators_may_rate_the_same_pair(self):
        self.vote(self.a, self.b, self.a, 'eval-1')
        self.vote(self.a, self.b, self.b, 'eval-2')
        valid, diag = validated_preferences(self.conn, self.base)
        self.assertEqual(len(valid), 2)
        self.assertEqual(diag['unique_pair_count'], 1)
        self.assertEqual(diag['duplicate_comparison_count'], 0)

    def test_legacy_unattributed_duplicates_count_once(self):
        for _ in range(3):  # no evaluator id: the insert guard can't help, validation must
            add_preference(self.conn, self.base, self.a, self.b, self.a, 'walk forward')
        valid, diag = validated_preferences(self.conn, self.base)
        self.assertEqual(len(valid), 1)
        self.assertEqual(diag['duplicate_comparison_count'], 2)
        self.assertEqual(diag['rejection_reasons'], {'duplicate_same_evaluator_pair': 2})

    def test_cross_prompt_pair_is_rejected(self):
        other = _write_motion(self.base, 'motionOther', .3, prompt='do a cartwheel')
        with self.assertRaisesRegex(ValueError, 'same comparison group'):
            self.vote(self.a, other, self.a)
        self.assertEqual(list_preferences(self.conn), [])

    def test_identical_motion_bytes_are_rejected(self):
        clone = 'motionClone'
        (self.base / f'{clone}.npz').write_bytes((self.base / f'{self.a}.npz').read_bytes())
        meta = json.loads((self.base / f'{self.a}.json').read_text())
        meta.update(id=clone, native_file=f'{clone}.npz')
        (self.base / f'{clone}.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'identical motion bytes'):
            self.vote(self.a, clone, self.a)

    def test_vote_on_a_motion_that_changed_afterwards_is_rejected(self):
        self.vote(self.a, self.b, self.a)
        self.vote(self.a, self.c, self.c)
        _write_motion(self.base, self.b, .9)  # B's bytes change after it was rated
        valid, diag = validated_preferences(self.conn, self.base)
        self.assertEqual([(r['left_id'], r['right_id']) for r in valid], [(self.a, self.c)])
        self.assertEqual(diag['rejection_reasons'], {'source_motion_changed': 1})

    def test_vote_on_a_deleted_motion_is_rejected(self):
        self.vote(self.a, self.b, self.a)
        (self.base / f'{self.b}.json').unlink()
        valid, diag = validated_preferences(self.conn, self.base)
        self.assertEqual(valid, [])
        self.assertEqual(diag['rejection_reasons'], {'missing_or_unreadable_motion': 1})

    def test_invalid_evaluator_id_is_rejected(self):
        for bad in ('has space', 'x' * 121, 'semi;colon'):
            with self.assertRaisesRegex(ValueError, 'Invalid evaluator id'):
                self.vote(self.a, self.b, self.a, bad)
        self.assertEqual(list_preferences(self.conn), [])


if __name__ == '__main__':
    unittest.main()
