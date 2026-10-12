import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from aura_preference import add_preference, connect, get_motion, train
from aura_downstream_benchmark import BASELINES, MIN_HOLDOUT, list_cohorts, run_benchmark

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / 'text2motion-aura/text2motion_aura/assets/demo/examples/text2motion-aura-g1-rp').glob('*/motion.npz'))
MIRROR = np.diag([-1.0, 1.0, 1.0])  # reflect across the sagittal (x = 0) plane


def _write_mirrored(source: Path, target: Path):
    """A distinct but realistic motion: the example reflected left/right (rotations too)."""
    with np.load(source) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['posed_joints'] = arrays['posed_joints'] * np.array([-1.0, 1.0, 1.0])
    if 'global_rot_mats' in arrays:
        arrays['global_rot_mats'] = MIRROR @ arrays['global_rot_mats'] @ MIRROR
    np.savez(target, **arrays)


def build_fixture(base: Path):
    assert len(EXAMPLES) >= 8
    prompt = 'fixture: same prompt G1 walking candidate'
    ids = []
    for i in range(16):  # 8 bundled examples + their 8 mirror images
        mid = f'fixture{i:02d}'
        native = base / f'{mid}.npz'
        if i < 8:
            shutil.copy2(EXAMPLES[i], native)
        else:
            _write_mirrored(EXAMPLES[i - 8], native)
        record = {
            'id': mid, 'name': prompt, 'model': 'text2motion-aura-g1', 'sample': 'fixture',
            'created_at': '2026-10-06T00:00:00Z', 'frames': 150, 'fps': 30.0,
            'native_file': native.name, 'preview_file': None,
        }
        (base / f'{mid}.json').write_text(json.dumps(record))
        ids.append(mid)
    db = base / 'aura-preferences.sqlite3'
    with connect(db) as conn:
        proxy = {}
        for mid in ids:
            report = get_motion(base, mid)['kinematic_evaluation']['report']
            toe = report['near_floor_toe_speed_m_s']
            proxy[mid] = (toe if toe is not None else 9.0) + 0.2 * report['root_height_range_m']
        # Preferences touch only the first eight motions; the other eight are downstream-unseen.
        pairs = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        for a, b in pairs:
            left, right = ids[a], ids[b]
            winner = left if proxy[left] <= proxy[right] else right
            add_preference(conn, base, left, right, winner, 'fixture comparison')
        model = train(conn, base)
        assert model['total_usable_votes'] == 12
        assert set(ids[8:]).isdisjoint(model['source_motion_ids'])
    return prompt, db, ids


def check_report(a, ids, selector):
    assert a['selector'] == selector
    assert len(a['aura_selected_ids']) == 6 and len(a['heldout_ids']) == MIN_HOLDOUT
    assert set(a['heldout_ids']) <= set(ids[8:]), 'held-out motions must be unseen by the selector'
    assert set(a['selectors']) == {selector, *BASELINES}
    for name, entry in a['selectors'].items():
        assert not set(entry['selected_ids']) & set(a['heldout_ids'])
        for metric, v in entry['vs_random'].items():
            lo, hi = v['ci95_cm']
            assert lo <= v['error_reduction_cm'] <= hi, (name, metric, v)
            assert v['verdict'] in ('better than random', 'worse than random', 'no evidence of a difference')
    for metric in ('one_step_mpjpe_cm', 'rollout_mpjpe_cm', 'rollout_root_error_cm'):
        c = a['comparison'][metric]
        assert c['aura'] >= 0 and c['random_mean'] >= 0 and len(c['ci95_cm']) == 2


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        prompt, db, ids = build_fixture(base)
        cohorts = list_cohorts(base)
        assert cohorts and cohorts[0]['count'] == 16
        kwargs = dict(cohort_name=prompt, train_count=6, holdout_count=6, random_trials=8,
                      samples_per_motion=60, horizon_s=0.4, seed=41)

        a = run_benchmark(base, db, selector='logistic', **kwargs)
        b = run_benchmark(base, db, selector='logistic', **kwargs)
        assert a['report_sha256'] == b['report_sha256'], 'benchmark must be reproducible'
        check_report(a, ids, 'logistic')

        # Too few held-out motions can't support an interval: rejected up front.
        try:
            run_benchmark(base, db, selector='logistic', **{**kwargs, 'holdout_count': 2})
            raise AssertionError('holdout_count=2 should be rejected')
        except ValueError as e:
            assert 'holdout_count' in str(e)

        # Default selector is the Transformer reward model: it refuses to run untrained...
        try:
            run_benchmark(base, db, **kwargs)
            raise AssertionError('an untrained reward model should be rejected')
        except ValueError as e:
            assert 'reward model' in str(e)
        # ...and runs once trained on the same comparisons.
        from aura_reward_model import train_reward_model
        with connect(db) as conn:
            meta = train_reward_model(conn, base, epochs=8)['model']
        assert set(ids[8:]).isdisjoint(meta['source_motion_ids'])
        t = run_benchmark(base, db, **kwargs)
        check_report(t, ids, 'reward_transformer')

        print('downstream benchmark test: PASS')
        for name, entry in t['selectors'].items():
            v = entry['vs_random']['rollout_mpjpe_cm']
            print(f"  {name:20s} rollout error vs random: {v['error_reduction_cm']:+.3f} cm "
                  f"[{v['ci95_cm'][0]:+.3f}, {v['ci95_cm'][1]:+.3f}] -> {v['verdict']}")


if __name__ == '__main__':
    main()
