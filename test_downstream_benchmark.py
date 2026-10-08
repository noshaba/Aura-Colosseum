import json
import shutil
import tempfile
from pathlib import Path

from aura_preference import add_preference, connect, get_motion, train
from aura_downstream_benchmark import list_cohorts, run_benchmark

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / 'text2motion-aura/text2motion_aura/assets/demo/examples/text2motion-aura-g1-rp').glob('*/motion.npz'))


def build_fixture(base: Path):
    assert len(EXAMPLES) >= 8
    prompt = 'fixture: same prompt G1 walking candidate'
    ids = []
    for i, source in enumerate(EXAMPLES[:8]):
        mid = f'fixture{i:02d}'
        native = base / f'{mid}.npz'
        shutil.copy2(source, native)
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
        # Preferences touch only first six motions; last two are downstream-unseen.
        pairs = [(0,1),(2,3),(4,5),(0,2),(1,3),(2,4),(3,5),(0,4),(1,5),(0,3),(1,4),(2,5)]
        for a, b in pairs:
            left, right = ids[a], ids[b]
            winner = left if proxy[left] <= proxy[right] else right
            add_preference(conn, base, left, right, winner, 'fixture comparison')
        model = train(conn, base)
        assert model['total_usable_votes'] == 12
        assert set(ids[6:]).isdisjoint(model['source_motion_ids'])
    return prompt, db


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        prompt, db = build_fixture(base)
        cohorts = list_cohorts(base)
        assert cohorts and cohorts[0]['count'] == 8
        a = run_benchmark(base, db, cohort_name=prompt, train_count=4, holdout_count=2,
                          random_trials=8, samples_per_motion=60, horizon_s=0.4, seed=41)
        b = run_benchmark(base, db, cohort_name=prompt, train_count=4, holdout_count=2,
                          random_trials=8, samples_per_motion=60, horizon_s=0.4, seed=41)
        assert a['report_sha256'] == b['report_sha256']
        assert len(a['aura_selected_ids']) == 4 and len(a['heldout_ids']) == 2
        assert not set(a['aura_selected_ids']) & set(a['heldout_ids'])
        for metric in ('one_step_mpjpe_cm','rollout_mpjpe_cm','rollout_root_error_cm'):
            assert a['comparison'][metric]['aura'] >= 0
            assert a['comparison'][metric]['random_mean'] >= 0
        print('downstream benchmark test: PASS')
        print(a['report_sha256'])
        print(json.dumps(a['comparison'], indent=2))

if __name__ == '__main__':
    main()
