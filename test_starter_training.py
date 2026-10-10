import tempfile
from pathlib import Path

from aura_preference import add_preference, connect, preference_diagnostics
from aura_starter_motions import ensure_starter_motions


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        rows = ensure_starter_motions(base, Path(__file__).resolve().parent)
        assert len(rows) >= 4
        assert all((base / row['native_file']).is_file() for row in rows)
        assert all(row.get('preference_group') == 'starter-general-motion-quality-v1' for row in rows)

        with connect(base / 'prefs.sqlite3') as conn:
            record = add_preference(
                conn, base, rows[0]['id'], rows[1]['id'], rows[0]['id'],
                context='Starter/reference comparison: overall motion quality.',
                evaluator_id='test:starter-evaluator',
            )
            assert record['winner_id'] == rows[0]['id']
            diag = preference_diagnostics(conn, base)
            assert diag['valid_comparison_count'] == 1
            assert diag['unique_motion_count'] == 2

    print('starter training integration OK')


if __name__ == '__main__':
    main()
