"""Real, small G1 human-preference learning loop. No implied robot validation.
SQLite persistence + deterministic held-out pairwise logistic regression (NumPy).
"""
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from aura_g1_metrics import canonical_bytes, sha256
from aura_io import atomic_write_text

FEATURES = (
    'near_floor_toe_speed_m_s',
    'estimated_below_floor_fraction',
    'root_horizontal_displacement_m',
    'root_horizontal_path_m',
    'root_height_range_m',
    'duration_s',
)
VERSION = 'aura-pairwise-logistic-v3'


def connect(db):
    conn = sqlite3.connect(db, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT, left_id TEXT NOT NULL, right_id TEXT NOT NULL,
        winner_id TEXT NOT NULL, context TEXT NOT NULL, left_hash TEXT NOT NULL,
        right_hash TEXT NOT NULL, created_at INTEGER NOT NULL,
        evaluator_id TEXT, pair_key TEXT)''')
    cols = {row[1] for row in conn.execute('PRAGMA table_info(preferences)')}
    if 'evaluator_id' not in cols:
        conn.execute('ALTER TABLE preferences ADD COLUMN evaluator_id TEXT')
    if 'pair_key' not in cols:
        conn.execute('ALTER TABLE preferences ADD COLUMN pair_key TEXT')
    conn.execute('''CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT, model_json TEXT NOT NULL, created_at INTEGER NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_pair_key ON preferences(pair_key)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_evaluator ON preferences(evaluator_id)')
    conn.commit()
    return conn


def get_motion(base, motion_id):
    if not motion_id or len(motion_id) > 120 or any(
        x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in motion_id
    ):
        raise ValueError('Invalid motion ID')
    path = base / (motion_id + '.json')
    if not path.is_file():
        raise ValueError('Motion not found')
    m = json.loads(path.read_text())
    e = m.get('kinematic_evaluation')
    if not e:
        from aura_g1_metrics import report_for_npz
        native = (base / m['native_file']).resolve()
        if native.parent != base.resolve() or native.suffix != '.npz' or not native.is_file():
            raise ValueError('No usable G1 motion data')
        e = report_for_npz(native, m['fps'])
        m['kinematic_evaluation'] = e
        atomic_write_text(path, json.dumps(m, indent=2))
    return m


def features(m):
    r = m['kinematic_evaluation']['report']
    if 'g1' not in str(m['model']).lower():
        raise ValueError('Only G1 samples are supported')
    return np.array([float(r[k]) if r.get(k) is not None else 0 for k in FEATURES], dtype=float)


def _pair_key(left_hash: str, right_hash: str) -> str:
    return '|'.join(sorted((str(left_hash), str(right_hash))))


def _comparison_group(m):
    """What two motions must share to be compared: the exact prompt (stored as the motion
    name), or an explicit preference_group for bundled starter/reference clips."""
    return str(m.get('preference_group') or m.get('name') or '').strip()


def split_unit_lookup(base):
    """motion_id -> split unit: its generation batch when known, else its own content hash.

    Holding out whole batches keeps every candidate of one prompt run on the same side,
    since tournament pairs never cross batches."""
    cache = {}

    def unit_of(motion_id):
        if motion_id not in cache:
            m = get_motion(base, motion_id)
            batch = str(m.get('batch_id') or '').strip()
            cache[motion_id] = f'batch:{batch}' if batch else f"motion:{m['kinematic_evaluation']['native_sha256']}"
        return cache[motion_id]
    return unit_of


def motion_level_split(rows, unit_of, *, seed, val_fraction=0.2, min_train=6, min_val=2):
    """Split comparisons so no motion appears in both training and validation.

    Units (batches, or single motions when there are too few batches) are held out in a
    seeded order until validation holds about `val_fraction` of the comparisons. A
    comparison is validation only if both motions are held out and training only if
    neither is; mixed comparisons are dropped from both and counted. When the data is
    too small for an honest split, everything trains and validation is withheld.

    Returns (train_idx, val_idx, info)."""
    units = [(unit_of(r['left_id']), unit_of(r['right_id'])) for r in rows]
    info = {'method': 'motion-level', 'split_unit': 'generation batch',
            'dropped_mixed_count': 0, 'heldout': False, 'reason': None}
    distinct = sorted({u for pair in units for u in pair})
    if len(distinct) < 3:
        # One or two batches can't be split by batch: fall back to single motions.
        units = [(f"motion:{r['left_hash']}", f"motion:{r['right_hash']}") for r in rows]
        distinct = sorted({u for pair in units for u in pair})
        info['split_unit'] = 'motion'
    everything = (list(range(len(rows))), [])
    if len(rows) < 10:
        info['reason'] = 'fewer than 10 valid comparisons'
        return (*everything, info)

    order = list(distinct)
    np.random.default_rng(seed).shuffle(order)
    target = max(min_val, round(val_fraction * len(rows)))
    held = set()
    val_idx = []
    for unit in order:
        held.add(unit)
        val_idx = [i for i, (a, b) in enumerate(units) if a in held and b in held]
        if len(val_idx) >= target:
            break
    train_idx = [i for i, (a, b) in enumerate(units) if a not in held and b not in held]
    if len(val_idx) < min_val or len(train_idx) < min_train:
        info['reason'] = (f'a motion-level split left {len(train_idx)} training and {len(val_idx)} '
                          f'validation comparisons (need {min_train} and {min_val})')
        return (*everything, info)
    info.update(heldout=True, dropped_mixed_count=len(rows) - len(train_idx) - len(val_idx),
                heldout_unit_count=len(held))
    return train_idx, val_idx, info


def _clean_evaluator_id(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if len(value) > 120 or any(
        c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:.' for c in value
    ):
        raise ValueError('Invalid evaluator id')
    return value


def add_preference(conn, base, left_id, right_id, winner_id, context='', evaluator_id=None, *, commit=True):
    """Store one validated vote. commit=False leaves it in the caller's open transaction."""
    if left_id == right_id:
        raise ValueError('Select two different motions')
    if winner_id not in (left_id, right_id):
        raise ValueError('Winner must be one of the two motions')
    if not isinstance(context, str) or not context.strip() or len(context) > 240:
        raise ValueError('A short evaluation context/task is required (max 240 characters)')
    evaluator_id = _clean_evaluator_id(evaluator_id)
    left = get_motion(base, left_id)
    right = get_motion(base, right_id)
    # Generated candidates still compare within the exact same prompt. Bundled
    # starter/reference clips share an explicit preference_group so their votes can
    # teach broad motion quality without pretending they measure prompt compliance.
    left_group = _comparison_group(left)
    right_group = _comparison_group(right)
    if left_group and right_group and left_group != right_group:
        raise ValueError('Aura preferences require two motions from the same comparison group')

    left_hash = left['kinematic_evaluation']['native_sha256']
    right_hash = right['kinematic_evaluation']['native_sha256']
    if left_hash == right_hash:
        raise ValueError('These candidates contain identical motion bytes; generate a fresh unique candidate instead')
    pair_key = _pair_key(left_hash, right_hash)

    # New browser clients send a stable local evaluator id. One evaluator should
    # not be able to inflate training by clicking the same content pair repeatedly.
    if evaluator_id:
        duplicate = conn.execute(
            'SELECT id FROM preferences WHERE evaluator_id=? AND pair_key=? ORDER BY id LIMIT 1',
            (evaluator_id, pair_key),
        ).fetchone()
        if duplicate:
            raise ValueError(
                'You already rated this exact motion pair. The duplicate vote was not added to Aura training data'
            )

    created_at = int(time.time())
    entry = (
        left_id,
        right_id,
        winner_id,
        context.strip(),
        left_hash,
        right_hash,
        created_at,
        evaluator_id,
        pair_key,
    )
    cur = conn.execute(
        'INSERT INTO preferences(left_id,right_id,winner_id,context,left_hash,right_hash,created_at,evaluator_id,pair_key) '
        'VALUES(?,?,?,?,?,?,?,?,?)',
        entry,
    )
    if commit:
        conn.commit()
    record = {
        'id': cur.lastrowid,
        'left_id': left_id,
        'right_id': right_id,
        'winner_id': winner_id,
        'context': context.strip(),
        'left_hash': left_hash,
        'right_hash': right_hash,
        'created_at': created_at,
        'evaluator_id': evaluator_id,
        'pair_key': pair_key,
    }
    return {**record, 'evidence_sha256': sha256(canonical_bytes(record))}


def list_preferences(conn):
    out = []
    for row in conn.execute('SELECT * FROM preferences ORDER BY id DESC'):
        o = dict(row)
        if not o.get('pair_key') and o.get('left_hash') and o.get('right_hash'):
            o['pair_key'] = _pair_key(o['left_hash'], o['right_hash'])
        o['evidence_sha256'] = sha256(canonical_bytes(o))
        out.append(o)
    return out


def validated_preferences(conn, base):
    """Integrity-check and de-duplicate the human preference dataset.

    New rows are de-duplicated per evaluator + content pair. Legacy rows have no
    evaluator id, so repeated legacy votes for the same content pair count once.
    Distinct evaluators may legitimately vote on the same pair.
    """
    rows = list(reversed(list_preferences(conn)))
    valid = []
    seen = set()
    unique_pairs = set()
    unique_motion_hashes = set()
    rejected = []
    duplicate_count = 0
    reasons = {}

    for r in rows:
        reason = None
        try:
            left = get_motion(base, r['left_id'])
            right = get_motion(base, r['right_id'])
            left_hash = left['kinematic_evaluation']['native_sha256']
            right_hash = right['kinematic_evaluation']['native_sha256']
            if left_hash != r['left_hash'] or right_hash != r['right_hash']:
                reason = 'source_motion_changed'
            elif left_hash == right_hash:
                reason = 'identical_motion_bytes'
            else:
                pair_key = _pair_key(left_hash, right_hash)
                evaluator = (r.get('evaluator_id') or '').strip() or 'legacy-unattributed'
                dedupe_key = (evaluator, pair_key)
                if dedupe_key in seen:
                    duplicate_count += 1
                    reasons['duplicate_same_evaluator_pair'] = reasons.get(
                        'duplicate_same_evaluator_pair', 0
                    ) + 1
                    continue
                seen.add(dedupe_key)
                row = dict(r)
                row['pair_key'] = pair_key
                valid.append(row)
                unique_pairs.add(pair_key)
                unique_motion_hashes.update((left_hash, right_hash))
                continue
        except (ValueError, KeyError, OSError, TypeError):
            reason = reason or 'missing_or_unreadable_motion'

        reasons[reason] = reasons.get(reason, 0) + 1
        rejected.append({
            'id': r.get('id'),
            'left_id': r.get('left_id'),
            'right_id': r.get('right_id'),
            'reason': reason,
        })

    diagnostics = {
        'stored_comparison_count': len(rows),
        'valid_comparison_count': len(valid),
        'unique_pair_count': len(unique_pairs),
        'unique_motion_count': len(unique_motion_hashes),
        'duplicate_comparison_count': duplicate_count,
        'rejected_comparison_count': len(rejected),
        'rejection_reasons': reasons,
        'rejected': rejected[:20],
        'reward_min_comparisons': 6,
        'reward_min_motions': 4,
        'reward_trainable': len(valid) >= 6 and len(unique_motion_hashes) >= 4,
    }
    return valid, diagnostics


def preference_diagnostics(conn, base):
    return validated_preferences(conn, base)[1]


def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


def latest(conn):
    row = conn.execute('SELECT model_json FROM models ORDER BY id DESC LIMIT 1').fetchone()
    return json.loads(row[0]) if row else None


def train(conn, base):
    valid, diagnostics = validated_preferences(conn, base)
    if len(valid) < 6:
        raise ValueError(
            f"Aura has {len(valid)} valid unique comparisons of 6 required "
            f"({diagnostics['stored_comparison_count']} stored, "
            f"{diagnostics['duplicate_comparison_count']} duplicates, "
            f"{diagnostics['rejected_comparison_count']} rejected)"
        )

    motions = {}
    for r in valid:
        for mid in (r['left_id'], r['right_id']):
            if mid not in motions:
                motions[mid] = get_motion(base, mid)

    X = np.stack([
        features(motions[r['left_id']]) - features(motions[r['right_id']])
        for r in valid
    ])
    y = np.array([1.0 if r['winner_id'] == r['left_id'] else 0.0 for r in valid])

    train_list, test_list, split = motion_level_split(valid, split_unit_lookup(base), seed=37)
    train_idx = np.array(train_list, dtype=int)
    test_idx = np.array(test_list, dtype=int)

    train_ids = {mid for idx in train_idx for mid in (valid[idx]['left_id'], valid[idx]['right_id'])}
    mat = np.stack([features(motions[mid]) for mid in sorted(train_ids)])
    scale = np.maximum(mat.std(axis=0), 1e-3)
    z = X / scale
    weights = np.zeros(z.shape[1])
    reg = .05
    for _ in range(1000):
        p = sigmoid(z[train_idx] @ weights)
        grad = z[train_idx].T @ (p - y[train_idx]) / len(train_idx) + reg * weights
        weights -= .08 * grad
    train_prob = sigmoid(z[train_idx] @ weights)

    def loss(probs, labels):
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return float(np.mean(-labels * np.log(probs) - (1 - labels) * np.log(1 - probs)))

    source_motion_ids = sorted({mid for r in valid for mid in (r['left_id'], r['right_id'])})
    training_motion_ids = sorted({
        mid for idx in train_idx for mid in (valid[idx]['left_id'], valid[idx]['right_id'])
    })
    validation_motion_ids = sorted({
        mid for idx in test_idx for mid in (valid[idx]['left_id'], valid[idx]['right_id'])
    })
    result = {
        'version': VERSION,
        'training_count': int(len(train_idx)),
        'total_usable_votes': len(valid),
        'unique_pair_count': diagnostics['unique_pair_count'],
        'unique_motion_count': diagnostics['unique_motion_count'],
        'duplicate_comparison_count': diagnostics['duplicate_comparison_count'],
        'rejected_comparison_count': diagnostics['rejected_comparison_count'],
        'heldout_count': int(len(test_idx)),
        'split': split,
        'train_log_loss': round(loss(train_prob, y[train_idx]), 5),
        'heldout_accuracy': None,
        'heldout_log_loss': None,
        'weights': weights.tolist(),
        'scale': scale.tolist(),
        'features': list(FEATURES),
        'source_motion_ids': source_motion_ids,
        'training_motion_ids': training_motion_ids,
        'validation_motion_ids': validation_motion_ids,
        'dataset_sha256': sha256(canonical_bytes(valid)),
        'trained_at': int(time.time()),
        'caveat': 'Only ranks G1 motion candidates from these proxy features for similar tasks; no physical correctness or real-robot safety guarantee.',
    }
    if len(test_idx):
        p = sigmoid(z[test_idx] @ weights)
        result['heldout_accuracy'] = round(float(np.mean((p >= .5) == y[test_idx])), 4)
        result['heldout_log_loss'] = round(loss(p, y[test_idx]), 5)

    conn.execute(
        'INSERT INTO models(model_json,created_at) VALUES(?,?)',
        (json.dumps(result), result['trained_at']),
    )
    conn.commit()
    return result


def rank(conn, base):
    model = latest(conn)
    if not model:
        return {'model': None, 'rankings': []}
    w = np.array(model['weights'])
    scale = np.array(model['scale'])
    motions = []
    for path in base.glob('*.json'):
        if path.name.endswith('.g1.json'):
            continue
        try:
            m = get_motion(base, path.stem)
            if m.get('source') == 'starter_reference':
                continue
            f = features(m)
            score = float(f / scale @ w)
            motions.append({
                'id': m['id'],
                'name': m.get('name', m['id']),
                'ranking_signal': round(score, 4),
                'native_sha256': m['kinematic_evaluation']['native_sha256'],
            })
        except (ValueError, KeyError, OSError, TypeError):
            continue
    motions.sort(key=lambda m: m['ranking_signal'], reverse=True)
    return {'model': model, 'rankings': motions}


def next_pair(conn, base):
    """Select closest predicted pair not yet compared; heuristic active sampling."""
    r = rank(conn, base)
    candidates = r['rankings']
    compared = {frozenset((x['left_id'], x['right_id'])) for x in list_preferences(conn)}
    # Only suggest pairs add_preference would accept: same prompt / comparison group.
    group = {}
    for c in candidates:
        try:
            group[c['id']] = _comparison_group(get_motion(base, c['id']))
        except (ValueError, KeyError, OSError):
            group[c['id']] = None
    best = None
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if frozenset((a['id'], b['id'])) in compared:
                continue
            if group[a['id']] is None or group[a['id']] != group[b['id']]:
                continue
            diff = abs(a['ranking_signal'] - b['ranking_signal'])
            if best is None or diff < best[0]:
                best = (diff, a['id'], b['id'])
    return {
        'pair': [best[1], best[2]] if best else [],
        'method': 'closest unreviewed learned signal' if r['model'] else 'none',
    }
