"""Real, small G1 human-preference learning loop. No implied robot validation.
SQLite persistence + deterministic held-out pairwise logistic regression (NumPy).
"""
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
import numpy as np
from aura_g1_metrics import canonical_bytes, sha256

FEATURES = ('near_floor_toe_speed_m_s', 'estimated_below_floor_fraction',
            'root_horizontal_displacement_m', 'root_horizontal_path_m', 'root_height_range_m', 'duration_s')
VERSION = 'aura-pairwise-logistic-v2'

def connect(db):
    conn = sqlite3.connect(db, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT, left_id TEXT NOT NULL, right_id TEXT NOT NULL,
        winner_id TEXT NOT NULL, context TEXT NOT NULL, left_hash TEXT NOT NULL,
        right_hash TEXT NOT NULL, created_at INTEGER NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT, model_json TEXT NOT NULL, created_at INTEGER NOT NULL)''')
    conn.commit()
    return conn

def get_motion(base, motion_id):
    if not motion_id or len(motion_id) > 120 or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in motion_id):
        raise ValueError('Invalid motion ID')
    path = base / (motion_id + '.json')
    if not path.is_file(): raise ValueError('Motion not found')
    m = json.loads(path.read_text())
    e = m.get('kinematic_evaluation')
    if not e:
        from aura_g1_metrics import report_for_npz
        native = (base / m['native_file']).resolve()
        if native.parent != base.resolve() or native.suffix != '.npz' or not native.is_file():
            raise ValueError('No usable G1 motion data')
        e = report_for_npz(native, m['fps'])
        m['kinematic_evaluation'] = e
        path.write_text(json.dumps(m, indent=2))
    return m

def features(m):
    r = m['kinematic_evaluation']['report']
    if 'g1' not in str(m['model']).lower(): raise ValueError('Only G1 samples are supported')
    return np.array([float(r[k]) if r.get(k) is not None else 0 for k in FEATURES], dtype=float)

def add_preference(conn, base, left_id, right_id, winner_id, context=''):
    if left_id == right_id: raise ValueError('Select two different motions')
    if winner_id not in (left_id, right_id): raise ValueError('Winner must be one of the two motions')
    if not isinstance(context, str) or not context.strip() or len(context) > 240:
        raise ValueError('A short evaluation context/task is required (max 240 characters)')
    left = get_motion(base, left_id); right = get_motion(base, right_id)
    left_prompt = str(left.get('name') or '').strip()
    right_prompt = str(right.get('name') or '').strip()
    if left_prompt and right_prompt and left_prompt != right_prompt:
        raise ValueError('Aura discovery requires two motions generated from the same exact prompt')
    # A single context captures what the user is judging; both motions must be for the same generation task.
    entry = (left_id, right_id, winner_id, context.strip(),
             left['kinematic_evaluation']['native_sha256'],
             right['kinematic_evaluation']['native_sha256'], int(time.time()))
    cur = conn.execute('INSERT INTO preferences(left_id,right_id,winner_id,context,left_hash,right_hash,created_at) VALUES(?,?,?,?,?,?,?)',entry)
    conn.commit()
    record = {'id':cur.lastrowid, 'left_id':left_id, 'right_id':right_id, 'winner_id':winner_id,
              'context':context.strip(), 'left_hash':entry[4], 'right_hash':entry[5], 'created_at':entry[6]}
    return {**record, 'evidence_sha256':sha256(canonical_bytes(record))}

def list_preferences(conn):
    out=[]
    for row in conn.execute('SELECT * FROM preferences ORDER BY id DESC'):
        o=dict(row); o['evidence_sha256']=sha256(canonical_bytes(o)); out.append(o)
    return out

def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))

def latest(conn):
    row=conn.execute('SELECT model_json FROM models ORDER BY id DESC LIMIT 1').fetchone()
    return json.loads(row[0]) if row else None

def train(conn, base):
    rows = list(reversed(list_preferences(conn)))
    if len(rows) < 6: raise ValueError('Collect at least six comparisons before training')
    motions = {}
    valid=[]
    for r in rows:
        try:
            for mid, digest_key in ((r['left_id'],'left_hash'), (r['right_id'],'right_hash')):
                if mid not in motions: motions[mid] = get_motion(base, mid)
                if motions[mid]['kinematic_evaluation']['native_sha256'] != r[digest_key]:
                    raise ValueError('Source motion has changed')
            valid.append(r)
        except (ValueError, KeyError, OSError): continue
    if len(valid) < 6: raise ValueError('At least six comparisons must reference unchanged G1 motion files')
    X = np.stack([features(motions[r['left_id']]) - features(motions[r['right_id']]) for r in valid])
    y = np.array([1.0 if r['winner_id'] == r['left_id'] else 0.0 for r in valid])
    # With fewer than ten, train on all and explicitly withhold validation claims.
    # Split by unordered motion pair, avoiding leakage when users repeat comparisons.
    groups={}
    for i,r in enumerate(valid):
        groups.setdefault(tuple(sorted((r['left_id'],r['right_id']))), []).append(i)
    heldout = len(valid)>=10 and len(groups)>=4
    if heldout:
        rng=np.random.default_rng(37)
        keys=list(groups)
        rng.shuffle(keys)
        held_keys=set(keys[-max(1,len(keys)//5):])
        test_idx=np.array([i for key in held_keys for i in groups[key]],dtype=int)
        train_idx=np.array([i for key in keys if key not in held_keys for i in groups[key]],dtype=int)
        # A heavily repeated pair must not push the training set below minimum size.
        if len(train_idx)<6:
            train_idx=np.arange(len(valid));test_idx=np.array([],dtype=int)
    else:
        train_idx=np.arange(len(valid));test_idx=np.array([],dtype=int)
    # Scale individual motion features (not differences) for interpretable reusable ranking.
    train_ids={mid for idx in train_idx for mid in (valid[idx]['left_id'],valid[idx]['right_id'])}
    mat=np.stack([features(motions[mid]) for mid in sorted(train_ids)])
    scale=np.maximum(mat.std(axis=0), 1e-3)
    z=X/scale
    weights=np.zeros(z.shape[1]); reg=.05
    for _ in range(1000):
        p=sigmoid(z[train_idx] @ weights)
        grad=z[train_idx].T @ (p-y[train_idx])/len(train_idx) + reg*weights
        weights-=.08*grad
    train_prob=sigmoid(z[train_idx]@weights)
    def loss(probs, labels):
        probs=np.clip(probs,1e-8,1-1e-8)
        return float(np.mean(-labels*np.log(probs)-(1-labels)*np.log(1-probs)))
    source_motion_ids=sorted({mid for r in valid for mid in (r['left_id'],r['right_id'])})
    training_motion_ids=sorted({mid for idx in train_idx for mid in (valid[idx]['left_id'],valid[idx]['right_id'])})
    validation_motion_ids=sorted({mid for idx in test_idx for mid in (valid[idx]['left_id'],valid[idx]['right_id'])})
    result={'version':VERSION,'training_count':int(len(train_idx)), 'total_usable_votes':len(valid),
            'heldout_count':int(len(test_idx)), 'train_log_loss':round(loss(train_prob,y[train_idx]),5),
            'heldout_accuracy':None,'heldout_log_loss':None,
            'weights':weights.tolist(),'scale':scale.tolist(),'features':list(FEATURES),
            'source_motion_ids':source_motion_ids,'training_motion_ids':training_motion_ids,
            'validation_motion_ids':validation_motion_ids,
            'dataset_sha256':sha256(canonical_bytes(valid)), 'trained_at':int(time.time()),
            'caveat':'Only ranks G1 motion candidates from these proxy features for similar tasks; no physical correctness or real-robot safety guarantee.'}
    if len(test_idx):
        p=sigmoid(z[test_idx]@weights)
        result['heldout_accuracy']=round(float(np.mean((p >= .5)==y[test_idx])),4)
        result['heldout_log_loss']=round(loss(p,y[test_idx]),5)
    # Snapshot is immutable; votes after this point require retraining.
    conn.execute('INSERT INTO models(model_json,created_at) VALUES(?,?)',(json.dumps(result),result['trained_at']))
    conn.commit()
    return result

def rank(conn, base):
    model=latest(conn)
    if not model: return {'model':None, 'rankings':[]}
    w=np.array(model['weights']); scale=np.array(model['scale'])
    motions=[]
    for path in base.glob('*.json'):
        if path.name.endswith('.g1.json'):continue
        try:
            m=get_motion(base,path.stem)
            f=features(m)
            score=float(f/scale @ w)
            motions.append({'id':m['id'],'name':m.get('name',m['id']), 'ranking_signal':round(score,4),
                           'native_sha256':m['kinematic_evaluation']['native_sha256']})
        except (ValueError,KeyError,OSError,TypeError):continue
    motions.sort(key=lambda m:m['ranking_signal'],reverse=True)
    return {'model':model,'rankings':motions}

def next_pair(conn, base):
    """Select closest predicted pair not yet compared; heuristic active sampling."""
    r=rank(conn,base)
    candidates=r['rankings']
    compared={frozenset((x['left_id'],x['right_id'])) for x in list_preferences(conn)}
    best=None
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if frozenset((a['id'],b['id'])) in compared:continue
            diff=abs(a['ranking_signal']-b['ranking_signal'])
            if best is None or diff<best[0]:best=(diff,a['id'],b['id'])
    return {'pair':[best[1],best[2]] if best else [], 'method':'closest unreviewed learned signal' if r['model'] else 'none'}
