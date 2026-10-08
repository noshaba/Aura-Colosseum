import json
import tempfile
from pathlib import Path
import numpy as np

from aura_preference import connect, add_preference
from aura_motion_prior import train_prior, status, transition_windows, VERSION


def make_motion(base: Path, mid: str, bias: float):
    t = 36
    fps = 30.0
    p = np.zeros((t, 34, 3), dtype=np.float32)
    # upright-ish synthetic skeleton cloud; enough variation for a deterministic smoke test
    p[:, :, 1] = np.linspace(0.05, 1.0, 34)[None, :]
    phase = np.linspace(0, 2*np.pi, t)
    p[:, :, 0] = (0.015 + bias * 0.004) * np.sin(phase)[:, None]
    p[:, :, 2] = np.linspace(0, 0.5 + bias * 0.05, t)[:, None]
    p[:, 7, 1] = 0.02 + 0.006*np.sin(phase)
    p[:, 14, 1] = 0.02 + 0.006*np.cos(phase)
    rot = np.tile(np.eye(3, dtype=np.float32), (t, 34, 1, 1))
    native = base / f'{mid}.npz'
    np.savez_compressed(native, posed_joints=p, global_rot_mats=rot)
    meta = {'id':mid,'name':'same prompt','model':'g1','created_at':'2026-01-01T00:00:00Z',
            'frames':t,'fps':fps,'native_file':native.name,'preview_file':None}
    (base/f'{mid}.json').write_text(json.dumps(meta))
    return p


def main():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        positions={}
        for i,mid in enumerate(('m1','m2','m3','m4')):
            positions[mid]=make_motion(base,mid,float(i))
        assert transition_windows(positions['m1'],30).shape[1] == 208
        db=base/'prefs.sqlite3'
        with connect(db) as conn:
            votes=[('m1','m2','m2'),('m1','m3','m3'),('m1','m4','m4'),
                   ('m2','m3','m3'),('m2','m4','m4'),('m3','m4','m4')]
            for a,b,w in votes:
                add_preference(conn,base,a,b,w,'same test prompt')
            result=train_prior(conn,base,epochs=60)
        assert result['model']['version']==VERSION
        assert len(result['scores'])==4
        assert all(0 <= x['prior_score'] <= 1 for x in result['scores'])
        st=status(base)
        assert st['model']['model_sha256']==result['model']['model_sha256']
        assert len(st['scores'])==4
        print('motion prior test passed', [(x['id'],x['prior_score']) for x in st['scores']])

if __name__=='__main__': main()
