import json
import tempfile
from pathlib import Path
import numpy as np
from aura_constraints import evaluate_candidates, PRESETS, VERSION
from aura_preference import connect


def make_motion(base: Path, mid: str, forward: float, drift: float = 0.0, torso_dx: float = 0.0):
    t=90; fps=30.0
    p=np.zeros((t,34,3),dtype=np.float32)
    root_z=np.linspace(0,forward,t); root_x=np.linspace(0,drift,t)
    p[:,:,1]=1.0
    p[:,0,0]=root_x;p[:,0,2]=root_z
    # legs / toes near floor
    p[:,7]=np.stack([root_x-0.1,np.full(t,0.02),root_z],axis=1)
    p[:,14]=np.stack([root_x+0.1,np.full(t,0.02),root_z],axis=1)
    # torso and hands
    p[:,17]=np.stack([root_x+torso_dx,np.full(t,1.55),root_z],axis=1)
    p[:,25]=np.stack([root_x-0.35,np.full(t,1.25),root_z],axis=1)
    p[:,33]=np.stack([root_x+0.35,np.full(t,1.25),root_z],axis=1)
    for j in range(34):
        if j in (0,7,14,17,25,33): continue
        p[:,j,0]=root_x;p[:,j,2]=root_z
    r=np.tile(np.eye(3,dtype=np.float32),(t,34,1,1))
    np.savez(base/f'{mid}.npz',posed_joints=p,global_rot_mats=r,foot_contacts=np.zeros((t,4),dtype=bool))
    (base/f'{mid}.json').write_text(json.dumps({'id':mid,'name':'same prompt','model':'g1','native_file':f'{mid}.npz','fps':fps}))


def main():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td); db=base/'prefs.sqlite3'
        make_motion(base,'good',1.5,0.05)
        make_motion(base,'bad',0.1,1.0,0.8)
        with connect(db): pass
        report=evaluate_candidates(base,db,['bad','good'],PRESETS['balanced_forward_walk']['constraints'])
        assert report['version']==VERSION
        assert report['ordered_ids'][0]=='good'
        assert report['results'][0]['passed']>report['results'][1]['passed']
        assert len(report['report_sha256'])==64
        print('constraint tests passed', report['results'][0]['passed'], report['results'][1]['passed'])

if __name__=='__main__': main()
