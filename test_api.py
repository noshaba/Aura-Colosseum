import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen, Request
import numpy as np
from aura_g1_metrics import report_for_npz

class ApiTest(unittest.TestCase):
    def test_http_integration(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            for i in range(3):
                p=np.zeros((10,34,3));p[:,:,1]=1;p[:,7,1]=p[:,14,1]=.1;p[:,0,2]=np.linspace(0,i+.1,10)
                file=base/f'motion{i}.npz';np.savez(file,posed_joints=p)
                entry={'id':f'motion{i}','name':'Walk forward','model':'G1','fps':30,'native_file':file.name,'preview_file':None,'kinematic_evaluation':report_for_npz(file,30)}
                (base/f'motion{i}.json').write_text(json.dumps(entry))
                (base/f'motion{i}.g1.json').write_text('{}')
            env={**os.environ,'AURA_MOTION_LIBRARY':temp}
            server=subprocess.Popen([sys.executable,'aura_library_server.py'],env=env,cwd=Path(__file__).parent,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                for attempt in range(40):
                    try:
                        with urlopen('http://127.0.0.1:8765/motions',timeout=1) as f:
                            motions=json.load(f)
                        break
                    except Exception:time.sleep(.1)
                else: self.fail('Server failed to start')
                self.assertEqual(len(motions),3) # preview JSON should not become motion entries
                def post(path,payload):
                    data=json.dumps(payload).encode()
                    with urlopen(Request('http://127.0.0.1:8765'+path,data=data,headers={'Content-Type':'application/json'}),timeout=5) as f:return json.load(f)
                saved=post('/preferences',{'left_id':'motion0','right_id':'motion1','winner_id':'motion0','context':'walking forward'})
                self.assertEqual(len(saved['evidence_sha256']),64)
                for i in range(5):post('/preferences',{'left_id':'motion0','right_id':'motion2','winner_id':'motion0','context':'walking forward'})
                trained=post('/train',{})
                self.assertEqual(trained['training_count'],6)
                with urlopen('http://127.0.0.1:8765/rankings') as f: ranked=json.load(f)
                self.assertEqual(len(ranked['rankings']),3)
                with urlopen('http://127.0.0.1:8765/preferences') as f: votes=json.load(f)
                self.assertEqual(len(votes['preferences']),6)
                with urlopen('http://127.0.0.1:8765/constraints/presets') as f: presets=json.load(f)
                self.assertGreaterEqual(len(presets['presets']),3)
                constraint_report=post('/constraints/evaluate',{'motion_ids':['motion0','motion1','motion2'],'constraints':{'max_root_height_range_m':0.5,'max_below_floor_fraction':0.5}})
                self.assertEqual(constraint_report['candidate_count'],3)
                self.assertEqual(len(constraint_report['report_sha256']),64)
            finally:
                server.terminate();server.wait(timeout=5)
if __name__=='__main__':unittest.main()
