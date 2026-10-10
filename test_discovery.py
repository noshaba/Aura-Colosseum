import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aura_g1_metrics import report_for_npz
from aura_preference import connect, add_preference, list_preferences, latest, train, rank, next_pair

class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        self.db=self.base/'preferences.db'
        self.conn=connect(self.db)
        self.ids=[]
        for i in range(5):
            pos=np.zeros((40,34,3))
            pos[:,:,1]=1.0
            pos[:,7,1]=pos[:,14,1]=.15
            pos[:,0,2]=np.linspace(0, .2*(i+1), 40)
            pos[:,7,0]=i*.012*np.arange(40)/40
            pos[:,14,0]=i*.013*np.arange(40)/40
            npz=self.base/f'motion{i}.npz'
            np.savez_compressed(npz,posed_joints=pos)
            meta={'id':f'motion{i}','model':'g1','fps':30,'native_file':npz.name,
                  'kinematic_evaluation':report_for_npz(npz,30)}
            (self.base/f'motion{i}.json').write_text(json.dumps(meta))
            self.ids.append(meta['id'])
    def tearDown(self):
        self.conn.close();self.tmp.cleanup()
    def test_learning_and_hashes(self):
        pairs=[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        for a,b in pairs:
            row=add_preference(self.conn,self.base,self.ids[a],self.ids[b],self.ids[a],'Forward walking')
            self.assertEqual(len(row['evidence_sha256']),64)
        self.assertEqual(len(list_preferences(self.conn)),10)
        model=train(self.conn,self.base)
        self.assertEqual(model['training_count']+model['heldout_count'],10)
        self.assertTrue(0<=model['train_log_loss']<5)
        r=rank(self.conn,self.base)
        self.assertEqual(len(r['rankings']),5)
        self.assertTrue(r['model'])
        self.assertEqual(len(next_pair(self.conn,self.base)['pair']),0)
        self.assertEqual(latest(self.conn)['dataset_sha256'], model['dataset_sha256'])
    def test_validation(self):
        with self.assertRaises(ValueError):add_preference(self.conn,self.base,'motion0','motion0','motion0','Task')
        with self.assertRaises(ValueError):add_preference(self.conn,self.base,'motion0','motion1','motion2','Task')
        with self.assertRaises(ValueError):add_preference(self.conn,self.base,'motion0','motion1','motion1','')
        with self.assertRaises(ValueError):train(self.conn,self.base)
    def test_repeated_legacy_pair_does_not_inflate_training(self):
        for _ in range(6):add_preference(self.conn,self.base,'motion0','motion1','motion0','Task')
        with self.assertRaisesRegex(ValueError, '1 valid unique comparisons'):
            train(self.conn,self.base)

if __name__=='__main__':unittest.main()
