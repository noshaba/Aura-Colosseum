import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from aura_preference import connect
from aura_g1_metrics import report_for_npz
from aura_bounties import create_bounty, list_bounties, curate, list_curations, record_payment, attach_post_signature

WALLET_A='11111111111111111111111111111111'
WALLET_B='So11111111111111111111111111111111111111112'
SIG_POST='2'*64
SIG_PAY='3'*64

class BountyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.base=Path(self.tmp.name); self.db=self.base/'db.sqlite3'; self.conn=connect(self.db)
        for i in range(2):
            p=np.zeros((40,34,3));p[:,:,1]=1.;p[:,7,1]=p[:,14,1]=.1;p[:,0,2]=np.linspace(0,.5+i*.1,40)
            npz=self.base/f'm{i}.npz';np.savez_compressed(npz,posed_joints=p)
            (self.base/f'm{i}.json').write_text(json.dumps({'id':f'm{i}','name':'Walk forward','model':'g1','fps':30,'native_file':npz.name,'kinematic_evaluation':report_for_npz(npz,30)}))
    def tearDown(self):self.conn.close();self.tmp.cleanup()
    def test_bounty_curation_payment(self):
        b=create_bounty(self.conn,{'title':'Walk curation','prompt':'Walk forward','criterion':'Prefer controlled forward walking','requester_wallet':WALLET_A,'reward_lamports':1000000,'target_curations':5})
        self.assertEqual(len(list_bounties(self.conn)),1);self.assertEqual(len(b['bounty_sha256']),64)
        with patch('aura_bounties.verify_bounty_post',return_value={'slot':101}):
            b=attach_post_signature(self.conn,b['id'],SIG_POST)
        self.assertEqual(b['posted_signature'],SIG_POST);self.assertEqual(b['posted_slot'],101)
        c=curate(self.conn,self.base,{'bounty_id':b['id'],'curator_wallet':WALLET_B,'left_id':'m0','right_id':'m1','winner_id':'m1'})
        self.assertEqual(c['reward_lamports'],1000000);self.assertEqual(len(c['evidence_sha256']),64)
        with self.assertRaises(ValueError):curate(self.conn,self.base,{'bounty_id':b['id'],'curator_wallet':WALLET_B,'left_id':'m1','right_id':'m0','winner_id':'m1'})
        with patch('aura_bounties.verify_curation_payment',return_value={'slot':202}):
            paid=record_payment(self.conn,{'curation_id':c['id'],'signature':SIG_PAY,'paid_lamports':1000000})
        self.assertEqual(paid['payment_signature'],SIG_PAY);self.assertEqual(paid['payment_slot'],202);self.assertTrue(paid['payment_verified_at']);self.assertEqual(len(list_curations(self.conn,b['id'])),1)

if __name__=='__main__':unittest.main()
