import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from aura_preference import connect
from aura_g1_metrics import report_for_npz
from aura_bounties import create_bounty, list_bounties, curate, list_curations, record_payment, attach_post_signature, close_bounty, create_wallet_challenge
from wallet_test_utils import TestWallet, signed

REQUESTER=TestWallet(1);CURATOR=TestWallet(2);STRANGER=TestWallet(3)
WALLET_A=REQUESTER.address
WALLET_B=CURATOR.address
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
        c=curate(self.conn,self.base,signed(self.conn,CURATOR,'curate',{'bounty_id':b['id'],'curator_wallet':WALLET_B,'left_id':'m0','right_id':'m1','winner_id':'m1'}))
        self.assertEqual(c['reward_lamports'],1000000);self.assertEqual(len(c['evidence_sha256']),64)
        with self.assertRaises(ValueError):curate(self.conn,self.base,signed(self.conn,CURATOR,'curate',{'bounty_id':b['id'],'curator_wallet':WALLET_B,'left_id':'m1','right_id':'m0','winner_id':'m1'}))
        with patch('aura_bounties.verify_curation_payment',return_value={'slot':202}):
            paid=record_payment(self.conn,{'curation_id':c['id'],'signature':SIG_PAY,'paid_lamports':1000000})
        self.assertEqual(paid['payment_signature'],SIG_PAY);self.assertEqual(paid['payment_slot'],202);self.assertTrue(paid['payment_verified_at']);self.assertEqual(len(list_curations(self.conn,b['id'])),1)

    def _posted_bounty(self,target=5):
        b=create_bounty(self.conn,{'title':'Walk curation','prompt':'Walk forward','criterion':'Prefer controlled forward walking','requester_wallet':WALLET_A,'reward_lamports':1000000,'target_curations':target})
        with patch('aura_bounties.verify_bounty_post',return_value={'slot':101}):
            return attach_post_signature(self.conn,b['id'],SIG_POST)
    def vote(self,b,wallet):
        return {'bounty_id':b['id'],'curator_wallet':wallet.address,'left_id':'m0','right_id':'m1','winner_id':'m1'}
    def test_unsigned_or_forged_curations_are_rejected(self):
        b=self._posted_bounty()
        with self.assertRaisesRegex(ValueError,'Sign a fresh wallet challenge'):
            curate(self.conn,self.base,self.vote(b,CURATOR))
        # Claiming CURATOR's address but signing with another key.
        forged=signed(self.conn,STRANGER,'curate',self.vote(b,CURATOR))
        with self.assertRaisesRegex(ValueError,'signature is invalid'):
            curate(self.conn,self.base,forged)
        self.assertEqual(list_curations(self.conn,b['id']),[])
    def test_signature_is_bound_to_the_exact_vote_and_single_use(self):
        b=self._posted_bounty()
        good=signed(self.conn,CURATOR,'curate',self.vote(b,CURATOR))
        with self.assertRaisesRegex(ValueError,'does not match'):
            curate(self.conn,self.base,{**good,'winner_id':'m0'})  # flipped winner after signing
        curate(self.conn,self.base,good)
        with self.assertRaisesRegex(ValueError,'already (used|evaluated)'):  # duplicate check or nonce: both refuse
            curate(self.conn,self.base,good)  # replay
    def test_expired_challenge_is_rejected(self):
        b=self._posted_bounty()
        payload=signed(self.conn,CURATOR,'curate',self.vote(b,CURATOR))
        with patch('aura_bounties._now',return_value=10**12):
            with self.assertRaisesRegex(ValueError,'expired'):
                curate(self.conn,self.base,payload)
    def test_only_the_requester_can_close(self):
        b=self._posted_bounty()
        with self.assertRaisesRegex(ValueError,'Only the bounty requester'):
            create_wallet_challenge(self.conn,{'action':'close','bounty_id':b['id'],'requester_wallet':STRANGER.address})
        with self.assertRaisesRegex(ValueError,'Sign a fresh wallet challenge'):
            close_bounty(self.conn,{'bounty_id':b['id'],'requester_wallet':WALLET_A})
        closed=close_bounty(self.conn,signed(self.conn,REQUESTER,'close',{'bounty_id':b['id'],'requester_wallet':WALLET_A}))
        self.assertEqual(closed['status'],'closed')

if __name__=='__main__':unittest.main()
