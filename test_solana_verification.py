import json
import unittest
from aura_bounties import (
    SYSTEM_PROGRAM_ID, MEMO_PROGRAM_ID,
    validate_bounty_post_transaction, validate_curation_payment_transaction,
)

B58='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
PAYER='11111111111111111111111111111111'
CURATOR='So11111111111111111111111111111111111111112'
SIG='2'*64


def b58encode(data: bytes) -> str:
    n=int.from_bytes(data,'big')
    out=''
    while n:
        n,r=divmod(n,58);out=B58[r]+out
    pad=len(data)-len(data.lstrip(b'\x00'))
    return '1'*pad+(out or '')


def tx_result(keys, instructions, *, signature=SIG, slot=4242, pre_balances=None, post_balances=None):
    pre_balances = pre_balances if pre_balances is not None else [10_000_000] * len(keys)
    post_balances = post_balances if post_balances is not None else list(pre_balances)
    return {
        'slot': slot,
        'meta': {'err': None, 'loadedAddresses': {'writable': [], 'readonly': []}, 'preBalances': pre_balances, 'postBalances': post_balances},
        'transaction': {
            'signatures': [signature],
            'message': {
                'header': {'numRequiredSignatures': 1},
                'accountKeys': keys,
                'instructions': instructions,
            },
        },
    }


def memo_ix(program_index: int, account_index: int, payload: dict):
    return {'programIdIndex': program_index, 'accounts': [account_index], 'data': b58encode(json.dumps(payload,separators=(',',':')).encode())}


def transfer_ix(program_index: int, source_index: int, destination_index: int, lamports: int):
    data=(2).to_bytes(4,'little')+int(lamports).to_bytes(8,'little')
    return {'programIdIndex': program_index, 'accounts': [source_index,destination_index], 'data': b58encode(data)}


class SolanaVerification(unittest.TestCase):
    def test_bounty_memo_is_verified(self):
        bounty={'id':'bounty_0123456789abcdef','bounty_sha256':'a'*64,'reward_lamports':1_000_000,'target_curations':5}
        memo={'app':'AURA','v':1,'kind':'curation-bounty','bountyId':bounty['id'],'bountySha256':bounty['bounty_sha256'],'rewardLamports':1_000_000,'targetCurations':5}
        result=tx_result([PAYER,MEMO_PROGRAM_ID],[memo_ix(1,0,memo)])
        self.assertEqual(validate_bounty_post_transaction(result,requester=PAYER,bounty=bounty)['slot'],4242)
        bad=dict(memo);bad['rewardLamports']=999
        with self.assertRaisesRegex(ValueError,'expected Aura terms memo'):
            validate_bounty_post_transaction(tx_result([PAYER,MEMO_PROGRAM_ID],[memo_ix(1,0,bad)]),requester=PAYER,bounty=bounty)

    def test_payment_transfer_and_memo_are_verified(self):
        c={'id':7,'bounty_id':'bounty_0123456789abcdef','requester_wallet':PAYER,'curator_wallet':CURATOR,'reward_lamports':1_000_000,'evidence_sha256':'b'*64}
        memo={'app':'AURA','v':1,'kind':'curation-payout','bountyId':c['bounty_id'],'curationId':7,'evidenceSha256':'b'*64,'paidLamports':1_000_000}
        keys=[PAYER,CURATOR,SYSTEM_PROGRAM_ID,MEMO_PROGRAM_ID]
        pre=[10_000_000,2_000_000,0,0];post=[8_995_000,3_000_000,0,0]
        result=tx_result(keys,[transfer_ix(2,0,1,1_000_000),memo_ix(3,0,memo)],pre_balances=pre,post_balances=post)
        self.assertEqual(validate_curation_payment_transaction(result,curation=c)['slot'],4242)
        wrong=tx_result(keys,[transfer_ix(2,0,1,900_000),memo_ix(3,0,memo)],pre_balances=pre,post_balances=post)
        with self.assertRaisesRegex(ValueError,'expected SOL transfer'):
            validate_curation_payment_transaction(wrong,curation=c)
        bad_memo=dict(memo);bad_memo['evidenceSha256']='c'*64
        wrong=tx_result(keys,[transfer_ix(2,0,1,1_000_000),memo_ix(3,0,bad_memo)],pre_balances=pre,post_balances=post)
        with self.assertRaisesRegex(ValueError,'expected Aura evidence memo'):
            validate_curation_payment_transaction(wrong,curation=c)
        bad_balance=tx_result(keys,[transfer_ix(2,0,1,1_000_000),memo_ix(3,0,memo)],pre_balances=pre,post_balances=[8_995_000,2_900_000,0,0])
        with self.assertRaisesRegex(ValueError,'balance change'):
            validate_curation_payment_transaction(bad_balance,curation=c)

if __name__=='__main__':unittest.main()
