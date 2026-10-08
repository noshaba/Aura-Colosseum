import json, os, subprocess, sys, tempfile, time, unittest, threading
from pathlib import Path
from urllib.request import urlopen, Request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from aura_g1_metrics import report_for_npz
from aura_bounties import SYSTEM_PROGRAM_ID, MEMO_PROGRAM_ID
from test_solana_verification import b58encode, memo_ix, transfer_ix, tx_result

A='11111111111111111111111111111111'; B='So11111111111111111111111111111111111111112'
SIG_POST='2'*64; SIG_PAY='3'*64
RPC_RESULTS={}

class RpcHandler(BaseHTTPRequestHandler):
  def log_message(self,*args): pass
  def do_POST(self):
    size=int(self.headers.get('Content-Length','0')); req=json.loads(self.rfile.read(size))
    sig=req.get('params',[None])[0]; result=RPC_RESULTS.get(sig)
    data=json.dumps({'jsonrpc':'2.0','id':req.get('id',1),'result':result}).encode()
    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

class CurationApi(unittest.TestCase):
  def test_market_http(self):
    RPC_RESULTS.clear()
    rpc=ThreadingHTTPServer(('127.0.0.1',0),RpcHandler);thread=threading.Thread(target=rpc.serve_forever,daemon=True);thread.start()
    try:
      with tempfile.TemporaryDirectory() as temp:
        base=Path(temp)
        for i in range(2):
          p=np.zeros((20,34,3));p[:,:,1]=1;p[:,7,1]=p[:,14,1]=.1;p[:,0,2]=np.linspace(0,.2+i*.2,20)
          n=base/f'm{i}.npz';np.savez(n,posed_joints=p)
          (base/f'm{i}.json').write_text(json.dumps({'id':f'm{i}','name':'Walk forward','model':'G1','fps':30,'native_file':n.name,'preview_file':None,'kinematic_evaluation':report_for_npz(n,30)}))
        env={**os.environ,'AURA_MOTION_LIBRARY':temp,'AURA_LIBRARY_PORT':'8766','AURA_SOLANA_RPC_URL':f'http://127.0.0.1:{rpc.server_port}'}
        proc=subprocess.Popen([sys.executable,'aura_library_server.py'],cwd=Path(__file__).parent,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
          for _ in range(40):
            try:urlopen('http://127.0.0.1:8766/bounties',timeout=.5);break
            except Exception:time.sleep(.1)
          def post(path,payload):
            data=json.dumps(payload).encode();return json.load(urlopen(Request('http://127.0.0.1:8766'+path,data=data,headers={'Content-Type':'application/json'}),timeout=5))
          b=post('/bounties',{'title':'Test','prompt':'Walk forward','criterion':'Controlled motion','requester_wallet':A,'reward_lamports':1000000,'target_curations':3})
          post_memo={'app':'AURA','v':1,'kind':'curation-bounty','bountyId':b['id'],'bountySha256':b['bounty_sha256'],'rewardLamports':1000000,'targetCurations':3}
          RPC_RESULTS[SIG_POST]=tx_result([A,MEMO_PROGRAM_ID],[memo_ix(1,0,post_memo)],signature=SIG_POST,slot=111)
          posted=post('/bounties/post-signature',{'bounty_id':b['id'],'signature':SIG_POST})
          self.assertEqual(posted['posted_slot'],111);self.assertTrue(posted['posted_verified_at'])
          c=post('/bounties/curate',{'bounty_id':b['id'],'curator_wallet':B,'left_id':'m0','right_id':'m1','winner_id':'m1'})
          pay_memo={'app':'AURA','v':1,'kind':'curation-payout','bountyId':b['id'],'curationId':c['id'],'evidenceSha256':c['evidence_sha256'],'paidLamports':1000000}
          keys=[A,B,SYSTEM_PROGRAM_ID,MEMO_PROGRAM_ID]
          RPC_RESULTS[SIG_PAY]=tx_result(keys,[transfer_ix(2,0,1,1000000),memo_ix(3,0,pay_memo)],signature=SIG_PAY,slot=222,pre_balances=[10000000,2000000,0,0],post_balances=[8995000,3000000,0,0])
          paid=post('/bounties/pay',{'curation_id':c['id'],'signature':SIG_PAY,'paid_lamports':1000000})
          self.assertEqual(paid['payment_signature'],SIG_PAY);self.assertEqual(paid['payment_slot'],222);self.assertTrue(paid['payment_verified_at'])
          ledger=json.load(urlopen(f"http://127.0.0.1:8766/bounties/curations?bounty_id={b['id']}"))
          self.assertEqual(len(ledger['curations']),1)
        finally:
          proc.terminate();proc.wait(timeout=5)
    finally:
      rpc.shutdown();rpc.server_close()
if __name__=='__main__':unittest.main()
