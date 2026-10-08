import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  commitCurationBountyOnChain,
  connectWallet,
  detectedWallets,
  explorerTransactionUrl,
  payCurationOnChain,
  shortAddress,
} from '../solana'
import type { WalletConnection } from '../solana'
import { GeneratedMotionPreview } from './GeneratedMotionPreview'

type Motion={id:string;name:string;model:string;preview_file:string|null}
type Bounty={
  id:string;title:string;prompt:string;criterion:string;requester_wallet:string;reward_lamports:number;
  target_curations:number;status:'open'|'closed';prompt_sha256:string;posted_signature:string|null;posted_slot:number|null;posted_verified_at:number|null;
  created_at:number;curation_count:number;paid_count:number;bounty_sha256:string
}
type Curation={
  id:number;bounty_id:string;preference_id:number;curator_wallet:string;left_id:string;right_id:string;
  winner_id:string;evidence_sha256:string;payment_signature:string|null;paid_lamports:number|null;
  created_at:number;paid_at:number|null;payment_slot:number|null;payment_verified_at:number|null;bounty_title:string;reward_lamports:number;requester_wallet:string
}

async function api<T>(path:string,body?:unknown):Promise<T>{
  const res=await fetch(`/aura-api${path}`,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  const json=await res.json()
  if(!res.ok) throw new Error(json.error||`Request failed (${res.status})`)
  return json as T
}
const sol=(lamports:number)=>lamports/1_000_000_000

export function CurationMarket(){
  const [motions,setMotions]=useState<Motion[]>([]);const [bounties,setBounties]=useState<Bounty[]>([]);const [curations,setCurations]=useState<Curation[]>([])
  const [wallet,setWallet]=useState<WalletConnection|null>(null);const [selected,setSelected]=useState('');const [left,setLeft]=useState('');const [right,setRight]=useState('')
  const [title,setTitle]=useState('G1 forward-walk curation');const [prompt,setPrompt]=useState('A humanoid robot walks forward for several steps while maintaining balance and comes to a controlled stop.')
  const [criterion,setCriterion]=useState('Prefer controlled forward progress, minimal foot sliding, and a clean stop.');const [rewardSol,setRewardSol]=useState('0.002');const [target,setTarget]=useState('10')
  const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [notice,setNotice]=useState('')

  const refresh=useCallback(async()=>{
    try{
      const [m,b]=await Promise.all([api<Motion[]>('/motions'),api<{bounties:Bounty[]}>('/bounties')])
      const eligible=m.filter(x=>x.model.toLowerCase().includes('g1')&&x.preview_file)
      setMotions(eligible);setBounties(b.bounties)
      setLeft(v=>eligible.some(x=>x.id===v)?v:eligible[0]?.id||'');setRight(v=>eligible.some(x=>x.id===v)?v:eligible[1]?.id||'')
      const next=selected&&b.bounties.some(x=>x.id===selected)?selected:b.bounties.find(x=>x.status==='open')?.id||b.bounties[0]?.id||''
      setSelected(next)
      if(next){const c=await api<{curations:Curation[]}>(`/bounties/curations?bounty_id=${encodeURIComponent(next)}`);setCurations(c.curations)}else setCurations([])
      setError('')
    }catch(e){setError(e instanceof Error?e.message:'Curation service unavailable')}
  },[selected])
  useEffect(()=>{void refresh()},[]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(()=>{if(!selected){setCurations([]);return}void api<{curations:Curation[]}>(`/bounties/curations?bounty_id=${encodeURIComponent(selected)}`).then(x=>setCurations(x.curations)).catch(()=>{})},[selected])

  const active=useMemo(()=>bounties.find(x=>x.id===selected)||null,[bounties,selected]);const a=motions.find(x=>x.id===left);const b=motions.find(x=>x.id===right)
  const connect=async()=>{let w=wallet;if(w)return w;const found=detectedWallets().find(x=>x.installed);if(!found)throw new Error('Install Phantom or Solflare to use devnet payouts.');w=await connectWallet(found.kind);setWallet(w);return w}
  const create=async()=>{setBusy(true);setError('');setNotice('');try{
    const w=await connect();const reward=Math.round(Number(rewardSol)*1_000_000_000);const targetN=Number(target)
    if(!Number.isFinite(reward)||reward<10_000)throw new Error('Reward must be at least 0.00001 SOL')
    const local=await api<Bounty>('/bounties',{title,prompt,criterion,requester_wallet:w.address,reward_lamports:reward,target_curations:targetN})
    setSelected(local.id)
    try{
      const receipt=await commitCurationBountyOnChain(w,local.id,local.bounty_sha256,local.reward_lamports,local.target_curations)
      await api('/bounties/post-signature',{bounty_id:local.id,signature:receipt.signature})
      setNotice('Bounty terms were published on Solana devnet and independently verified by the Aura server. Funds are not escrowed; accepted curations are paid directly.')
    }catch(chainError){setNotice(`Bounty saved locally, but its on-chain posting did not complete: ${chainError instanceof Error?chainError.message:'wallet error'}`)}
    await refresh()
  }catch(e){setError(e instanceof Error?e.message:'Could not create bounty')}finally{setBusy(false)}}
  const curate=async(winner:string)=>{if(!active) return;setBusy(true);setError('');setNotice('');try{
    const w=await connect();const row=await api<Curation>('/bounties/curate',{bounty_id:active.id,curator_wallet:w.address,left_id:left,right_id:right,winner_id:winner})
    setNotice(`Curation #${row.id} saved. It also enters Aura's preference dataset. The requester can now pay ${sol(row.reward_lamports).toFixed(4)} SOL on devnet.`);await refresh()
  }catch(e){setError(e instanceof Error?e.message:'Could not save curation')}finally{setBusy(false)}}
  const pay=async(c:Curation)=>{setBusy(true);setError('');setNotice('');try{
    const w=await connect();if(w.address!==c.requester_wallet)throw new Error('Connect the bounty requester wallet to approve this payout.')
    const receipt=await payCurationOnChain(w,c.curator_wallet,c.reward_lamports,c.id,c.bounty_id,c.evidence_sha256)
    await api('/bounties/pay',{curation_id:c.id,signature:receipt.signature,paid_lamports:c.reward_lamports})
    setNotice(`Paid ${sol(c.reward_lamports).toFixed(4)} SOL to ${shortAddress(c.curator_wallet)}. Aura independently verified the payer, recipient, amount, success status, and evidence memo from Solana RPC.`);await refresh()
  }catch(e){setError(e instanceof Error?e.message:'Payout failed')}finally{setBusy(false)}}

  return <section className="curation-market" aria-label="Solana curation market">
    <div className="card-kicker">06 / CURATION MARKET · SOLANA DEVNET</div><h2>Pay for judgment. <em>Keep the evidence.</em></h2>
    <p className="curation-intro">Requesters publish a motion-curation task, humans compare G1 candidates, and accepted curations are paid in SOL. The payout transaction also carries the curation evidence hash, and Aura verifies the transaction server-side before marking work paid. <strong>This demo uses direct requester-approved payouts, not escrow.</strong></p>
    {error&&<div className="generated-library-warning" role="alert">{error}</div>}{notice&&<p className="curation-notice" role="status">{notice}</p>}
    <div className="curation-wallet"><span>{wallet?`Wallet ${shortAddress(wallet.address)}`:'No wallet connected'}</span><button disabled={busy} onClick={()=>void connect().catch(e=>setError(e.message))}>{wallet?'Wallet connected':'Connect Phantom / Solflare'}</button></div>
    <div className="curation-create">
      <div><label>BOUNTY TITLE</label><input value={title} maxLength={80} onChange={e=>setTitle(e.target.value)}/><label>EXACT GENERATION PROMPT</label><textarea value={prompt} maxLength={480} onChange={e=>setPrompt(e.target.value)}/><label>HUMAN CURATION CRITERION</label><textarea value={criterion} maxLength={240} onChange={e=>setCriterion(e.target.value)}/></div>
      <div><label>REWARD / ACCEPTED COMPARISON · SOL</label><input type="number" min="0.00001" step="0.001" value={rewardSol} onChange={e=>setRewardSol(e.target.value)}/><label>TARGET COMPARISONS</label><input type="number" min="1" max="1000" value={target} onChange={e=>setTarget(e.target.value)}/><p>Maximum planned direct payout: <strong>{(Number(rewardSol||0)*Number(target||0)).toFixed(4)} SOL</strong>. Aura does not custody or escrow these funds in this version.</p><button disabled={busy||!title.trim()||!prompt.trim()||!criterion.trim()} onClick={()=>void create()}>Create + publish bounty on devnet</button></div>
    </div>
    <div className="curation-workspace">
      <aside><h3>Open bounties</h3>{!bounties.length?<p>No bounties yet.</p>:bounties.map(x=><button className={x.id===selected?'active':''} key={x.id} onClick={()=>setSelected(x.id)}><strong>{x.title}</strong><small>{sol(x.reward_lamports).toFixed(4)} SOL · {x.curation_count}/{x.target_curations} curations</small><span>{x.posted_verified_at?'ON-CHAIN TERMS · RPC VERIFIED':x.posted_signature?'VERIFYING TERMS':'LOCAL DRAFT'} · {x.status.toUpperCase()}</span></button>)}</aside>
      <div className="curation-task">{active?<><div className="curation-task-head"><div><h3>{active.title}</h3><p>{active.prompt}</p><small>Criterion: {active.criterion}</small></div><div><strong>{sol(active.reward_lamports).toFixed(4)} SOL</strong><small>per accepted comparison</small>{active.posted_signature&&<a href={explorerTransactionUrl(active.posted_signature)} target="_blank" rel="noreferrer">View posted terms ↗</a>}</div></div>
        <div className="curation-pair">{([a,b] as const).map((m,i)=><article key={i}><label>Candidate {i===0?'A':'B'}</label><select value={i===0?left:right} onChange={e=>i===0?setLeft(e.target.value):setRight(e.target.value)}>{motions.map(x=><option key={x.id} value={x.id}>{x.name} · {x.id}</option>)}</select>{m?.preview_file?<GeneratedMotionPreview file={`/aura-api/files/${encodeURIComponent(m.preview_file)}`}/>:<p>No preview available.</p>}<button disabled={busy||!m||left===right||active.status!=='open'} onClick={()=>void curate(m?.id||'')}>Prefer {i===0?'A':'B'} + submit curation</button></article>)}</div>
      </>:<p>Create or select a bounty.</p>}</div>
    </div>
    {active&&<div className="curation-ledger"><h3>Curations + payouts</h3>{!curations.length?<p>No comparisons submitted for this bounty yet.</p>:curations.map(c=><article key={c.id}><div><strong>#{c.id} · preferred {c.winner_id}</strong><small>Curator {shortAddress(c.curator_wallet)} · evidence {c.evidence_sha256.slice(0,12)}…</small></div>{c.payment_signature?<a href={explorerTransactionUrl(c.payment_signature)} target="_blank" rel="noreferrer">Paid {sol(c.paid_lamports||0).toFixed(4)} SOL · RPC verified · Explorer ↗</a>:<button disabled={busy||wallet?.address!==c.requester_wallet} title={wallet?.address===c.requester_wallet?'Pay curator on devnet':'Connect requester wallet to pay'} onClick={()=>void pay(c)}>Pay {sol(c.reward_lamports).toFixed(4)} SOL</button>}</article>)}</div>}
    <p className="curation-caveat">Hackathon implementation: the local coordinator stores task state and evidence; wallet keys never touch the server. Bounty terms and payouts are checked against Solana devnet RPC before being recorded. Payment verification checks transaction success, requester signer/fee payer, curator recipient, exact lamports, and the Aura evidence memo. This version is intentionally non-custodial but is <strong>not trustless escrow</strong>; a production marketplace would add an audited Solana program for locked funds, claim rules, disputes, and anti-Sybil controls.</p>
  </section>
}
