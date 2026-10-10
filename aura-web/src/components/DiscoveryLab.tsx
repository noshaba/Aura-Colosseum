import { useCallback, useEffect, useState } from 'react'
import { GeneratedMotionPreview } from './GeneratedMotionPreview'
import { commitPreferenceOnChain, connectWallet, detectedWallets, explorerTransactionUrl } from '../solana'
import type { PreferenceChainReceipt, WalletConnection } from '../solana'

type Motion = { id:string;name:string;model:string;preview_file:string|null;kinematic_evaluation?:{native_sha256:string} }
type Preference = {id:number;left_id:string;right_id:string;winner_id:string;context:string;left_hash:string;right_hash:string;created_at:number;evidence_sha256:string}
type Model = {version:string;training_count:number;heldout_count:number;train_log_loss:number;heldout_accuracy:number|null;heldout_log_loss:number|null;trained_at:number;dataset_sha256:string;caveat:string}
type Ranking = {id:string;name:string;ranking_signal:number}
const STORAGE='aura-discovery-preference-chain-v1'
async function api<T>(path:string, body?:unknown):Promise<T>{
  const res=await fetch(`/aura-api${path}`,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  const json=await res.json()
  if(!res.ok) throw new Error(json.error || `Request failed (${res.status})`)
  return json as T
}
function exportJson(data:unknown){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='aura-preference-evidence.json';a.click();URL.revokeObjectURL(url)}
export function DiscoveryLab(){
  const [motions,setMotions]=useState<Motion[]>([])
  const [left,setLeft]=useState('');const [right,setRight]=useState('');const [context,setContext]=useState('G1 walking: prefer natural, stable-looking forward motion')
  const [votes,setVotes]=useState<Preference[]>([]);const [model,setModel]=useState<Model|null>(null);const [ranking,setRanking]=useState<Ranking[]>([])
  const [suggested,setSuggested]=useState<string[]>([]);const [error,setError]=useState('');const [notice,setNotice]=useState('')
  const [busy,setBusy]=useState(false);const [wallet,setWallet]=useState<WalletConnection|null>(null)
  const [receipts,setReceipts]=useState<PreferenceChainReceipt[]>(()=>{try{return JSON.parse(localStorage.getItem(STORAGE)||'[]')}catch{return []}})
  const refresh=useCallback(async()=>{
    try{
      const [m,p,c,r,n]=await Promise.all([
        api<Motion[]>('/motions'),api<{preferences:Preference[]}>('/preferences'),api<{model:Model|null}>('/critic'),
        api<{rankings:Ranking[]}>('/rankings'),api<{pair:string[]}>('/next-pair')])
      const eligible=m.filter(x=>x.model.toLowerCase().includes('g1')&&x.kinematic_evaluation)
      setMotions(eligible);setVotes(p.preferences);setModel(c.model);setRanking(r.rankings);setSuggested(n.pair)
      setLeft(old=>eligible.some(x=>x.id===old)?old:eligible[0]?.id||'')
      setRight(old=>eligible.some(x=>x.id===old)?old:eligible[1]?.id||'')
      setError('')
    }catch(e){setError(e instanceof Error?e.message:'Discovery server unavailable')}
  },[])
  useEffect(()=>{void refresh();const t=setInterval(()=>void refresh(),10000);return()=>clearInterval(t)},[refresh])
  const submit=async(winner:string)=>{
    setBusy(true);setError('');setNotice('')
    try{
      const record=await api<Preference>('/preferences',{left_id:left,right_id:right,winner_id:winner,context})
      setNotice(`Preference #${record.id} saved to the server. You can anchor its evidence hash on devnet below.`)
      await refresh()
    }catch(e){setError(e instanceof Error?e.message:'Could not save preference')}finally{setBusy(false)}
  }
  const train=async()=>{setBusy(true);setError('');try{await api('/train',{});setNotice('A real preference model was trained on your saved G1 comparisons.');await refresh()}catch(e){setError(e instanceof Error?e.message:'Training failed')}finally{setBusy(false)}}
  const anchor=async(v:Preference)=>{setBusy(true);setError('');try{
    let signer=wallet;if(!signer){const found=detectedWallets().find(x=>x.installed);if(!found)throw new Error('Install Phantom or Solflare and fund a devnet wallet.');signer=await connectWallet(found.kind);setWallet(signer)}
    const receipt=await commitPreferenceOnChain(signer,v.id,v.evidence_sha256)
    const next=[...receipts,receipt];setReceipts(next);localStorage.setItem(STORAGE,JSON.stringify(next));setNotice('Signed preference evidence recorded on devnet.')
  }catch(e){setError(e instanceof Error?e.message:'Could not anchor preference')}finally{setBusy(false)}}
  const a=motions.find(x=>x.id===left);const b=motions.find(x=>x.id===right)
  return <section className="discovery-section" aria-label="G1 human preference discovery">
    <div className="card-kicker">04 / DISCOVER WITH HUMAN PREFERENCES</div>
    <h2>Human judgment. <em>Learned selection.</em></h2>
    <p>Compare two generated G1 motions for the <strong>same task</strong>. A real small model learns from saved choices and ranks existing candidates. Ranking does not retrain or steer NVIDIA Kimodo's generator.</p>
    {error&&<div className="generated-library-warning" role="alert">{error}</div>}{notice&&<p role="status">{notice}</p>}
    <div className="discovery-context"><label htmlFor="discovery-context">What are you judging?</label><input id="discovery-context" maxLength={240} value={context} onChange={e=>setContext(e.target.value)} placeholder="Describe the shared task and preference criterion"/></div>
    <div className="discovery-pair">{([a,b] as const).map((m,i)=><div className="discovery-candidate" key={i}>
      <label htmlFor={`discovery-${i}`}>Candidate {i===0?'A':'B'}</label>
      <select id={`discovery-${i}`} value={i===0?left:right} onChange={e=>i===0?setLeft(e.target.value):setRight(e.target.value)}>
        {motions.map(x=><option key={x.id} value={x.id}>{x.name} · {x.id}</option>)}
      </select>
      {m?.preview_file?<GeneratedMotionPreview file={`/aura-api/files/${encodeURIComponent(m.preview_file)}`}/>:<p>No G1 preview yet.</p>}
      <button disabled={busy||!a||!b||left===right||!context.trim()} onClick={()=>void submit(m?.id||'')}>Prefer candidate {i===0?'A':'B'}</button>
    </div>)}</div>
    <p className="discovery-note">Each preview has its own playback controls. They are not frame-synchronized. Compare similar-length motions and use the same evaluation criterion.</p>
    <div className="discovery-actions">
      <button onClick={()=>void refresh()}>Refresh comparisons</button>
      <button disabled={!suggested.length} onClick={()=>{setLeft(suggested[0]);setRight(suggested[1])}}>Load closest unreviewed pair</button>
      <button disabled={busy||votes.length<6} onClick={()=>void train()}>Train preference model ({votes.length}/6 minimum)</button>
      <button onClick={()=>exportJson({preferences:votes,model,ranking,receipts})}>Export learning evidence</button>
    </div>
    <div className="discovery-results"><div><h3>Real model status</h3>
      {model?<><p>Version: {model.version}</p><p>Training pairs: {model.training_count} · held out: {model.heldout_count}</p><p>Training loss: {model.train_log_loss.toFixed(4)}</p><p>Held-out accuracy: {model.heldout_accuracy===null?'Not available (need ≥10 usable pairs)':`${(model.heldout_accuracy*100).toFixed(1)}%`}</p><p>Held-out log loss: {model.heldout_log_loss===null?'Not available':model.heldout_log_loss.toFixed(4)}</p><small>Small, correlated datasets can inflate performance. No generalization claim without independent tasks and users.</small></>:<p>Not trained. Collect at least 6 real comparisons, then train.</p>}
    </div><div><h3>Learned candidate ordering</h3>{model&&ranking.length?<ol>{ranking.slice(0,8).map(r=><li key={r.id}>{r.name} <small>· relative signal {r.ranking_signal.toFixed(2)}</small></li>)}</ol>:<p>No learned ranking until training succeeds.</p>}<small>Relative ranking is task-conditional; signals are not motion quality or physical safety scores.</small></div></div>
    <details className="discovery-ledger"><summary>Human preference records &amp; optional Solana receipts ({votes.length})</summary>
      {votes.slice(0,25).map(v=>{const receipt=receipts.find(x=>x.preferenceId===v.id&&x.evidenceSha256===v.evidence_sha256);return <article key={v.id}>
        <p>#{v.id} · {v.context} · preferred {v.winner_id}</p><code>{v.evidence_sha256}</code>
        {receipt?<a target="_blank" rel="noreferrer" href={explorerTransactionUrl(receipt.signature)}>View devnet receipt ↗</a>:<button disabled={busy} onClick={()=>void anchor(v)}>Anchor preference hash on devnet</button>}
      </article>})}</details>
  </section>
}
