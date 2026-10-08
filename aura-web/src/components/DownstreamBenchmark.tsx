import { useCallback, useEffect, useMemo, useState } from 'react'
import { commitDownstreamBenchmarkOnChain, connectWallet, detectedWallets, explorerTransactionUrl } from '../solana'
import type { DownstreamBenchmarkReceipt, WalletConnection } from '../solana'

type Cohort = { name:string;count:number;motion_ids:string[];recommended:boolean;selector_unseen_count:number;selector_model_version:string|null }
type MetricSummary = { mean:number;std:number;min:number;max:number }
type BenchmarkResult = {
  version:string; selection_model_version:string|null; selection_model_dataset_sha256:string|null;
  cohort_name:string;cohort_count:number;train_count:number;holdout_count:number;random_trials:number;
  samples_per_motion:number;horizon_s:number;seed:number;aura_selected_ids:string[];heldout_ids:string[];selector_seen_motion_ids:string[];downstream_holdout_is_selector_unseen:boolean;
  aura_result:{one_step_mpjpe_cm:number;rollout_mpjpe_cm:number;rollout_root_error_cm:number;training_transitions:number;test_transitions:number;rollout_windows:number};
  random_summary:{one_step_mpjpe_cm:MetricSummary;rollout_mpjpe_cm:MetricSummary;rollout_root_error_cm:MetricSummary};
  comparison:Record<string,{aura:number;random_mean:number;relative_improvement_pct:number;aura_beats_random_trials_pct:number;lower_is_better:true}>;
  metric_definition:Record<string,string>;scope:string;interpretation:string;created_at:number;report_sha256:string;
}

async function api<T>(path:string, body?:unknown):Promise<T>{
  const res=await fetch(`/aura-api${path}`,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  const json=await res.json()
  if(!res.ok) throw new Error(json.error || `Request failed (${res.status})`)
  return json as T
}
function saveJson(result:BenchmarkResult){
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}))
  const a=document.createElement('a');a.href=url;a.download=`aura-downstream-benchmark-${result.report_sha256.slice(0,12)}.json`;a.click();URL.revokeObjectURL(url)
}
const STORAGE='aura-downstream-benchmark-chain-v1'
const METRICS:[keyof BenchmarkResult['aura_result'],string][]=[
  ['one_step_mpjpe_cm','One-step MPJPE'],['rollout_mpjpe_cm','0.5 s rollout MPJPE'],['rollout_root_error_cm','0.5 s root error'],
]

export function DownstreamBenchmark(){
  const [cohorts,setCohorts]=useState<Cohort[]>([]);const [cohort,setCohort]=useState('')
  const [result,setResult]=useState<BenchmarkResult|null>(null);const [busy,setBusy]=useState(false)
  const [error,setError]=useState('');const [notice,setNotice]=useState('');const [wallet,setWallet]=useState<WalletConnection|null>(null)
  const [receipts,setReceipts]=useState<DownstreamBenchmarkReceipt[]>(()=>{try{return JSON.parse(localStorage.getItem(STORAGE)||'[]')}catch{return []}})
  const refresh=useCallback(async()=>{
    try{
      const [c,r]=await Promise.all([api<{cohorts:Cohort[]}>('/benchmark/cohorts'),api<{result:BenchmarkResult|null}>('/benchmark')])
      setCohorts(c.cohorts);setResult(r.result);setCohort(old=>c.cohorts.some(x=>x.name===old)?old:(c.cohorts[0]?.name||''));setError('')
    }catch(e){setError(e instanceof Error?e.message:'Benchmark server unavailable')}
  },[])
  useEffect(()=>{void refresh()},[refresh])
  const selected=useMemo(()=>cohorts.find(x=>x.name===cohort),[cohorts,cohort])
  const run=async()=>{setBusy(true);setError('');setNotice('')
    try{
      const r=await api<BenchmarkResult>('/benchmark',{cohort_name:cohort,train_count:4,holdout_count:2,random_trials:20,samples_per_motion:100,horizon_s:.5,seed:37})
      setResult(r);setNotice('Benchmark finished. These are measured results; Aura can win, tie, or lose against random selection.')
    }catch(e){setError(e instanceof Error?e.message:'Benchmark failed')}finally{setBusy(false)}
  }
  const anchor=async()=>{if(!result)return;setBusy(true);setError('');setNotice('')
    try{let signer=wallet;if(!signer){const found=detectedWallets().find(x=>x.installed);if(!found)throw new Error('Install Phantom or Solflare and fund a devnet wallet.');signer=await connectWallet(found.kind);setWallet(signer)}
      const receipt=await commitDownstreamBenchmarkOnChain(signer,result.report_sha256,result.version);const next=[...receipts,receipt];setReceipts(next);localStorage.setItem(STORAGE,JSON.stringify(next));setNotice('Experiment report hash anchored on Solana devnet. This records provenance, not physical validity.')
    }catch(e){setError(e instanceof Error?e.message:'Could not anchor benchmark')}finally{setBusy(false)}}
  return <section className="benchmark-section" aria-label="Downstream imitation benchmark">
    <div className="card-kicker">05 / TEST THE DISCOVERY CLAIM</div>
    <div className="benchmark-heading"><div><h2>Does selection help <em>downstream learning?</em></h2><p>Aura trains the same small behavior-cloning learner twice: once on the highest-ranked motions and repeatedly on equally sized random subsets. Both are evaluated on the same held-out motions from the exact same prompt.</p></div><div className="benchmark-badge">REAL MEASURED BENCHMARK</div></div>
    <div className="benchmark-warning"><strong>Scope:</strong> this is an offline kinematic imitation proxy, not MuJoCo physics, balance validation, or real-robot task success. It tests whether Aura's selected demonstrations make a fixed next-motion learner generalize better to held-out trajectories.</div>
    {error&&<div className="generated-library-warning" role="alert">{error}</div>}{notice&&<p className="benchmark-notice" role="status">{notice}</p>}
    <div className="benchmark-controls"><label>Same-prompt cohort<select value={cohort} onChange={e=>setCohort(e.target.value)}><option value="">Choose a cohort</option>{cohorts.map(c=><option key={c.name} value={c.name}>{c.name.slice(0,72)} · {c.count} motions · {c.selector_unseen_count||0} unseen</option>)}</select></label><div><strong>{selected?.count||0}</strong><span>matching G1 motions</span></div><div><strong>{selected?.selector_unseen_count||0}</strong><span>never seen by selector</span></div><button disabled={busy||!cohort||(selected?.count||0)<7||(selected?.selector_unseen_count||0)<2||selected?.selector_model_version!=='aura-pairwise-logistic-v2'} onClick={()=>void run()}>{busy?'Running equal-budget experiment…':'Run Aura vs random'}</button></div>
    {selected?.selector_model_version&&selected.selector_model_version!=='aura-pairwise-logistic-v2'&&<p className="benchmark-hint">Retrain the preference model once with this build. Version 2 records exactly which motions the selector has seen.</p>}
    {(selected?.count||0)>0&&(selected?.count||0)<7&&<p className="benchmark-hint">Generate at least 7 motions using exactly the same prompt. Eight or more is recommended.</p>}
    {(selected?.selector_model_version==='aura-pairwise-logistic-v2')&&(selected?.selector_unseen_count||0)<2&&<p className="benchmark-hint">Now generate at least two fresh motions with this exact prompt and do not compare/vote on them. They become the selector-unseen downstream holdout.</p>}
    {!cohorts.length&&<p className="benchmark-hint">No repeated G1 prompt cohorts yet. Generate at least six candidates with one identical Text2Motion Aura prompt, collect preferences, train the selector, then generate two fresh candidates without voting on them.</p>}
    {result&&<div className="benchmark-result">
      <div className="benchmark-result-head"><div><div className="card-kicker">LATEST EXPERIMENT</div><h3>{result.cohort_name}</h3><p>{result.train_count} training motions · {result.holdout_count} held out · {result.random_trials} random baselines · {result.samples_per_motion} transitions per training motion</p></div><div className="benchmark-result-actions"><button onClick={()=>saveJson(result)}>Export experiment evidence</button>{(()=>{const receipt=receipts.find(x=>x.reportSha256===result.report_sha256);return receipt?<a target="_blank" rel="noreferrer" href={explorerTransactionUrl(receipt.signature)}>View devnet receipt ↗</a>:<button disabled={busy} onClick={()=>void anchor()}>Anchor report hash on devnet</button>})()}</div></div>
      <div className="benchmark-metrics">{METRICS.map(([key,label])=>{const c=result.comparison[key];if(!c)return null;const better=c.relative_improvement_pct>0;return <article key={key} className={better?'benchmark-metric positive':'benchmark-metric'}><small>{label} · lower is better</small><div className="benchmark-values"><span><b>Aura</b><strong>{c.aura.toFixed(2)} cm</strong></span><span><b>Random mean</b><strong>{c.random_mean.toFixed(2)} cm</strong></span></div><p>{c.relative_improvement_pct===0?'No difference':`${Math.abs(c.relative_improvement_pct).toFixed(1)}% ${better?'lower':'higher'} error`} · Aura beats {c.aura_beats_random_trials_pct.toFixed(0)}% of random trials.</p></article>})}</div>
      <div className="benchmark-evidence"><div><h4>Aura-selected training set</h4>{result.aura_selected_ids.map(id=><code key={id}>{id}</code>)}</div><div><h4>Held-out selector-unseen test set</h4>{result.heldout_ids.map(id=><code key={id}>{id}</code>)}</div><div><h4>Reproducibility</h4><p>Algorithm: {result.version}</p><p>Seed: {result.seed}</p><code>{result.report_sha256}</code></div></div>
      <p className="benchmark-interpretation">{result.interpretation}</p>
    </div>}
  </section>
}
