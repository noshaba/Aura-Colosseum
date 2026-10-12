import { useCallback, useEffect, useMemo, useState } from 'react'
import { commitDownstreamBenchmarkOnChain, connectWallet, detectedWallets, explorerTransactionUrl } from '../solana'
import type { DownstreamBenchmarkReceipt, WalletConnection } from '../solana'

type SelectorName = 'reward_transformer'|'logistic'
type SelectorStatus = { label:string;trained:boolean;version:string|null;unseen_count:number }
type Cohort = { name:string;count:number;motion_ids:string[];recommended:boolean;selectors?:Record<SelectorName,SelectorStatus> }
type VsRandom = { error_reduction_cm:number;ci95_cm:[number,number];relative_improvement_pct:number;verdict:string }
type MetricSummary = { mean:number;std:number;min:number;max:number }
type BenchmarkResult = {
  version:string; selector?:string; selector_label?:string; selection_model_version:string|null; selection_model_dataset_sha256:string|null;
  cohort_name:string;cohort_count:number;train_count:number;holdout_count:number;random_trials:number;
  samples_per_motion:number;horizon_s:number;seed:number;aura_selected_ids:string[];heldout_ids:string[];selector_seen_motion_ids:string[];downstream_holdout_is_selector_unseen:boolean;
  aura_result:{one_step_mpjpe_cm:number;rollout_mpjpe_cm:number;rollout_root_error_cm:number;training_transitions:number;test_transitions:number;rollout_windows:number};
  random_summary:{one_step_mpjpe_cm:MetricSummary;rollout_mpjpe_cm:MetricSummary;rollout_root_error_cm:MetricSummary};
  comparison:Record<string,{aura:number;random_mean:number;relative_improvement_pct:number;aura_beats_random_trials_pct:number;lower_is_better:true;ci95_cm?:[number,number];verdict?:string}>;
  selectors?:Record<string,{label:string;learned:boolean;selected_ids:string[];vs_random:Record<string,VsRandom>}>;
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
// Mirrors aura_downstream_benchmark.py: 6 training + 6 held-out motions + 1 spare so random subsets differ.
const TRAIN_COUNT=6, HOLDOUT_COUNT=6, MIN_COHORT=TRAIN_COUNT+HOLDOUT_COUNT+1
const SELECTOR_OPTIONS:[SelectorName,string][]=[['reward_transformer','Aura reward model (Transformer)'],['logistic','Pairwise logistic (6 kinematic features)']]
const ci=(v:[number,number])=>`95% CI ${v[0]>=0?'+':''}${v[0].toFixed(2)} to ${v[1]>=0?'+':''}${v[1].toFixed(2)} cm`
const METRICS:[keyof BenchmarkResult['aura_result'],string][]=[
  ['one_step_mpjpe_cm','One-step MPJPE'],['rollout_mpjpe_cm','0.5 s rollout MPJPE'],['rollout_root_error_cm','0.5 s root error'],
]

export function DownstreamBenchmark(){
  const [cohorts,setCohorts]=useState<Cohort[]>([]);const [cohort,setCohort]=useState('');const [selectorName,setSelectorName]=useState<SelectorName>('reward_transformer')
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
  const sel=selected?.selectors?.[selectorName];const unseen=sel?.unseen_count||0;const count=selected?.count||0
  const run=async()=>{setBusy(true);setError('');setNotice('')
    try{
      const r=await api<BenchmarkResult>('/benchmark',{cohort_name:cohort,selector:selectorName,train_count:TRAIN_COUNT,holdout_count:HOLDOUT_COUNT,random_trials:20,samples_per_motion:100,horizon_s:.5,seed:37})
      setResult(r);setNotice('Benchmark finished. These are measured results: a selector can win, tie, or lose against random selection.')
    }catch(e){setError(e instanceof Error?e.message:'Benchmark failed')}finally{setBusy(false)}
  }
  const anchor=async()=>{if(!result)return;setBusy(true);setError('');setNotice('')
    try{let signer=wallet;if(!signer){const found=detectedWallets().find(x=>x.installed);if(!found)throw new Error('Install Phantom or Solflare and fund a devnet wallet.');signer=await connectWallet(found.kind);setWallet(signer)}
      const receipt=await commitDownstreamBenchmarkOnChain(signer,result.report_sha256,result.version);const next=[...receipts,receipt];setReceipts(next);localStorage.setItem(STORAGE,JSON.stringify(next));setNotice('Experiment report hash anchored on Solana devnet. This records provenance, not physical validity.')
    }catch(e){setError(e instanceof Error?e.message:'Could not anchor benchmark')}finally{setBusy(false)}}
  return <section className="benchmark-section" aria-label="Downstream imitation benchmark">
    <div className="card-kicker">TEST THE DISCOVERY CLAIM</div>
    <div className="benchmark-heading"><div><h2>Does selection help <em>downstream learning?</em></h2><p>Aura trains the same small behavior-cloning learner on the motions a selector ranks highest, on two training-free baselines, and repeatedly on equally sized random subsets. All are evaluated on the same held-out motions from the exact same prompt, with a bootstrap confidence interval against random.</p></div><div className="benchmark-badge">REAL MEASURED BENCHMARK</div></div>
    <div className="benchmark-warning"><strong>Scope:</strong> this is an offline kinematic imitation proxy, not MuJoCo physics, balance validation, or real-robot task success. It tests whether Aura's selected demonstrations make a fixed next-motion learner generalize better to held-out trajectories.</div>
    {error&&<div className="generated-library-warning" role="alert">{error}</div>}{notice&&<p className="benchmark-notice" role="status">{notice}</p>}
    <div className="benchmark-controls"><label>Same-prompt cohort<select value={cohort} onChange={e=>setCohort(e.target.value)}><option value="">Choose a cohort</option>{cohorts.map(c=><option key={c.name} value={c.name}>{c.name.slice(0,72)} · {c.count} motions</option>)}</select></label><label>Selector<select value={selectorName} onChange={e=>setSelectorName(e.target.value as SelectorName)}>{SELECTOR_OPTIONS.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></label><div><strong>{count}</strong><span>matching G1 motions</span></div><div><strong>{unseen}</strong><span>never seen by selector</span></div><button disabled={busy||!cohort||!sel?.trained||count<MIN_COHORT||unseen<HOLDOUT_COUNT} onClick={()=>void run()}>{busy?'Running equal-budget experiment…':'Run selector vs random'}</button></div>
    {selected&&sel&&!sel.trained&&<p className="benchmark-hint">Train the {sel.label} first. Its report records which motions it has seen, so the held-out set can exclude them.</p>}
    {count>0&&count<MIN_COHORT&&<p className="benchmark-hint">Generate at least {MIN_COHORT} motions with exactly the same prompt: {TRAIN_COUNT} to train on, {HOLDOUT_COUNT} held out, and 1 spare so random subsets can differ.</p>}
    {sel?.trained&&unseen<HOLDOUT_COUNT&&<p className="benchmark-hint">Now generate {HOLDOUT_COUNT-unseen} more fresh motion{HOLDOUT_COUNT-unseen===1?'':'s'} with this exact prompt and don't vote on them. They become the selector-unseen held-out set ({HOLDOUT_COUNT} needed for a confidence interval).</p>}
    {!cohorts.length&&<p className="benchmark-hint">No repeated G1 prompt cohorts yet. Generate candidates with one identical NVIDIA Kimodo prompt, collect preferences, train a selector, then generate six fresh candidates without voting on them.</p>}
    {result&&<div className="benchmark-result">
      <div className="benchmark-result-head"><div><div className="card-kicker">LATEST EXPERIMENT</div><h3>{result.cohort_name}</h3><p>{result.selector_label||'Pairwise logistic selector'} · {result.train_count} training motions · {result.holdout_count} held out · {result.random_trials} random baselines · {result.samples_per_motion} transitions per training motion</p></div><div className="benchmark-result-actions"><button onClick={()=>saveJson(result)}>Export experiment evidence</button>{(()=>{const receipt=receipts.find(x=>x.reportSha256===result.report_sha256);return receipt?<a target="_blank" rel="noreferrer" href={explorerTransactionUrl(receipt.signature)}>View devnet receipt ↗</a>:<button disabled={busy} onClick={()=>void anchor()}>Anchor report hash on devnet</button>})()}</div></div>
      <div className="benchmark-metrics">{METRICS.map(([key,label])=>{const c=result.comparison[key];if(!c)return null;const better=c.relative_improvement_pct>0;return <article key={key} className={better?'benchmark-metric positive':'benchmark-metric'}><small>{label} · lower is better</small><div className="benchmark-values"><span><b>Aura</b><strong>{c.aura.toFixed(2)} cm</strong></span><span><b>Random mean</b><strong>{c.random_mean.toFixed(2)} cm</strong></span></div><p>{c.relative_improvement_pct===0?'No difference':`${Math.abs(c.relative_improvement_pct).toFixed(1)}% ${better?'lower':'higher'} error`} · Aura beats {c.aura_beats_random_trials_pct.toFixed(0)}% of random trials.</p>{c.verdict&&c.ci95_cm&&<p><b>{c.verdict}</b> · {ci(c.ci95_cm)}</p>}</article>})}</div>
      {result.selectors&&<table className="benchmark-selectors"><caption>Every selector vs random, 0.5 s rollout error (positive = lower error than random)</caption><thead><tr><th>Selector</th><th>Error reduction</th><th>95% interval</th><th>Verdict</th></tr></thead><tbody>{Object.entries(result.selectors).map(([name,s])=>{const v=s.vs_random.rollout_mpjpe_cm;return v&&<tr key={name}><td>{s.label}{s.learned?'':' · baseline'}</td><td>{v.error_reduction_cm>=0?'+':''}{v.error_reduction_cm.toFixed(2)} cm</td><td>{ci(v.ci95_cm).replace('95% CI ','')}</td><td>{v.verdict}</td></tr>})}</tbody></table>}
      <div className="benchmark-evidence"><div><h4>Aura-selected training set</h4>{result.aura_selected_ids.map(id=><code key={id}>{id}</code>)}</div><div><h4>Held-out selector-unseen test set</h4>{result.heldout_ids.map(id=><code key={id}>{id}</code>)}</div><div><h4>Reproducibility</h4><p>Algorithm: {result.version}</p><p>Seed: {result.seed}</p><code>{result.report_sha256}</code></div></div>
      <p className="benchmark-interpretation">{result.interpretation}</p>
    </div>}
  </section>
}
