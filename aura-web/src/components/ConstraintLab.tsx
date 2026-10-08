import { useEffect, useMemo, useState } from 'react'
import { commitConstraintReportOnChain, connectWallet, detectedWallets, explorerTransactionUrl } from '../solana'
import type { ConstraintReportReceipt, WalletConnection } from '../solana'

type Motion={id:string;name:string;model:string;preview_file?:string|null}
type Constraints=Record<string,number>
type Preset={id:string;label:string;description:string;constraints:Constraints}
type Check={key:string;label:string;observed:number|null;operator:string;limit:number;unit:string;pass:boolean}
type Candidate={id:string;name:string;native_sha256:string;passed:number;total:number;all_pass:boolean;pass_fraction:number;preference_signal:number|null;checks:Check[]}
type Report={version:string;scope:string;constraints:Constraints;selection_rule:string;selector_model_version:string|null;candidate_count:number;all_pass_count:number;ordered_ids:string[];results:Candidate[];report_sha256:string}

const API='/aura-api'
const STORAGE='aura-constraint-report-chain-v1'
async function api<T>(path:string,body?:unknown):Promise<T>{
  const r=await fetch(`${API}${path}`,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  const j=await r.json();if(!r.ok)throw new Error(j.error||`Request failed (${r.status})`);return j as T
}
function exportJson(data:unknown,hash:string){const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download=`aura-constraint-report-${hash.slice(0,12)}.json`;a.click();URL.revokeObjectURL(u)}
const FIELDS:[string,string,string][]=[
  ['min_forward_progress_m','Min forward progress','m'],['max_lateral_drift_m','Max lateral drift','m'],
  ['min_path_efficiency','Min path efficiency','ratio'],['max_root_displacement_m','Max root displacement','m'],
  ['max_root_height_range_m','Max root height range','m'],['max_near_floor_toe_speed_m_s','Max near-floor toe speed','m/s'],
  ['max_below_floor_fraction','Max below-floor fraction','fraction'],['max_torso_lean_deg','Max torso lean','deg'],
  ['max_end_root_speed_m_s','Max ending root speed','m/s'],
]

export function ConstraintLab(){
  const [motions,setMotions]=useState<Motion[]>([]);const [presets,setPresets]=useState<Preset[]>([])
  const [preset,setPreset]=useState('balanced_forward_walk');const [constraints,setConstraints]=useState<Constraints>({})
  const [cohort,setCohort]=useState('');const [report,setReport]=useState<Report|null>(null);const [busy,setBusy]=useState(false)
  const [error,setError]=useState('');const [notice,setNotice]=useState('');const [wallet,setWallet]=useState<WalletConnection|null>(null)
  const [receipts,setReceipts]=useState<ConstraintReportReceipt[]>(()=>{try{return JSON.parse(localStorage.getItem(STORAGE)||'[]')}catch{return []}})
  const cohorts=useMemo(()=>{const m=new Map<string,Motion[]>();motions.filter(x=>x.model.toLowerCase().includes('g1')).forEach(x=>m.set(x.name,[...(m.get(x.name)||[]),x]));return [...m.entries()].filter(([,v])=>v.length>=2).sort((a,b)=>b[1].length-a[1].length)},[motions])
  useEffect(()=>{void (async()=>{try{const [m,p]=await Promise.all([api<Motion[]>('/motions'),api<{presets:Preset[]}>('/constraints/presets')]);setMotions(m);setPresets(p.presets);const chosen=p.presets.find(x=>x.id===preset)||p.presets[0];if(chosen)setConstraints({...chosen.constraints});const groups=new Map<string,number>();m.filter(x=>x.model.toLowerCase().includes('g1')).forEach(x=>groups.set(x.name,(groups.get(x.name)||0)+1));const first=[...groups].filter(([,n])=>n>=2).sort((a,b)=>b[1]-a[1])[0]?.[0]||'';setCohort(first)}catch(e){setError(e instanceof Error?e.message:'Constraint engine unavailable')}})()},[])
  const choosePreset=(id:string)=>{setPreset(id);const p=presets.find(x=>x.id===id);if(p)setConstraints({...p.constraints});setReport(null)}
  const selected=cohorts.find(([name])=>name===cohort)?.[1]||[]
  const run=async()=>{setBusy(true);setError('');setNotice('');try{const out=await api<Report>('/constraints/evaluate',{motion_ids:selected.map(x=>x.id),constraints});setReport(out);setNotice(`${out.all_pass_count} of ${out.candidate_count} candidates satisfy every selected constraint.`)}catch(e){setError(e instanceof Error?e.message:'Constraint evaluation failed')}finally{setBusy(false)}}
  const anchor=async()=>{if(!report)return;setBusy(true);setError('');try{let signer=wallet;if(!signer){const found=detectedWallets().find(x=>x.installed);if(!found)throw new Error('Install Phantom or Solflare and fund a devnet wallet.');signer=await connectWallet(found.kind);setWallet(signer)}const receipt=await commitConstraintReportOnChain(signer,report.report_sha256,report.version);const next=[...receipts,receipt];setReceipts(next);localStorage.setItem(STORAGE,JSON.stringify(next));setNotice('Constraint-selection report hash anchored on devnet.')}catch(e){setError(e instanceof Error?e.message:'Could not anchor report')}finally{setBusy(false)}}
  const receipt=report?receipts.find(x=>x.reportSha256===report.report_sha256):undefined
  return <section className="constraint-section" aria-label="Aura native G1 constraint engine">
    <div className="constraint-head"><div><div className="card-kicker">03 / NATIVE CONSTRAINT ENGINE</div><h2>Filter for the task. <em>Then learn the preference.</em></h2><p>Aura checks explicit kinematic requirements directly on Text2Motion Aura's saved G1 trajectories. Hard constraints come first; the human-learned preference signal only breaks ties between equally satisfying candidates.</p></div><div className="constraint-badge">NO BIOIK · NO AI4ANIMATION</div></div>
    <div className="constraint-warning"><strong>Scope:</strong> this is Aura-owned kinematic screening, not inverse-kinematics solving, collision detection, dynamics, balance proof, or hardware safety validation.</div>
    {error&&<div className="generated-library-warning" role="alert">{error}</div>}{notice&&<p className="constraint-notice" role="status">{notice}</p>}
    <div className="constraint-controls"><label>Same-prompt cohort<select value={cohort} onChange={e=>{setCohort(e.target.value);setReport(null)}}><option value="">Choose generated cohort</option>{cohorts.map(([name,rows])=><option key={name} value={name}>{name.slice(0,85)} · {rows.length} motions</option>)}</select></label><label>Constraint preset<select value={preset} onChange={e=>choosePreset(e.target.value)}>{presets.map(p=><option key={p.id} value={p.id}>{p.label}</option>)}</select></label><button disabled={busy||!cohort||!Object.keys(constraints).length} onClick={()=>void run()}>{busy?'Evaluating…':'Evaluate candidates'}</button></div>
    <div className="constraint-grid">{FIELDS.map(([key,label,unit])=>{const active=constraints[key]!==undefined;return <label className={active?'constraint-field active':'constraint-field'} key={key}><span><input type="checkbox" checked={active} onChange={e=>{const next={...constraints};if(e.target.checked)next[key]=key.startsWith('min_')?0:1;else delete next[key];setConstraints(next);setReport(null)}}/>{label}</span><div><input disabled={!active} type="number" step="0.01" value={active?constraints[key]:''} onChange={e=>{const n=Number(e.target.value);setConstraints(c=>({...c,[key]:Number.isFinite(n)?n:0}));setReport(null)}}/><small>{unit}</small></div></label>})}</div>
    {report&&<div className="constraint-result"><div className="constraint-summary"><div><small>STRICT PASSES</small><strong>{report.all_pass_count}/{report.candidate_count}</strong></div><div><small>SELECTION METHOD</small><span>constraints → preference tie-break</span></div><div className="constraint-actions"><button onClick={()=>exportJson(report,report.report_sha256)}>Export evidence</button>{receipt?<a href={explorerTransactionUrl(receipt.signature)} target="_blank" rel="noreferrer">View devnet receipt ↗</a>:<button disabled={busy} onClick={()=>void anchor()}>Anchor report hash</button>}</div></div>
      <div className="constraint-candidates">{report.results.slice(0,8).map((r,i)=><article key={r.id} className={r.all_pass?'constraint-candidate pass':'constraint-candidate'}><header><div><small>#{i+1} · {r.all_pass?'ALL CONSTRAINTS PASS':'PARTIAL MATCH'}</small><h3>{r.name}</h3><code>{r.id}</code></div><strong>{r.passed}/{r.total}</strong></header><div className="constraint-checks">{r.checks.map(c=><div className={c.pass?'ok':'fail'} key={c.key}><span>{c.pass?'✓':'×'} {c.label}</span><small>{c.observed===null?'n/a':Number(c.observed).toFixed(c.unit==='fraction'||c.unit==='ratio'?3:2)} {c.unit} {c.operator} {c.limit}</small></div>)}</div><footer>{r.preference_signal===null?'No trained preference signal':`Preference tie-break signal ${r.preference_signal.toFixed(2)}`}</footer></article>)}</div>
      <p className="constraint-hash">Reproducible report SHA-256 · <code>{report.report_sha256}</code></p></div>}
  </section>
}
