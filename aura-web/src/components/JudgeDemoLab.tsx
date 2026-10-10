import { useEffect, useMemo, useState, type ComponentProps } from 'react'
import { GeneratedMotionPreview } from './GeneratedMotionPreview'
import { Marquee, PIPELINE_ITEMS } from './Marquee'
import { ShapeIcon } from './ShapeIcon'
import { UseCaseShapes } from './UseCaseShapes'

type DemoMotion={id:string;name:string;prompt:string;model:string;fps:number;duration_s:number;preview_file:string;features:number[];feature_names:string[];screening:Record<string,number|null>;source:string}
type Vote={left:string;right:string;winner:string;createdAt:number}
type Model={weights:number[];scale:number[];ranking:{id:string;name:string;signal:number}[]}
const STORAGE='aura-judge-demo-votes-v1'
const FLOW:ComponentProps<typeof UseCaseShapes>['cases']=[
  {title:'Compare generated motions',body:'Pick between two G1 candidates side by side. Every pick is stored as one pairwise comparison.'},
  {title:'Learn preferences',body:'A pairwise logistic model trains in-browser on six kinematic feature families once four comparisons exist.'},
  {title:'Rank candidates',body:'The learned weights order every bundled motion by predicted human preference, with the raw signal shown.'},
  {title:'Live mode adds the rest',body:'Constraints, the downstream benchmark and SOL curation payouts run in the live workspace with NVIDIA Kimodo.'},
]
const sigmoid=(x:number)=>1/(1+Math.exp(-Math.max(-30,Math.min(30,x))))
function readVotes():Vote[]{try{return JSON.parse(localStorage.getItem(STORAGE)||'[]')}catch{return[]}}
function std(values:number[]){if(values.length<2)return 1;const m=values.reduce((a,b)=>a+b,0)/values.length;return Math.sqrt(values.reduce((a,b)=>a+(b-m)**2,0)/values.length)||1}
function train(motions:DemoMotion[],votes:Vote[]):Model|null{
  if(votes.length<4||motions.length<2)return null
  const map=new Map(motions.map(m=>[m.id,m]));const d=motions[0].features.length
  const scale=Array.from({length:d},(_,j)=>Math.max(.001,std(motions.map(m=>m.features[j]))));const w=Array(d).fill(0)
  const rows=votes.map(v=>{const a=map.get(v.left),b=map.get(v.right);if(!a||!b)return null;return{x:a.features.map((z,j)=>(z-b.features[j])/scale[j]),y:v.winner===v.left?1:0}}).filter(Boolean) as {x:number[];y:number}[]
  if(rows.length<4)return null
  for(let step=0;step<900;step++){
    const g=Array(d).fill(0);for(const r of rows){const p=sigmoid(r.x.reduce((s,x,j)=>s+x*w[j],0));for(let j=0;j<d;j++)g[j]+=(p-r.y)*r.x[j]/rows.length+.04*w[j]}
    for(let j=0;j<d;j++)w[j]-=.06*g[j]
  }
  const ranking=motions.map(m=>({id:m.id,name:m.name,signal:m.features.reduce((s,x,j)=>s+(x/scale[j])*w[j],0)})).sort((a,b)=>b.signal-a.signal)
  return{weights:w,scale,ranking}
}
export function JudgeDemoLab(){
  const [motions,setMotions]=useState<DemoMotion[]>([]);const [caveat,setCaveat]=useState('');const [left,setLeft]=useState('');const [right,setRight]=useState('');const [votes,setVotes]=useState<Vote[]>(readVotes);const [model,setModel]=useState<Model|null>(null)
  useEffect(()=>{fetch(`${import.meta.env.BASE_URL}demo/motions.json`).then(r=>r.json()).then(x=>{setMotions(x.motions||[]);setCaveat(x.caveat||'');setLeft((x.motions||[])[0]?.id||'');setRight((x.motions||[])[1]?.id||'')}).catch(()=>setCaveat('Demo assets failed to load.'))},[])
  useEffect(()=>localStorage.setItem(STORAGE,JSON.stringify(votes)),[votes])
  const a=motions.find(x=>x.id===left),b=motions.find(x=>x.id===right)
  const choose=(winner:string)=>{if(!a||!b||a.id===b.id)return;setVotes(v=>[...v,{left:a.id,right:b.id,winner,createdAt:Date.now()}]);const i=Math.floor(Math.random()*motions.length),j=(i+1+Math.floor(Math.random()*Math.max(1,motions.length-1)))%motions.length;setLeft(motions[i]?.id||left);setRight(motions[j]?.id||right)}
  const learned=useMemo(()=>model,[model])
  return <>
    <section className="judge-demo fx-hero__asset" aria-label="GPU-free judge demo">
      <div className="fx-demo-banner">
        <span className="fx-tag">Judge mode · no GPU / no Python server</span>
        <h2>Try Aura with bundled motion samples.</h2>
        <p className="fx-demo-warning"><strong>Interface demo, not benchmark evidence.</strong> {caveat}</p>
        <a className="fx-btn fx-btn--secondary" href="?judge=0">Open live workspace</a>
      </div>
      <div className="fx-pair">{([a,b] as const).map((m,i)=><article key={i}><div className="fx-pair__head"><span className="fx-tag">Candidate {i===0?'A':'B'}</span><select aria-label={`Candidate ${i===0?'A':'B'} motion`} value={i===0?left:right} onChange={e=>i===0?setLeft(e.target.value):setRight(e.target.value)}>{motions.map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></div>{m&&<><GeneratedMotionPreview file={`${import.meta.env.BASE_URL}${m.preview_file.replace(/^\/+/,'')}`}/><p>{m.prompt}</p><small>Root displacement {Number(m.screening.root_horizontal_displacement_m||0).toFixed(2)} m · toe-slide proxy {Number(m.screening.near_floor_toe_speed_m_s||0).toFixed(3)} m/s</small><button className="fx-btn fx-btn--primary" disabled={!a||!b||a.id===b.id} onClick={()=>choose(m.id)}>Prefer {i===0?'A':'B'}</button></>}</article>)}</div>
    </section>
    <Marquee label="Physical AI selection study / 001" items={PIPELINE_ITEMS} />
    <section className="fx-section fx-learning-section">
      <h2 className="fx-h2 fx-section__title">Learn from every comparison.<br />Right here in the browser.</h2>
      <div className="fx-learning">
        <div><ShapeIcon kind="pair" size="lg" /><h3>Human comparisons</h3><strong className="fx-learning__count">{votes.length}</strong><p>Votes are stored only in this browser in Judge Mode.</p><button className="fx-btn fx-btn--tertiary" onClick={()=>{setVotes([]);setModel(null);localStorage.removeItem(STORAGE)}}>Reset demo votes</button></div>
        <div><ShapeIcon kind="arch" size="lg" /><h3>Train local preference model</h3><p>A real pairwise logistic model runs in-browser on the same six kinematic feature families used by Aura's local discovery service.</p><button className="fx-btn fx-btn--primary" disabled={votes.length<4} onClick={()=>setModel(train(motions,votes))}>Train on {votes.length} comparisons</button></div>
        <div><ShapeIcon kind="flag" size="lg" /><h3>Learned ordering</h3>{learned?<ol>{learned.ranking.slice(0,5).map(x=><li key={x.id}>{x.name}<small>{x.signal.toFixed(2)}</small></li>)}</ol>:<p>Collect at least four comparisons, then train. No accuracy claim is shown because these bundled samples are from different tasks.</p>}</div>
      </div>
    </section>
    <section className="fx-section fx-flow" aria-label="How the judge demo works">
      <UseCaseShapes cases={FLOW} />
    </section>
    <section className="fx-section fx-arch">
      <h2 className="fx-h2 fx-section__title">Live mode runs the full pipeline, from prompt to payout</h2>
      <div className="fx-about">
        <div className="fx-about__left">
          <div className="fx-about__pill"><span className="fx-about__mark" aria-hidden="true"><img src={`${import.meta.env.BASE_URL}brand/aura-mark.png`} alt="" /></span><p>Live pipeline<br />on your own GPU</p></div>
          <a className="fx-btn fx-btn--secondary fx-btn--block" href="?judge=0">Open live workspace</a>
        </div>
        <div className="fx-about__right">
          <ol className="fx-steps">{PIPELINE_ITEMS.slice(0,6).map((s,i)=><li key={s.label}><span className={`fx-steps__dot is-${i+1}`} aria-hidden="true" /><b>{String(i+1).padStart(2,'0')}</b>{s.label}</li>)}</ol>
          <p>Judge Mode uses NVIDIA Kimodo's bundled G1 examples solely to make the interface accessible without model weights or CUDA. Submission claims should use your separately collected same-prompt experiment, not these samples.</p>
        </div>
      </div>
    </section>
  </>
}
