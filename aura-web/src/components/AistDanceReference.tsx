import { useEffect, useRef, useState } from 'react'
import { AIST_REFERENCE_MOTIONS, STUDY_CLIP_SECONDS } from '../aistReferenceMotions'
import { DEFAULT_VIEW_POSE, XBotScene } from './XBotScene'
import type { ViewPose } from './XBotScene'

export function AistDanceReference(){
  const [index,setIndex]=useState(0)
  const [paused,setPaused]=useState(false)
  const [progress,setProgress]=useState(0)
  const [autoRotate,setAutoRotate]=useState(true)
  const [trails,setTrails]=useState(true)
  const [landmarks,setLandmarks]=useState(false)
  const [resetViewSignal,setResetViewSignal]=useState(0)
  const [viewPose,setViewPose]=useState<ViewPose>({...DEFAULT_VIEW_POSE})
  const dragging=useRef(false)
  const motion=AIST_REFERENCE_MOTIONS[index]

  useEffect(()=>{
    setProgress(0)
    setPaused(false)
    setResetViewSignal(v=>v+1)
  },[index])

  useEffect(()=>{
    if(paused) return
    let raf=0
    let previous=performance.now()
    const tick=(now:number)=>{
      const dt=Math.min((now-previous)/1000,.1)
      previous=now
      if(!dragging.current) setProgress(v=>(v+(dt/STUDY_CLIP_SECONDS)*100)%100)
      raf=requestAnimationFrame(tick)
    }
    raf=requestAnimationFrame(tick)
    return()=>cancelAnimationFrame(raf)
  },[paused])

  const reset=()=>{setProgress(0);setPaused(false);setResetViewSignal(v=>v+1)}
  const front=()=>{setAutoRotate(false);setViewPose({...DEFAULT_VIEW_POSE});setResetViewSignal(v=>v+1)}

  return <section className="aist-reference-section">
    <div className="aist-reference-head">
      <div><div className="card-kicker">REFERENCE / AIST++</div><h2>Retargeted <em>dance motion.</em></h2><p>This is the original bundled AIST++ human dance reference retargeted onto Aura's Unitree G1 viewer. It is a reference/demo asset, not a Text2Motion Aura generation and not part of the downstream experiment unless you explicitly include it.</p></div>
      <div className="aist-reference-controls"><label>CLIP<select value={index} onChange={e=>setIndex(Number(e.target.value))}>{AIST_REFERENCE_MOTIONS.map((m,i)=><option key={m.id} value={i}>{m.label}</option>)}</select></label></div>
    </div>
    <div className="aist-reference-viewer">
      <XBotScene embodiment="g1" side="A" quality="reference" motionFile={motion.file} motionUrl={motion.url} degradationSeed={motion.seed} startOffsetSeconds={motion.startOffsetSeconds} paused={paused} playhead={progress} autoRotate={autoRotate} showTrails={trails} showLandmarks={landmarks} resetViewSignal={resetViewSignal} viewPose={viewPose} onViewPoseChange={setViewPose}/>
    </div>
    <div className="aist-reference-playback">
      <button onClick={()=>setPaused(v=>!v)}>{paused?'Play':'Pause'}</button>
      <button onClick={reset}>Restart</button>
      <input type="range" min="0" max="100" step="0.1" value={progress} onPointerDown={()=>{dragging.current=true}} onPointerUp={()=>{dragging.current=false}} onPointerCancel={()=>{dragging.current=false}} onChange={e=>setProgress(Number(e.target.value))}/>
      <label><input type="checkbox" checked={autoRotate} onChange={e=>setAutoRotate(e.target.checked)}/> Auto-rotate</label>
      <label><input type="checkbox" checked={trails} onChange={e=>setTrails(e.target.checked)}/> Trails</label>
      <label><input type="checkbox" checked={landmarks} onChange={e=>setLandmarks(e.target.checked)}/> Landmarks</label>
      <button onClick={front}>Front view</button>
    </div>
  </section>
}
