import { useMemo, useState } from 'react'
import { GeneratedG1RobotPreview } from './GeneratedG1RobotPreview'
import { SplatG1SceneViewer } from './SplatG1SceneViewer'
import './scene-lab.css'

type Obstacle = { name: string; x: number; z: number; width: number; depth: number }
type SceneProxy = {
  name: string
  bounds: [number, number, number, number]
  start: [number, number]
  goal: [number, number]
  floor_y: number
  robot_radius: number
  grid_resolution: number
  obstacles: Obstacle[]
}
type Plan = {
  method: string
  scene: SceneProxy
  path: number[][]
  path_length_m: number
  grid_cells: number
  note: string
}
type Motion = {
  id: string
  name: string
  model: string
  preview_file: string | null
  native_file: string
  native_sha256?: string
  scene_fit?: { method?: string; path_length_m?: number; claim?: string }
}
type GenerationResponse = { ok?: boolean; motions?: Motion[]; plan?: Plan; error?: string; scope?: string }

const DEFAULT_SCENE: SceneProxy = {
  name: 'Demo room',
  bounds: [-3, 3, -2.4, 2.4],
  start: [-2.2, -1.2],
  goal: [2.2, 1.1],
  floor_y: 0,
  robot_radius: 0.34,
  grid_resolution: 0.18,
  obstacles: [{ name: 'table', x: 0, z: 0, width: 1.35, depth: 0.82 }],
}

function n(value: string, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function ScenePlanSvg({ scene, path }: { scene: SceneProxy; path?: number[][] }) {
  const [xmin, xmax, zmin, zmax] = scene.bounds
  const w = xmax - xmin, h = zmax - zmin
  const toX = (x: number) => ((x - xmin) / w) * 100
  const toY = (z: number) => 100 - ((z - zmin) / h) * 100
  const polyline = path?.map(([x, z]) => `${toX(x)},${toY(z)}`).join(' ') || ''
  return <svg className="scene-plan-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Top-down geometry proxy and planned path">
    <rect x="0" y="0" width="100" height="100" className="scene-plan-floor" />
    {scene.obstacles.map((obstacle, index) => {
      const x = toX(obstacle.x - obstacle.width / 2)
      const y = toY(obstacle.z + obstacle.depth / 2)
      const ow = obstacle.width / w * 100
      const oh = obstacle.depth / h * 100
      return <g key={`${obstacle.name}-${index}`}><rect x={x} y={y} width={ow} height={oh} className="scene-plan-obstacle" /><text x={x + ow / 2} y={y + oh / 2} textAnchor="middle" dominantBaseline="central">{obstacle.name}</text></g>
    })}
    {polyline && <polyline points={polyline} className="scene-plan-path" />}
    <circle cx={toX(scene.start[0])} cy={toY(scene.start[1])} r="2.4" className="scene-plan-start" />
    <circle cx={toX(scene.goal[0])} cy={toY(scene.goal[1])} r="2.4" className="scene-plan-goal" />
    <text x={toX(scene.start[0]) + 3} y={toY(scene.start[1]) - 2}>start</text>
    <text x={toX(scene.goal[0]) + 3} y={toY(scene.goal[1]) - 2}>goal</text>
  </svg>
}

export function SceneLab() {
  const [scene, setScene] = useState<SceneProxy>(DEFAULT_SCENE)
  const [prompt, setPrompt] = useState('Walk from the start point to the goal and stop in a controlled stance.')
  const [plan, setPlan] = useState<Plan | null>(null)
  const [motions, setMotions] = useState<Motion[]>([])
  const [selected, setSelected] = useState(0)
  const [splatBuffer, setSplatBuffer] = useState<ArrayBuffer | null>(null)
  const [splatName, setSplatName] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const selectedMotion = motions[selected] || null
  const previewUrl = selectedMotion?.preview_file ? `/aura-api/files/${encodeURIComponent(selectedMotion.preview_file)}` : null
  const obstacle = scene.obstacles[0] || { name: 'obstacle', x: 0, z: 0, width: 1, depth: 1 }

  const updateObstacle = (patch: Partial<Obstacle>) => setScene(current => ({
    ...current,
    obstacles: [{ ...obstacle, ...patch }, ...current.obstacles.slice(1)],
  }))

  const planOnly = async () => {
    setBusy(true); setError(''); setMessage('Planning a collision-aware root path…')
    try {
      const response = await fetch('/aura-api/scene/plan', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scene })
      })
      const data = await response.json() as Plan & { error?: string }
      if (!response.ok) throw new Error(data.error || `Scene planner returned ${response.status}`)
      setPlan(data)
      setScene(data.scene)
      setMessage(`Path ready · ${data.path_length_m.toFixed(2)} m · ${data.grid_cells} grid cells`)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not plan the scene'); setMessage('') }
    finally { setBusy(false) }
  }

  const generate = async () => {
    if (!prompt.trim()) return
    setBusy(true); setError(''); setMessage('NVIDIA Kimodo is generating two motions, then Aura is fitting their root paths to the scene…')
    try {
      const response = await fetch('/aura-api/scene/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), count: 2, scene }),
      })
      const data = await response.json() as GenerationResponse
      if (!response.ok || !data.motions?.length || !data.plan) throw new Error(data.error || 'Scene generation did not return two fitted motions')
      setPlan(data.plan); setScene(data.plan.scene); setMotions(data.motions); setSelected(0)
      setMessage(`2 scene-fitted candidates ready · ${data.plan.path_length_m.toFixed(2)} m planned route`)
      window.dispatchEvent(new CustomEvent('aura:motion-generated', { detail: { ids: data.motions.map(m => m.id) } }))
    } catch (e) { setError(e instanceof Error ? e.message : 'Scene-aware generation failed'); setMessage('') }
    finally { setBusy(false) }
  }

  const status = useMemo(() => {
    if (splatName) return `${splatName} · visual context only`
    return 'Optional: load a standard .splat reconstruction for visual context'
  }, [splatName])

  return <main className="fx-page scene-lab-page">
    <section className="scene-lab-hero">
      <span className="card-kicker">SCENE LAB · GAUSSIAN SPLAT + G1</span>
      <h1>Put Aura inside<br />a real environment.</h1>
      <p>Use a Gaussian-splat capture for visual context, an explicit geometry proxy for planning, NVIDIA Kimodo for body motion, and Aura for human-preference learning. This first scene mode is intentionally limited to locomotion and navigation.</p>
    </section>

    <section className="scene-lab-grid">
      <div className="scene-lab-viewer-card">
        <div className="scene-lab-card-head"><div><span>01 · Visual scene</span><strong>Gaussian-splat preview</strong></div><div className="scene-lab-upload-actions"><button type="button" onClick={async () => {
          try {
            const response = await fetch(`${import.meta.env.BASE_URL || '/'}demo/aura-scene-demo.splat`)
            if (!response.ok) throw new Error(`Demo splat request failed (${response.status})`)
            setSplatBuffer(await response.arrayBuffer()); setSplatName('Synthetic demo room'); setError('')
          } catch (e) { setError(e instanceof Error ? e.message : 'Could not load demo splat') }
        }}>Demo scene</button><label className="scene-lab-upload">Load .splat<input type="file" accept=".splat" onChange={async e => {
          const file = e.currentTarget.files?.[0]
          if (!file) return
          try { setSplatBuffer(await file.arrayBuffer()); setSplatName(file.name); setError('') }
          catch { setError('Could not read the selected .splat file') }
        }} /></label></div></div>
        <SplatG1SceneViewer previewFile={previewUrl} splatBuffer={splatBuffer} plan={plan} />
        <div className="scene-lab-viewer-foot"><span>{status}</span><small>The browser preview is intentionally lightweight. Robot clearance and path planning use the geometry proxy below, not splat opacity.</small></div>
      </div>

      <aside className="scene-lab-controls">
        <div className="scene-lab-card-head"><div><span>02 · Planning proxy</span><strong>Floor + obstacle geometry</strong></div></div>
        <ScenePlanSvg scene={scene} path={plan?.path} />
        <div className="scene-lab-fields compact">
          <label>Start X<input value={scene.start[0]} type="number" step="0.1" onChange={e => setScene(s => ({ ...s, start: [n(e.target.value, s.start[0]), s.start[1]] }))} /></label>
          <label>Start Z<input value={scene.start[1]} type="number" step="0.1" onChange={e => setScene(s => ({ ...s, start: [s.start[0], n(e.target.value, s.start[1])] }))} /></label>
          <label>Goal X<input value={scene.goal[0]} type="number" step="0.1" onChange={e => setScene(s => ({ ...s, goal: [n(e.target.value, s.goal[0]), s.goal[1]] }))} /></label>
          <label>Goal Z<input value={scene.goal[1]} type="number" step="0.1" onChange={e => setScene(s => ({ ...s, goal: [s.goal[0], n(e.target.value, s.goal[1])] }))} /></label>
          <label>Obstacle X<input value={obstacle.x} type="number" step="0.1" onChange={e => updateObstacle({ x: n(e.target.value, obstacle.x) })} /></label>
          <label>Obstacle Z<input value={obstacle.z} type="number" step="0.1" onChange={e => updateObstacle({ z: n(e.target.value, obstacle.z) })} /></label>
          <label>Width<input value={obstacle.width} type="number" step="0.1" min="0.1" onChange={e => updateObstacle({ width: n(e.target.value, obstacle.width) })} /></label>
          <label>Depth<input value={obstacle.depth} type="number" step="0.1" min="0.1" onChange={e => updateObstacle({ depth: n(e.target.value, obstacle.depth) })} /></label>
        </div>
        <button className="scene-lab-secondary" disabled={busy} onClick={() => void planOnly()}>Plan route</button>
      </aside>
    </section>

    <section className="scene-lab-generate">
      <div><span className="card-kicker">03 · GENERATE FOR THIS SCENE</span><h2>Kimodo motion. Aura scene path.</h2><p>Kimodo still creates the body motion. Aura bends only the global root trajectory onto the planned route and preserves the G1 joint animation as much as possible.</p></div>
      <div className="scene-lab-prompt"><label htmlFor="scene-prompt">Navigation prompt</label><textarea id="scene-prompt" rows={3} value={prompt} onChange={e => setPrompt(e.target.value)} /><button disabled={busy || !prompt.trim()} onClick={() => void generate()}>{busy ? 'Working…' : 'Generate 2 scene motions'}</button></div>
      {message && <p className="scene-lab-message">{message}</p>}
      {error && <p className="scene-lab-error">{error}</p>}
    </section>

    {motions.length > 0 && <section className="scene-lab-results">
      <div className="scene-lab-results-head"><div><span className="card-kicker">04 · ROBOT TRAINING CANDIDATES</span><h2>Same scene. Two trajectories.</h2></div><p>These are saved back into the normal Aura motion library, so later A/B choices can train the same preference reward model.</p></div>
      <div className="scene-lab-motion-tabs">{motions.map((motion, index) => <button className={index === selected ? 'is-active' : ''} key={motion.id} onClick={() => setSelected(index)}><span>Motion {index === 0 ? 'A' : 'B'}</span><strong>{motion.name}</strong><small>{motion.scene_fit?.path_length_m?.toFixed(2) ?? plan?.path_length_m.toFixed(2)} m scene path</small></button>)}</div>
      {selectedMotion?.preview_file && <div className="scene-lab-result-player"><GeneratedG1RobotPreview key={selectedMotion.id} file={`/aura-api/files/${encodeURIComponent(selectedMotion.preview_file)}`} /></div>}
      <div className="scene-lab-evidence"><span>Generator</span><strong>NVIDIA Kimodo</strong><span>Scene adaptation</span><strong>{selectedMotion?.scene_fit?.method || 'Aura scene-fit'}</strong><span>Training meaning</span><strong>locomotion/navigation demonstration candidate</strong></div>
    </section>}

    <section className="scene-lab-caveat"><strong>Important boundary</strong><p>The Gaussian splat is the visual digital twin. The collision-aware route comes from the explicit floor/obstacle proxy. Aura does not claim physics, balance, contact-rich manipulation, or that Kimodo itself is scene-conditioned. For chair sitting, reaching, grasping or other geometry-dependent poses, a genuinely scene-conditioned generator would be the next step.</p></section>
  </main>
}
