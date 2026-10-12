import { useEffect, useMemo, useRef, useState } from 'react'
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
type CaptureBackend = { available: boolean; kind: string; label: string; detail?: string }
type CaptureStatus = {
  id: string
  name: string
  status: 'uploaded' | 'queued' | 'processing' | 'ready' | 'error'
  stage?: string
  progress?: number
  duration_s?: number | null
  bytes?: number
  splat_ready?: boolean
  splat_count?: number
  error?: string | null
  backend?: CaptureBackend
  log_tail?: string[]
}
type CaptureRuntime = { backend: CaptureBackend; max_upload_mb: number; capture_note?: string }

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

function formatBytes(bytes?: number) {
  if (!bytes) return '—'
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`
}

function preferredRecorderMime() {
  if (typeof MediaRecorder === 'undefined') return ''
  const options = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
  return options.find(type => MediaRecorder.isTypeSupported(type)) || ''
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
  const [splatRotation, setSplatRotation] = useState<[number, number, number]>([0, 0, 0])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const [cameraOpen, setCameraOpen] = useState(false)
  const [recording, setRecording] = useState(false)
  const [captureElapsed, setCaptureElapsed] = useState(0)
  const [captureRuntime, setCaptureRuntime] = useState<CaptureRuntime | null>(null)
  const [capture, setCapture] = useState<CaptureStatus | null>(null)
  const [captureBusy, setCaptureBusy] = useState(false)
  const cameraVideo = useRef<HTMLVideoElement>(null)
  const cameraStream = useRef<MediaStream | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const captureChunks = useRef<Blob[]>([])
  const captureStartedAt = useRef(0)
  const loadedCapture = useRef('')

  const selectedMotion = motions[selected] || null
  const previewUrl = selectedMotion?.preview_file ? `/aura-api/files/${encodeURIComponent(selectedMotion.preview_file)}` : null
  const obstacle = scene.obstacles[0] || { name: 'obstacle', x: 0, z: 0, width: 1, depth: 1 }

  useEffect(() => {
    void fetch('/aura-api/scene/capture/status', { cache: 'no-store' }).then(async response => {
      if (response.ok) setCaptureRuntime(await response.json() as CaptureRuntime)
    }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!cameraOpen) return
    let cancelled = false
    const open = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
          throw new Error('This browser does not expose camera recording. Use Chrome/Safari on localhost or HTTPS.')
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        })
        if (cancelled) { stream.getTracks().forEach(track => track.stop()); return }
        cameraStream.current = stream
        if (cameraVideo.current) {
          cameraVideo.current.srcObject = stream
          await cameraVideo.current.play().catch(() => undefined)
        }
        setError('')
      } catch (e) {
        setCameraOpen(false)
        setError(e instanceof Error ? e.message : 'Could not open camera')
      }
    }
    void open()
    return () => {
      cancelled = true
      cameraStream.current?.getTracks().forEach(track => track.stop())
      cameraStream.current = null
    }
  }, [cameraOpen])

  useEffect(() => {
    if (!recording) return
    const timer = window.setInterval(() => setCaptureElapsed((performance.now() - captureStartedAt.current) / 1000), 200)
    return () => window.clearInterval(timer)
  }, [recording])

  useEffect(() => () => {
    try { if (recorder.current?.state === 'recording') recorder.current.stop() } catch { /* unmount */ }
    cameraStream.current?.getTracks().forEach(track => track.stop())
  }, [])

  useEffect(() => {
    if (!capture?.id || !['queued', 'processing'].includes(capture.status)) return
    const captureId = capture.id
    let cancelled = false
    const poll = async () => {
      try {
        const response = await fetch(`/aura-api/scene/captures/${encodeURIComponent(captureId)}`, { cache: 'no-store' })
        const data = await response.json() as CaptureStatus & { error?: string }
        if (!response.ok) throw new Error(data.error || 'Could not read reconstruction status')
        if (cancelled) return
        setCapture(data)
        if (data.status === 'ready' && data.splat_ready && loadedCapture.current !== data.id) {
          const splatResponse = await fetch(`/aura-api/scene/captures/${encodeURIComponent(data.id)}/splat`, { cache: 'no-store' })
          if (!splatResponse.ok) throw new Error(`Splat download failed (${splatResponse.status})`)
          setSplatBuffer(await splatResponse.arrayBuffer())
          // Camera reconstructions are converted server-side from Nerfstudio Z-up to Aura/Three.js Y-up.
          setSplatRotation([0, 0, 0])
          setSplatName(`${data.name || 'Camera scene'} · camera capture`)
          loadedCapture.current = data.id
          setMessage(`Camera reconstruction loaded · ${data.splat_count?.toLocaleString() || 'Gaussian'} splats`)
          setCaptureBusy(false)
        } else if (data.status === 'error') {
          setCaptureBusy(false)
          setError(data.error || 'Gaussian reconstruction failed')
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not poll Gaussian reconstruction')
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2200)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [capture?.id, capture?.status])

  const updateObstacle = (patch: Partial<Obstacle>) => setScene(current => ({
    ...current,
    obstacles: [{ ...obstacle, ...patch }, ...current.obstacles.slice(1)],
  }))

  const beginReconstruction = async (captureId: string) => {
    setCaptureBusy(true)
    setError('')
    setMessage('Camera capture uploaded. Starting Gaussian reconstruction…')
    try {
      const response = await fetch(`/aura-api/scene/captures/${encodeURIComponent(captureId)}/reconstruct`, { method: 'POST' })
      const data = await response.json() as CaptureStatus & { error?: string }
      if (!response.ok) throw new Error(data.error || `Reconstruction start failed (${response.status})`)
      setCapture(data)
    } catch (e) {
      setCaptureBusy(false)
      setError(e instanceof Error ? e.message : 'Could not start Gaussian reconstruction')
    }
  }

  const uploadCapture = async (blob: Blob, duration: number) => {
    setCaptureBusy(true)
    setMessage('Uploading camera scan to the local Aura reconstruction service…')
    setError('')
    try {
      const controller = new AbortController()
      const timeout = window.setTimeout(() => controller.abort(), 120_000)
      const response = await fetch('/aura-api/scene/captures', {
        method: 'POST',
        headers: {
          'Content-Type': blob.type || 'video/webm',
          'X-Aura-Capture-Name': `Aura room ${new Date().toLocaleTimeString()}`,
          'X-Aura-Capture-Duration': duration.toFixed(3),
        },
        body: blob,
        signal: controller.signal,
      })
      window.clearTimeout(timeout)
      const data = await response.json() as CaptureStatus & { error?: string }
      if (!response.ok) throw new Error(data.error || `Capture upload failed (${response.status})`)
      setCapture(data)
      if (data.backend?.available ?? captureRuntime?.backend.available) {
        await beginReconstruction(data.id)
      } else {
        setCaptureBusy(false)
        setMessage('Camera scan saved locally. Install/configure Nerfstudio to reconstruct it into a Gaussian splat.')
      }
    } catch (e) {
      setCaptureBusy(false)
      const message = e instanceof Error ? e.message : 'Could not upload camera capture'
      setError(message === 'This operation was aborted' || message.toLowerCase().includes('abort')
        ? 'Camera upload timed out after 120 seconds. Check that the Aura API on port 8765 is running and inspect .aura-logs/library.log.'
        : message)
    }
  }

  const startCameraRecording = () => {
    const stream = cameraStream.current
    if (!stream) { setError('Camera is still starting.'); return }
    const mime = preferredRecorderMime()
    captureChunks.current = []
    try {
      let next: MediaRecorder
      try {
        next = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 4_000_000 } : { videoBitsPerSecond: 4_000_000 })
      } catch {
        next = new MediaRecorder(stream)
      }
      next.ondataavailable = event => { if (event.data.size) captureChunks.current.push(event.data) }
      next.onerror = () => setError('The browser camera recorder reported an error.')
      next.onstop = () => {
        const duration = Math.max(0, (performance.now() - captureStartedAt.current) / 1000)
        const blob = new Blob(captureChunks.current, { type: next.mimeType || mime || 'video/webm' })
        setRecording(false)
        setCaptureElapsed(duration)
        cameraStream.current?.getTracks().forEach(track => track.stop())
        cameraStream.current = null
        setCameraOpen(false)
        void uploadCapture(blob, duration)
      }
      recorder.current = next
      captureStartedAt.current = performance.now()
      setCaptureElapsed(0)
      setRecording(true)
      setMessage('Scanning… move slowly around the static scene and keep surfaces in view from overlapping angles.')
      next.start(1000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start camera recording')
    }
  }

  const stopCameraRecording = () => {
    if (recorder.current?.state === 'recording') recorder.current.stop()
  }

  const closeCamera = () => {
    if (recording) return
    cameraStream.current?.getTracks().forEach(track => track.stop())
    cameraStream.current = null
    setCameraOpen(false)
  }

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
    return 'Capture a room with the camera, load a .splat, or use the synthetic demo scene'
  }, [splatName])

  return <main className="fx-page scene-lab-page">
    <section className="scene-lab-hero">
      <span className="card-kicker">SCENE LAB · CAMERA → GAUSSIAN SPLAT → G1</span>
      <h1>Capture a room.<br />Train inside it.</h1>
      <p>Record a static environment with your camera. Aura can pass that video through a Gaussian-splat reconstruction backend, load the result directly into Scene Lab, plan against an explicit geometry proxy, and place Kimodo-generated G1 motion inside the captured environment.</p>
    </section>

    <section className="scene-lab-grid">
      <div className="scene-lab-viewer-card">
        <div className="scene-lab-card-head"><div><span>01 · Visual scene</span><strong>Camera capture / Gaussian-splat preview</strong></div><div className="scene-lab-upload-actions">
          <button type="button" className="is-camera" onClick={() => { setCameraOpen(true); setError('') }} disabled={recording}>◉ Capture room</button>
          <button type="button" onClick={async () => {
            try {
              const response = await fetch(`${import.meta.env.BASE_URL || '/'}demo/aura-scene-demo.splat`)
              if (!response.ok) throw new Error(`Demo splat request failed (${response.status})`)
              setSplatBuffer(await response.arrayBuffer()); setSplatRotation([0, 0, 0]); setSplatName('Synthetic demo room'); setError('')
            } catch (e) { setError(e instanceof Error ? e.message : 'Could not load demo splat') }
          }}>Demo scene</button>
          <label className="scene-lab-upload">Load .splat<input type="file" accept=".splat" onChange={async e => {
            const file = e.currentTarget.files?.[0]
            if (!file) return
            try {
              setSplatBuffer(await file.arrayBuffer())
              // Common standalone .splat exports arrive inverted relative to Aura's Y-up scene.
              // Start manual files with a 180° X correction; the toolbar below can adjust other conventions.
              setSplatRotation([Math.PI, 0, 0])
              setSplatName(file.name)
              setMessage('Loaded .splat with Aura upright correction. Use Scene orientation if this file uses another coordinate convention.')
              setError('')
            }
            catch { setError('Could not read the selected .splat file') }
          }} /></label>
        </div></div>

        {splatBuffer && !cameraOpen && <div className="scene-splat-orientation">
          <span>Scene orientation</span>
          <button type="button" onClick={() => setSplatRotation(([x, y, z]) => [x + Math.PI, y, z])}>Flip 180°</button>
          <button type="button" onClick={() => setSplatRotation(([x, y, z]) => [x, y + Math.PI / 2, z])}>Rotate +90°</button>
          <button type="button" onClick={() => setSplatRotation(([x, y, z]) => [x, y - Math.PI / 2, z])}>Rotate −90°</button>
          <button type="button" onClick={() => setSplatRotation([0, 0, 0])}>Raw axes</button>
          <small>X {(splatRotation[0] * 180 / Math.PI).toFixed(0)}° · Y {(splatRotation[1] * 180 / Math.PI).toFixed(0)}°</small>
        </div>}

        {cameraOpen ? <div className="scene-camera-capture">
          <video ref={cameraVideo} playsInline muted autoPlay />
          <div className="scene-camera-hud">
            <div><span className={recording ? 'scene-camera-dot is-recording' : 'scene-camera-dot'} /> <strong>{recording ? `SCANNING ${captureElapsed.toFixed(1)}s` : 'CAMERA READY'}</strong></div>
            <p>Walk slowly around the scene for about 20–40 seconds. Keep the room static, avoid motion blur, and revisit surfaces from overlapping angles.</p>
            <div className="scene-camera-actions">
              {!recording ? <button type="button" className="is-primary" onClick={startCameraRecording}>Start scan</button> : <button type="button" className="is-stop" onClick={stopCameraRecording}>Stop &amp; reconstruct</button>}
              <button type="button" onClick={closeCamera} disabled={recording}>Cancel</button>
            </div>
          </div>
        </div> : <SplatG1SceneViewer previewFile={previewUrl} splatBuffer={splatBuffer} plan={plan} splatRotation={splatRotation} />}

        {capture && <div className={`scene-capture-progress is-${capture.status}`}>
          <div className="scene-capture-progress-head"><div><span>Camera reconstruction</span><strong>{capture.stage || capture.status}</strong></div><small>{capture.duration_s ? `${capture.duration_s.toFixed(1)}s scan · ` : ''}{formatBytes(capture.bytes)}</small></div>
          <div className="scene-capture-progress-bar"><i style={{ width: `${Math.round((capture.progress ?? 0) * 100)}%` }} /></div>
          <div className="scene-capture-progress-foot"><span>{capture.backend?.label || captureRuntime?.backend.label || 'Reconstruction backend'}</span><small>{capture.status === 'ready' ? `${capture.splat_count?.toLocaleString() || ''} splats loaded automatically` : capture.error || capture.backend?.detail || 'Processing continues in the background.'}</small></div>
          {(capture.status === 'uploaded' || capture.status === 'error') && (capture.backend?.available ?? captureRuntime?.backend.available) && <button type="button" className="scene-capture-retry" disabled={captureBusy} onClick={() => void beginReconstruction(capture.id)}>{captureBusy ? 'Starting…' : 'Reconstruct this capture'}</button>}
        </div>}

        <div className="scene-lab-viewer-foot"><span>{status}</span><small>The splat is visual context. A normal phone video has no guaranteed metric scale, so robot clearance and path planning use the geometry proxy below—not splat opacity.</small></div>
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
        <div className="scene-capture-backend"><span>Camera → splat backend</span><strong>{captureRuntime?.backend.available ? 'Ready' : 'Setup required'}</strong><small>{captureRuntime?.backend.label || 'Checking local Aura API…'}{captureRuntime?.backend.detail ? ` · ${captureRuntime.backend.detail}` : ''}</small></div>
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

    <section className="scene-lab-caveat"><strong>Important boundary</strong><p>The camera reconstruction is a visual digital twin. Monocular video reconstruction has arbitrary scale unless you calibrate it, and Aura does not infer trustworthy collision geometry from Gaussian opacity. The collision-aware route therefore comes from the explicit floor/obstacle proxy. This remains locomotion/navigation root-path fitting—not physics, balance validation, or contact-rich scene-conditioned motion generation.</p></section>
  </main>
}
