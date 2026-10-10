import { useEffect, useRef, useState } from 'react'
import { GeneratedMotionPreview } from './GeneratedMotionPreview'
import { GeneratedG1RobotPreview } from './GeneratedG1RobotPreview'
import { AIST_REFERENCE_MOTIONS, STUDY_CLIP_SECONDS } from '../aistReferenceMotions'
import { DEFAULT_VIEW_POSE, XBotScene } from './XBotScene'
import type { ViewPose } from './XBotScene'
import {
  commitG1ScreenOnChain, connectWallet, detectedWallets, explorerTransactionUrl,
} from '../solana'
import type { G1ScreenReceipt, WalletConnection, WalletKind } from '../solana'

type Foot = { foot: string; contact_frame_fraction: number; contact_intervals: number; contact_horizontal_speed_m_s: number | null }
type Report = {
  method: string; scope: string; frames: number; fps: number; duration_s: number;
  root_horizontal_displacement_m: number; root_horizontal_path_m: number;
  root_height_range_m: number; estimated_floor_y_m: number;
  estimated_below_floor_fraction: number; near_floor_toe_speed_m_s: number | null;
  feet: Foot[]; notes: string[];
}
type Result = { native_sha256: string; report_sha256: string; report: Report }
type Motion = {
  id: string; name: string; model: string; created_at: string; frames: number;
  fps: number; native_file: string; preview_file: string | null;
  preview_error?: string | null; evaluation_error?: string | null;
  kinematic_evaluation?: Result;
  source?: string;
}
type Review = { score: number; note: string; reviewed_at: string }
type PriorState = { model: null | { version: string; caveat: string }; scores: { id: string; prior_score: number }[] }
type RewardState = { model: null | { version: string; total_comparison_count: number; motion_count: number; caveat: string }; scores: { id: string; reward: number; rank: number; rank_percentile: number }[] }
const STORAGE = 'aura-generated-motion-reviews-v1'
const CHAIN_STORAGE = 'aura-g1-screen-receipts-v1'
function stored<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) || '') as T } catch { return fallback }
}
function metric(value: number | null | undefined, suffix = '') {
  return value === null || value === undefined ? 'Not measurable' : `${value.toFixed(3)}${suffix}`
}
function saveJson(filename: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
export function GeneratedMotionLibrary() {
  const [motions, setMotions] = useState<Motion[]>([])
  const [selected, setSelected] = useState('')
  const [compare, setCompare] = useState('')
  const [reviews, setReviews] = useState<Record<string, Review>>(() => stored(STORAGE, {}))
  const [receipts, setReceipts] = useState<G1ScreenReceipt[]>(() => stored(CHAIN_STORAGE, []))
  const [score, setScore] = useState(3)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [chainError, setChainError] = useState('')
  const [busy, setBusy] = useState(false)
  const [wallet, setWallet] = useState<WalletConnection | null>(null)
  const [aistProgress, setAistProgress] = useState(0)
  const [aistView, setAistView] = useState<ViewPose>({ ...DEFAULT_VIEW_POSE })
  const [prior, setPrior] = useState<PriorState>({ model: null, scores: [] })
  const [reward, setReward] = useState<RewardState>({ model: null, scores: [] })
  const newestMotionRef = useRef<string | null>(null)
  const libraryInitializedRef = useRef(false)
  const refresh = async () => {
    try {
      const response = await fetch('/aura-api/motions', { cache: 'no-store' })
      if (!response.ok) throw new Error('Local Aura library server is not running on port 8765.')
      const items = await response.json() as Motion[]
      if (!Array.isArray(items)) throw new Error('Invalid motion-library response')
      const visibleItems = items.filter(item => item.source !== 'starter_reference')
      setMotions(visibleItems)
      const newest = visibleItems[0]?.id || ''
      setSelected(previous => {
        const starterDefault = `aist:${AIST_REFERENCE_MOTIONS[0].id}`
        if (!libraryInitializedRef.current) {
          libraryInitializedRef.current = true
          newestMotionRef.current = newest || null
          return newest || starterDefault
        }
        // Do not steal focus from a starter clip while the user is browsing it.
        if (previous.startsWith('aist:')) {
          newestMotionRef.current = newest || null
          return previous
        }
        if (newest && newest !== newestMotionRef.current) {
          newestMotionRef.current = newest
          setScore(reviews[newest]?.score || 3)
          setNote(reviews[newest]?.note || '')
          return newest
        }
        newestMotionRef.current = newest || null
        return visibleItems.some(x => x.id === previous) ? previous : (newest || starterDefault)
      })
      try {
        const [priorResponse, rewardResponse] = await Promise.all([
          fetch('/aura-api/prior', { cache: 'no-store' }),
          fetch('/aura-api/reward', { cache: 'no-store' }),
        ])
        if (priorResponse.ok) setPrior(await priorResponse.json() as PriorState)
        if (rewardResponse.ok) setReward(await rewardResponse.json() as RewardState)
      } catch { /* learned models are optional until enough preference data exists */ }
      setError('')
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not connect to the motion library.') }
  }
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 1200); return () => clearInterval(timer) }, [])
  useEffect(() => {
    if (!selected.startsWith('aist:')) return
    let raf = 0
    let previous = performance.now()
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.1)
      previous = now
      setAistProgress(value => (value + (dt / STUDY_CLIP_SECONDS) * 100) % 100)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [selected])
  const current = motions.find(item => item.id === selected)
  const currentAist = AIST_REFERENCE_MOTIONS.find(item => `aist:${item.id}` === selected)
  const other = motions.find(item => item.id === compare && item.id !== selected)
  const result = current?.kinematic_evaluation
  const matchingReceipts = receipts.filter(r => r.motionId === current?.id && r.nativeSha256 === result?.native_sha256 && r.reportSha256 === result?.report_sha256)
  const saveReview = () => {
    if (!current) return
    const next = { ...reviews, [current.id]: { score, note, reviewed_at: new Date().toISOString() } }
    setReviews(next); localStorage.setItem(STORAGE, JSON.stringify(next))
  }
  const attest = async () => {
    if (!current || !result) return
    setChainError(''); setBusy(true)
    try {
      let signer = wallet
      if (!signer) {
        const available = detectedWallets().find(x => x.installed)
        if (!available) throw new Error('Install Phantom or Solflare, enable devnet and fund the wallet with devnet SOL.')
        signer = await connectWallet(available.kind as WalletKind)
        setWallet(signer)
      }
      const receipt = await commitG1ScreenOnChain(signer, current.id, result.native_sha256, result.report_sha256, result.report.method)
      const next = [...receipts, receipt]
      setReceipts(next); localStorage.setItem(CHAIN_STORAGE, JSON.stringify(next))
    } catch (e) { setChainError(e instanceof Error ? e.message : 'Could not submit attestation.') }
    finally { setBusy(false) }
  }
  const priorScore = (id: string) => prior.scores.find(item => item.id === id)?.prior_score
  const rewardScore = (id: string) => reward.scores.find(item => item.id === id)
  const fields: { key: keyof Report; label: string; unit?: string; detail: string }[] = [
    { key: 'near_floor_toe_speed_m_s', label: 'Near-floor toe speed', unit: ' m/s', detail: 'Lower may indicate less apparent sliding; contact is height-estimated.' },
    { key: 'estimated_below_floor_fraction', label: 'Below-estimated-floor fraction', detail: 'Diagnostic only: floor is inferred from the motion, not a collision model.' },
    { key: 'root_horizontal_displacement_m', label: 'Root displacement', unit: ' m', detail: 'Horizontal displacement, not a quality score.' },
    { key: 'root_horizontal_path_m', label: 'Root path length', unit: ' m', detail: 'Total horizontal trajectory length.' },
    { key: 'root_height_range_m', label: 'Root vertical range', unit: ' m', detail: 'Vertical movement of pelvis, not a stability assessment.' },
  ]
  return (
    <section className="generated-library" aria-label="Generated G1 motion library and evaluation">
      <div className="generated-library-heading"><div><div className="card-kicker">02 / BROWSE &amp; INSPECT</div><h2>Generate once. <em>Inspect the result.</em></h2><p>Start with the bundled AIST++ reference motions or browse Aura generations as soon as they arrive. Click any motion to play it in the viewer and inspect its available details.</p></div><button onClick={() => void refresh()}>↻ Refresh library</button></div>
      {error && <div className="generated-library-warning">{error} Generated motions need <code>python aura_library_server.py</code>; the preloaded AIST++ references remain available.</div>}
      <div className="generated-library-layout"><aside className="generated-library-list">
        <div className="generated-library-group-label"><span>PRELOADED AIST++</span><small>{AIST_REFERENCE_MOTIONS.length} references</small></div>
        {AIST_REFERENCE_MOTIONS.map(m => { const key = `aist:${m.id}`; return <button key={key} className={key === selected ? 'generated-motion selected starter' : 'generated-motion starter'} onClick={() => { setSelected(key); setAistProgress(0); setAistView({ ...DEFAULT_VIEW_POSE }) }}><strong>{m.label}</strong><small>AIST++ · retargeted to G1 · {STUDY_CLIP_SECONDS}s clip</small><span>Reference motion · click to play</span></button> })}
        <div className="generated-library-group-label aura-generations"><span>AURA GENERATIONS</span><small>{motions.length} saved</small></div>
        {motions.length ? motions.map(m => <button key={m.id} className={m.id === selected ? 'generated-motion selected' : 'generated-motion'} onClick={() => { setSelected(m.id); setScore(reviews[m.id]?.score || 3); setNote(reviews[m.id]?.note || ''); window.dispatchEvent(new CustomEvent('aura:select-motion', { detail: { id: m.id } })) }}><strong>{m.name}</strong><small>{m.model} · {m.frames} frames · {new Date(m.created_at).toLocaleString()}</small><span>{rewardScore(m.id) ? `Aura #${rewardScore(m.id)!.rank} · reward ${rewardScore(m.id)!.reward.toFixed(2)} · ` : priorScore(m.id) !== undefined ? `Learned prior ${priorScore(m.id)!.toFixed(2)} · ` : ''}{m.kinematic_evaluation ? '✓ G1 screening ready' : m.evaluation_error ? 'Evaluation unavailable' : 'Saved motion · click to play'}</span></button>) : <p>No Aura generations yet. You can still explore the AIST++ references above.</p>}
      </aside><div className="generated-library-detail">
        {currentAist ? <div className="aist-library-detail"><div className="generated-library-detail-title"><strong>{currentAist.label}</strong><span>AIST++ reference · retargeted to G1</span></div><div className="aist-library-player"><XBotScene embodiment="g1" side="A" quality="reference" motionFile={currentAist.file} motionUrl={currentAist.url} degradationSeed={currentAist.seed} startOffsetSeconds={currentAist.startOffsetSeconds} paused={false} playhead={aistProgress} autoRotate showTrails showLandmarks={false} resetViewSignal={0} viewPose={aistView} onViewPoseChange={setAistView} /></div><div className="aist-library-meta"><span>Preloaded onboarding/reference motion</span><span>{STUDY_CLIP_SECONDS}s looping excerpt</span><span>Does not enter Aura's generated-motion training set</span></div><p className="generated-library-disclaimer">This is an AIST++ reference clip retargeted to the G1 viewer so visitors can browse motion immediately. Generate an Aura batch above to create same-prompt candidates for preference learning.</p></div> : current ? <><div className="generated-library-detail-title"><strong>{current.name}</strong><span>{rewardScore(current.id) ? `Aura reward ${rewardScore(current.id)!.reward.toFixed(3)} · rank #${rewardScore(current.id)!.rank}` : priorScore(current.id) !== undefined ? `AMP-inspired prior ${priorScore(current.id)!.toFixed(2)}` : 'Aura reward model not trained yet'}</span><a href={`/aura-api/files/${encodeURIComponent(current.native_file)}`} download>Native NPZ ↗</a></div>
          {current.preview_file ? <div className={other?.preview_file?.endsWith('.g1.json') ? 'generated-playback-grid comparing' : 'generated-playback-grid'}>
            <div className="generated-playback-card"><div className="generated-playback-label"><span>SELECTED / A</span><strong>{current.name}</strong></div>{current.preview_file.endsWith('.g1.json') ? <GeneratedG1RobotPreview key={current.id} file={`/aura-api/files/${encodeURIComponent(current.preview_file)}`} /> : <GeneratedMotionPreview file={`/aura-api/files/${encodeURIComponent(current.preview_file)}`} />}</div>
            {other?.preview_file?.endsWith('.g1.json') && <div className="generated-playback-card"><div className="generated-playback-label"><span>COMPARISON / B</span><strong>{other.name}</strong></div><GeneratedG1RobotPreview key={other.id} file={`/aura-api/files/${encodeURIComponent(other.preview_file)}`} compact /></div>}
          </div> : <div className="generated-preview-placeholder">Preview unavailable; native NPZ remains saved. {current.preview_error}</div>}
          {result ? <div className="g1-evaluation"><div className="g1-heading"><h3>G1 kinematic screening</h3><small>Calculated from saved NPZ · {result.report.method}</small></div>
            <div className="g1-metrics">{fields.map(f => <div className="g1-metric" key={f.key}><small>{f.label}</small><strong>{metric(result.report[f.key] as number | null, f.unit)}</strong><p>{f.detail}</p></div>)}</div>
            <p className="g1-disclaimer">These results are kinematic heuristics—not physics simulation, physical balance, task success, prompt compliance or safety certification. Estimated floor: {result.report.estimated_floor_y_m.toFixed(3)} m (5th percentile of toe heights).</p>
            <div className="g1-hashes"><span>NPZ SHA-256: <code>{result.native_sha256}</code></span><span>Report SHA-256: <code>{result.report_sha256}</code></span></div>
            <div className="g1-actions"><button onClick={() => saveJson(`aura-g1-${current.id}-evidence.json`, { motion_id: current.id, motion_name: current.name, result, receipt: matchingReceipts.at(-1) || null })}>Export verification evidence</button><button disabled={busy} onClick={() => void attest()}>{busy ? 'Waiting for wallet…' : 'Sign & anchor on Solana devnet'}</button></div>
            {matchingReceipts.length > 0 && <p className="g1-chain-success">✓ Wallet-signed memo confirmed · <a href={explorerTransactionUrl(matchingReceipts[matchingReceipts.length - 1].signature)} target="_blank" rel="noreferrer">View devnet transaction ↗</a></p>}
            {chainError && <p className="generated-library-warning">{chainError}</p>}
          </div> : <p className="generated-library-warning">{current.evaluation_error || 'Automatic G1 screening is available for G1 NPZ motions only.'}</p>}
          <div className="g1-compare"><h3>Compare two generated motions</h3><p className="g1-compare-note">Choose a second NVIDIA Kimodo generation to play both G1 motions side by side. New generations automatically become motion A and start playing when Kimodo finishes.</p><label htmlFor="compare-motion">Second motion</label><select id="compare-motion" value={other?.id || ''} onChange={e => setCompare(e.target.value)}><option value="">Choose a second G1 motion</option>{motions.filter(m => m.id !== current.id && m.kinematic_evaluation).map(m => <option key={m.id} value={m.id}>{m.name} · {m.id}</option>)}</select>
            {result && other?.kinematic_evaluation && <table><thead><tr><th>Measure</th><th>Selected</th><th>Comparison</th></tr></thead><tbody>{fields.map(f => <tr key={f.key}><td>{f.label}</td><td>{metric(result.report[f.key] as number | null, f.unit)}</td><td>{metric(other.kinematic_evaluation!.report[f.key] as number | null, f.unit)}</td></tr>)}</tbody></table>}
            <small>Measurements are descriptive: different tasks and desired motion trajectories require different criteria.</small>
          </div>
          <div className="generated-review"><label htmlFor="generated-review-score">Human motion review (1–5)</label><select id="generated-review-score" value={score} onChange={e => setScore(Number(e.target.value))}>{[1,2,3,4,5].map(n => <option value={n} key={n}>{n}/5</option>)}</select><textarea value={note} onChange={e => setNote(e.target.value)} placeholder="Observations and task-specific judgment…" rows={2}/><button onClick={saveReview}>Save review</button>{reviews[current.id] && <span>Saved locally · {reviews[current.id].score}/5</span>}</div>
        </> : <p>Select a motion to inspect.</p>}
      </div></div>
      <p className="generated-library-disclaimer">Robot-native G1 playback and kinematic screening are diagnostic tools, not physics validation. SHA-256 receipts prove that particular bytes were attested by a wallet—not the correctness of a motion or the identity of its original creator. Human reviews and receipt history are stored in this browser; back them up by exporting evidence.</p>
    </section>
  )
}
