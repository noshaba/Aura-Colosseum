import { useCallback, useEffect, useMemo, useState } from 'react'
import { GeneratedG1RobotPreview } from './GeneratedG1RobotPreview'
import { GeneratingDanceStage } from './GeneratingDanceStage'
import { AIST_REFERENCE_MOTIONS, STUDY_CLIP_SECONDS } from '../aistReferenceMotions'
import { DEFAULT_VIEW_POSE, XBotScene } from './XBotScene'
import type { ViewPose } from './XBotScene'

type Motion = {
  id: string
  name: string
  model: string
  created_at: string
  frames: number
  fps: number
  native_file: string
  preview_file: string | null
  batch_id?: string
  candidate_index?: number
  candidate_count?: number
  generation_seed?: number
  native_sha256?: string
}

type GeneratorExample = {
  id: string
  label: string
  prompt: string
  has_constraints: boolean
}

type GeneratorStatus = {
  model: string
  model_loaded: boolean
  busy: boolean
  cuda_available: boolean
  cuda_name?: string | null
  default_duration: number
  default_diffusion_steps: number
  seed_policy?: string
  last_error?: string | null
}

type TrainingData = {
  stored_comparison_count: number
  valid_comparison_count: number
  unique_pair_count: number
  unique_motion_count: number
  duplicate_comparison_count: number
  rejected_comparison_count: number
  reward_trainable: boolean
}
type PreferenceRecord = { id: number; winner_id: string; training_data?: TrainingData }
type MotionPrior = {
  model: null | { version: string; vote_count: number; motion_count: number; heldout_motion_accuracy: number | null; caveat: string }
  scores: { id: string; prior_score: number }[]
  error?: string
}
type RewardState = {
  model: null | {
    version: string
    total_comparison_count: number
    motion_count: number
    parameter_count: number
    heldout_pair_accuracy: number | null
    caveat: string
  }
  scores: { id: string; reward: number; rank: number; rank_percentile: number }[]
  error?: string
}

const AIST_STARTER_STORAGE = 'aura:aist-starter-preferences:v1'
const EVALUATOR_STORAGE = 'aura:generated-evaluator-id:v1'

function localEvaluatorId() {
  try {
    const existing = localStorage.getItem(EVALUATOR_STORAGE)
    if (existing) return existing
    const id = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? `browser:${crypto.randomUUID()}`
      : `browser:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
    localStorage.setItem(EVALUATOR_STORAGE, id)
    return id
  } catch {
    return `browser:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  }
}

const FALLBACK_PROMPTS: GeneratorExample[] = [
  { id: 'fallback-walk', label: 'Forward walk', prompt: 'A humanoid robot walks forward for several steps and comes to a controlled stop.', has_constraints: false },
  { id: 'fallback-reach', label: 'Reach low', prompt: 'A humanoid robot walks forward and reaches down to pick something up.', has_constraints: false },
  { id: 'fallback-balance', label: 'Recover balance', prompt: 'A humanoid robot stumbles while walking forward but recovers its balance.', has_constraints: false },
]

function latestComparableBatch(items: Motion[]) {
  const groups = new Map<string, Motion[]>()
  for (const motion of items) {
    if (!motion.batch_id) continue
    const group = groups.get(motion.batch_id) || []
    group.push(motion)
    groups.set(motion.batch_id, group)
  }
  for (const motion of items) {
    if (!motion.batch_id) continue
    const group = groups.get(motion.batch_id) || []
    if (group.length >= 2) return [...group].sort((a, b) => (a.candidate_index || 0) - (b.candidate_index || 0))
  }
  // Backward-compatible fallback: find two recent motions with the same exact prompt.
  for (let i = 0; i < items.length; i += 1) {
    const group = items.filter(item => item.name === items[i].name)
    if (group.length >= 2) return group.slice(0, 2)
  }
  return []
}

export function AuraGeneratorStage() {
  const [motions, setMotions] = useState<Motion[]>([])
  const [batch, setBatch] = useState<Motion[]>([])
  const [leftId, setLeftId] = useState('')
  const [rightId, setRightId] = useState('')
  const [nextIndex, setNextIndex] = useState(2)
  const [winnerId, setWinnerId] = useState('')
  const [examples, setExamples] = useState<GeneratorExample[]>([])
  const [selectedExample, setSelectedExample] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('A humanoid robot takes several careful steps forward while maintaining balance.')
  const [candidateCount, setCandidateCount] = useState(2)
  const [status, setStatus] = useState<GeneratorStatus | null>(null)
  const [generating, setGenerating] = useState(false)
  const [voting, setVoting] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [, setPrior] = useState<MotionPrior>({ model: null, scores: [] })
  const [priorTraining, setPriorTraining] = useState(false)
  const [reward, setReward] = useState<RewardState>({ model: null, scores: [] })
  const [rewardTraining, setRewardTraining] = useState(false)
  const [evaluatorId] = useState(() => localEvaluatorId())
  const [starterLeftIndex, setStarterLeftIndex] = useState(0)
  const [starterRightIndex, setStarterRightIndex] = useState(1)
  const [starterNextIndex, setStarterNextIndex] = useState(2)
  const [starterWinnerIndex, setStarterWinnerIndex] = useState<number | null>(null)
  const [starterProgress, setStarterProgress] = useState(0)
  const [starterLeftView, setStarterLeftView] = useState<ViewPose>({ ...DEFAULT_VIEW_POSE })
  const [starterRightView, setStarterRightView] = useState<ViewPose>({ ...DEFAULT_VIEW_POSE })

  const installBatch = useCallback((items: Motion[]) => {
    if (items.length < 2) return
    const ordered = [...items].sort((a, b) => (a.candidate_index || 0) - (b.candidate_index || 0))
    setBatch(ordered)
    setLeftId(ordered[0].id)
    setRightId(ordered[1].id)
    setNextIndex(2)
    setWinnerId('')
  }, [])

  const refreshMotions = useCallback(async (preferNewestBatch = false) => {
    const response = await fetch('/aura-api/motions', { cache: 'no-store' })
    if (!response.ok) throw new Error('Aura motion library is unavailable.')
    const items = await response.json() as Motion[]
    const g1 = Array.isArray(items) ? items.filter(item => item.preview_file?.endsWith('.g1.json')) : []
    setMotions(g1)
    if (preferNewestBatch) {
      const latest = latestComparableBatch(g1)
      if (latest.length >= 2) installBatch(latest)
    }
  }, [batch.length, installBatch])

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch('/aura-api/generator/status', { cache: 'no-store' })
      if (response.ok) setStatus(await response.json() as GeneratorStatus)
    } catch { /* optional while the local API starts */ }
  }, [])

  const refreshPrior = useCallback(async () => {
    try {
      const response = await fetch('/aura-api/prior', { cache: 'no-store' })
      if (response.ok) setPrior(await response.json() as MotionPrior)
    } catch { /* learned prior is optional until enough generated comparisons exist */ }
  }, [])

  const refreshReward = useCallback(async () => {
    try {
      const response = await fetch('/aura-api/reward', { cache: 'no-store' })
      if (response.ok) setReward(await response.json() as RewardState)
    } catch { /* reward model is optional until enough generated comparisons exist */ }
  }, [])

  const updateReward = useCallback(async () => {
    if (rewardTraining) return
    setRewardTraining(true)
    try {
      const response = await fetch('/aura-api/reward/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      if (response.ok) setReward(await response.json() as RewardState)
      else await refreshReward()
    } catch { /* never interrupt the human-rating loop if training is not ready yet */ }
    finally { setRewardTraining(false) }
  }, [rewardTraining, refreshReward])

  const updatePrior = useCallback(async () => {
    if (priorTraining) return
    setPriorTraining(true)
    try {
      const response = await fetch('/aura-api/prior/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      if (response.ok) setPrior(await response.json() as MotionPrior)
      else await refreshPrior()
    } catch { /* do not interrupt rating if the learned prior is not trainable yet */ }
    finally { setPriorTraining(false) }
  }, [priorTraining, refreshPrior])

  useEffect(() => {
    void refreshMotions().catch(() => undefined)
    void refreshStatus()
    void refreshPrior()
    void refreshReward()
    fetch('/aura-api/generator/examples', { cache: 'no-store' })
      .then(r => r.ok ? r.json() : Promise.reject(new Error('examples unavailable')))
      .then((data: { examples?: GeneratorExample[] }) => setExamples(data.examples?.length ? data.examples : FALLBACK_PROMPTS))
      .catch(() => setExamples(FALLBACK_PROMPTS))
    const timer = window.setInterval(() => void refreshStatus(), 3000)
    return () => window.clearInterval(timer)
  }, [refreshMotions, refreshStatus, refreshPrior, refreshReward])

  useEffect(() => {
    if (generating || batch.length >= 2) return
    let raf = 0
    let previous = performance.now()
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.1)
      previous = now
      setStarterProgress(value => (value + (dt / STUDY_CLIP_SECONDS) * 100) % 100)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [generating, batch.length])

  const left = useMemo(() => batch.find(item => item.id === leftId), [batch, leftId])
  const right = useMemo(() => batch.find(item => item.id === rightId), [batch, rightId])
  const starterLeft = AIST_REFERENCE_MOTIONS[starterLeftIndex]
  const starterRight = AIST_REFERENCE_MOTIONS[starterRightIndex]
  const shownExamples = examples.slice(0, 8)

  const chooseExample = (example: GeneratorExample) => {
    setSelectedExample(example.id.startsWith('fallback-') ? null : example.id)
    setPrompt(example.prompt)
    setMessage(example.has_constraints ? 'Example selected · its saved NVIDIA Kimodo constraints will be applied to every candidate.' : 'Example prompt selected.')
    setError('')
  }

  const generate = async () => {
    if (!prompt.trim() && !selectedExample) return
    setGenerating(true)
    setError('')
    setWinnerId('')
    setMessage(status?.model_loaded ? `Aura is generating ${candidateCount} candidates…` : `Aura is loading the G1 model, then generating ${candidateCount} candidates…`)
    try {
      const response = await fetch('/aura-api/generator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), example_id: selectedExample, count: candidateCount }),
      })
      const data = await response.json() as { motions?: Motion[]; motion?: Motion; error?: string; generation?: { batch_id?: string } }
      const generated = data.motions || (data.motion ? [data.motion] : [])
      if (!response.ok || generated.length < 2) throw new Error(data.error || 'Aura did not receive at least two generated candidates.')
      setMotions(previous => [...generated, ...previous.filter(item => !generated.some(next => next.id === item.id))])
      installBatch(generated)
      setSelectedExample(null)
      setMessage(`${generated.length} candidates ready. Pick the better motion to teach Aura.`)
      window.dispatchEvent(new CustomEvent('aura:motion-generated', { detail: { ids: generated.map(item => item.id), batch_id: data.generation?.batch_id } }))
      void refreshStatus()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Aura generation failed.')
      setMessage('')
    } finally {
      setGenerating(false)
    }
  }

  const vote = async (preferredId: string) => {
    if (!left || !right || voting || winnerId) return
    setVoting(true)
    setError('')
    try {
      const response = await fetch('/aura-api/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          left_id: left.id,
          right_id: right.id,
          winner_id: preferredId,
          evaluator_id: evaluatorId,
          context: 'Generated batch comparison: prefer the motion that better satisfies the shared prompt.',
        }),
      })
      const data = await response.json() as PreferenceRecord & { error?: string }
      if (!response.ok) throw new Error(data.error || 'Could not save preference.')
      const trainingData = data.training_data
      if (trainingData?.reward_trainable) {
        void updateReward().then(() => updatePrior())
      } else {
        void refreshReward()
      }
      if (nextIndex < batch.length) {
        const next = batch[nextIndex]
        setLeftId(preferredId)
        setRightId(next.id)
        setNextIndex(value => value + 1)
        setMessage(`Preference #${data.id} saved. The winner now faces candidate ${nextIndex + 1} of ${batch.length}.`)
      } else {
        setWinnerId(preferredId)
        setMessage(trainingData?.reward_trainable
          ? `Preference #${data.id} saved. Batch winner selected. Aura has ${trainingData.valid_comparison_count} valid comparisons and is updating its reward model.`
          : `Preference #${data.id} saved. Batch winner selected. Aura has ${trainingData?.valid_comparison_count ?? 'fewer than 6'} valid unique comparisons so far.`)
      }
      window.dispatchEvent(new CustomEvent('aura:preference-saved', { detail: { id: data.id, winner_id: preferredId } }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save preference.')
    } finally {
      setVoting(false)
    }
  }

  const skipMatchup = () => {
    if (!left || !right || voting || winnerId) return
    if (nextIndex < batch.length) {
      const next = batch[nextIndex]
      setRightId(next.id)
      setNextIndex(value => value + 1)
      setMessage(`Matchup skipped · no training label saved. Motion A now faces candidate ${nextIndex + 1} of ${batch.length}.`)
    } else {
      setWinnerId('__skipped__')
      setMessage('Final matchup skipped · no preference was saved for this pair.')
    }
  }

  const resetTournament = () => {
    if (batch.length < 2) return
    setLeftId(batch[0].id)
    setRightId(batch[1].id)
    setNextIndex(2)
    setWinnerId('')
    setMessage('Comparison reset. Previous saved preferences remain in Aura’s dataset.')
  }

  const voteStarter = (preferredIndex: number) => {
    const loserIndex = preferredIndex === starterLeftIndex ? starterRightIndex : starterLeftIndex
    const record = {
      winner_id: AIST_REFERENCE_MOTIONS[preferredIndex].id,
      loser_id: AIST_REFERENCE_MOTIONS[loserIndex].id,
      source: 'AIST++ starter comparison',
      context: 'Which retargeted G1 dance motion looks better?',
      created_at: new Date().toISOString(),
    }
    try {
      const existing = JSON.parse(localStorage.getItem(AIST_STARTER_STORAGE) || '[]') as unknown[]
      localStorage.setItem(AIST_STARTER_STORAGE, JSON.stringify([...existing, record]))
    } catch {
      localStorage.setItem(AIST_STARTER_STORAGE, JSON.stringify([record]))
    }
    if (starterNextIndex < AIST_REFERENCE_MOTIONS.length) {
      setStarterLeftIndex(preferredIndex)
      setStarterRightIndex(starterNextIndex)
      setStarterNextIndex(value => value + 1)
      setStarterProgress(0)
      setMessage(`Starter rating saved. The winner now faces ${AIST_REFERENCE_MOTIONS[starterNextIndex].label}.`)
    } else {
      setStarterWinnerIndex(preferredIndex)
      setMessage('AIST++ starter set complete. Generate an Aura batch when you are ready to create training preferences.')
    }
    window.dispatchEvent(new CustomEvent('aura:starter-preference-saved', { detail: record }))
  }

  const resetStarter = () => {
    setStarterLeftIndex(0)
    setStarterRightIndex(1)
    setStarterNextIndex(2)
    setStarterWinnerIndex(null)
    setStarterProgress(0)
    setMessage('Starter comparison reset. Previous starter ratings remain saved locally.')
  }

  const rewardFor = (id?: string) => id ? reward.scores.find(item => item.id === id) : undefined

  return (
    <section className="aura-generator-stage" aria-label="Aura G1 generation and comparison">
      <div className="aura-generator-toolbar">
        <div>
          <div className="card-kicker">01 / GENERATE &amp; COMPARE</div>
          <h2>One prompt. <em>Multiple possibilities.</em></h2>
          <p>Aura asks NVIDIA Kimodo for multiple G1 candidates, then puts them head-to-head. Your choice becomes real preference data for the Aura Motion Reward Model.</p>
        </div>
        <div className={`aura-engine-status ${status?.cuda_available ? 'ready' : ''}`}>
          <span />
          <strong>{generating ? 'AURA GENERATING' : status?.cuda_available ? 'NVIDIA KIMODO READY' : 'ENGINE OFFLINE'}</strong>
          <small>{status?.cuda_name || 'Waiting for local generator (CUDA or MPS)'}</small>
        </div>
      </div>

      <div className="aura-generator-grid">
        <aside className="aura-generator-controls">
          <label htmlFor="aura-motion-prompt">MOTION PROMPT</label>
          <textarea
            id="aura-motion-prompt"
            value={prompt}
            rows={5}
            disabled={generating}
            onChange={event => { setPrompt(event.target.value); setSelectedExample(null); setMessage('') }}
            onKeyDown={event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') void generate() }}
            placeholder="Describe the motion you want the Unitree G1 to perform…"
          />

          <div className="aura-candidate-slider">
            <div><label htmlFor="aura-candidate-count">CANDIDATES PER PROMPT</label><strong>{candidateCount}</strong></div>
            <input id="aura-candidate-count" type="range" min="2" max="6" step="1" value={candidateCount} disabled={generating} onChange={event => setCandidateCount(Number(event.target.value))} />
            <small>2 is fastest · more candidates create more human comparisons and take longer to generate.</small>
          </div>

          <button className={`aura-generate-button ${generating ? 'loading' : ''}`} disabled={generating || (!prompt.trim() && !selectedExample)} onClick={() => void generate()}>
            {generating ? <><span className="aura-button-spinner" aria-hidden="true" />Aura is generating…</> : `Generate ${candidateCount} motions →`}
          </button>
          <small className="aura-generator-shortcut">Ctrl/Cmd + Enter to generate · fresh random seeds per batch · {status?.default_diffusion_steps || 30} denoising steps per candidate</small>

          <div className="aura-example-block">
            <div className="aura-example-head"><span>EXAMPLES</span><small>click to load</small></div>
            <div className="aura-example-buttons">
              {shownExamples.map(example => (
                <button type="button" key={example.id} className={selectedExample === example.id ? 'selected' : ''} disabled={generating} onClick={() => chooseExample(example)} title={example.prompt}>
                  <strong>{example.label}</strong>
                  {example.has_constraints && <span>CONSTRAINED</span>}
                </button>
              ))}
            </div>
          </div>
          {message && <p className="aura-generator-message">{message}</p>}
          {error && <p className="aura-generator-error">{error}</p>}
        </aside>

        <div className="aura-generator-viewer">
          <div className="aura-viewer-head">
            <div><span>{generating ? 'GENERATING CANDIDATES' : batch.length >= 2 ? winnerId ? winnerId === '__skipped__' ? 'COMPARISON COMPLETE' : 'BATCH WINNER' : 'MOTION VS MOTION' : starterWinnerIndex !== null ? 'STARTER WINNER' : 'STARTER MOTION VS MOTION'}</span><strong>{batch.length >= 2 ? (winnerId ? winnerId === '__skipped__' ? 'No forced winner · skipped matchup' : batch.find(item => item.id === winnerId)?.name : `${batch.length} Aura candidates · tournament comparison`) : 'AIST++ preloaded motions · rate immediately'}</strong></div>
            {!generating && batch.length >= 2 && <small>{winnerId ? 'Preference saved' : `match ${Math.min(nextIndex - 1, batch.length - 1)} of ${batch.length - 1}`}</small>}
            {!generating && batch.length < 2 && <small>{starterWinnerIndex !== null ? 'Starter set complete' : `starter match ${Math.min(starterNextIndex - 1, AIST_REFERENCE_MOTIONS.length - 1)} of ${AIST_REFERENCE_MOTIONS.length - 1}`}</small>}
            {!generating && batch.length >= 2 && <small className="aura-prior-status">{reward.model ? `Aura reward model active · ${reward.model.total_comparison_count} comparisons · ${reward.model.motion_count} motions` : rewardTraining ? 'Training Aura reward model…' : 'Aura reward model learns after 6+ generated comparisons'}</small>}
          </div>

          {generating ? (
            <GeneratingDanceStage candidateCount={candidateCount} />
          ) : left?.preview_file && right?.preview_file ? (
            <div className="aura-head-to-head">
              <article className={winnerId === left.id ? 'aura-versus-card winner' : 'aura-versus-card'}>
                <div className="aura-versus-label"><span>MOTION A</span><small>{left.candidate_index ? `candidate ${left.candidate_index}${left.generation_seed !== undefined ? ` · seed ${left.generation_seed}` : ''}` : left.id}</small></div>
                <GeneratedG1RobotPreview key={`left-${left.id}`} file={`/aura-api/files/${encodeURIComponent(left.preview_file)}`} compact />
                <button disabled={voting || Boolean(winnerId)} onClick={() => void vote(left.id)}>{winnerId === left.id ? '✓ Batch winner' : 'Pick motion A'}</button>
              </article>
              <div className="aura-versus-mark" aria-hidden="true">VS</div>
              <article className={winnerId === right.id ? 'aura-versus-card winner' : 'aura-versus-card'}>
                <div className="aura-versus-label"><span>MOTION B</span><small>{right.candidate_index ? `candidate ${right.candidate_index}${right.generation_seed !== undefined ? ` · seed ${right.generation_seed}` : ''}` : right.id}</small></div>
                <GeneratedG1RobotPreview key={`right-${right.id}`} file={`/aura-api/files/${encodeURIComponent(right.preview_file)}`} compact />
                <button disabled={voting || Boolean(winnerId)} onClick={() => void vote(right.id)}>{winnerId === right.id ? '✓ Batch winner' : 'Pick motion B'}</button>
              </article>
              <button type="button" className="aura-skip-matchup" disabled={voting || Boolean(winnerId)} onClick={skipMatchup}>Neither / skip matchup</button>
            </div>
          ) : (
            <div className="aura-head-to-head aura-starter-head-to-head">
              <article className={starterWinnerIndex === starterLeftIndex ? 'aura-versus-card winner' : 'aura-versus-card'}>
                <div className="aura-versus-label"><span>MOTION A</span><small>{starterLeft.label} · AIST++</small></div>
                <div className="aura-starter-aist-player">
                  <XBotScene embodiment="g1" side="A" quality="reference" motionFile={starterLeft.file} motionUrl={starterLeft.url} degradationSeed={starterLeft.seed} startOffsetSeconds={starterLeft.startOffsetSeconds} paused={false} playhead={starterProgress} autoRotate showTrails showLandmarks={false} resetViewSignal={0} viewPose={starterLeftView} onViewPoseChange={setStarterLeftView} />
                </div>
                <button disabled={starterWinnerIndex !== null} onClick={() => voteStarter(starterLeftIndex)}>{starterWinnerIndex === starterLeftIndex ? '✓ Starter winner' : 'Pick motion A'}</button>
              </article>
              <div className="aura-versus-mark" aria-hidden="true">VS</div>
              <article className={starterWinnerIndex === starterRightIndex ? 'aura-versus-card winner' : 'aura-versus-card'}>
                <div className="aura-versus-label"><span>MOTION B</span><small>{starterRight.label} · AIST++</small></div>
                <div className="aura-starter-aist-player">
                  <XBotScene embodiment="g1" side="B" quality="reference" motionFile={starterRight.file} motionUrl={starterRight.url} degradationSeed={starterRight.seed} startOffsetSeconds={starterRight.startOffsetSeconds} paused={false} playhead={starterProgress} autoRotate showTrails showLandmarks={false} resetViewSignal={0} viewPose={starterRightView} onViewPoseChange={setStarterRightView} />
                </div>
                <button disabled={starterWinnerIndex !== null} onClick={() => voteStarter(starterRightIndex)}>{starterWinnerIndex === starterRightIndex ? '✓ Starter winner' : 'Pick motion B'}</button>
              </article>
              <div className="aura-starter-note">Preloaded AIST++ starter motions are for immediate onboarding. These ratings are stored locally and kept separate from Aura's same-prompt generated-motion training data.</div>
            </div>
          )}

          {!generating && batch.length >= 2 && <div className="aura-batch-strip">
            <span>BATCH CANDIDATES</span>
            <div>{batch.map((motion, index) => <div key={motion.id} className={motion.id === winnerId ? 'winner' : motion.id === leftId || motion.id === rightId ? 'active' : ''}><b>{String(index + 1).padStart(2, '0')}</b><span>{motion.id}{motion.generation_seed !== undefined ? ` · seed ${motion.generation_seed}` : ''}{motion.native_sha256 ? ` · ${motion.native_sha256.slice(0, 8)}…` : ''}</span></div>)}</div>
            {winnerId && winnerId !== '__skipped__' && reward.model && <section className="aura-reward-reveal" aria-label="Aura model ranking after human vote">
              <strong>AURA MODEL · POST-VOTE RANKING</strong>
              <small>Scores are revealed only after the human comparison to avoid biasing the label.</small>
              <div>{[...batch].sort((a, b) => (rewardFor(a.id)?.rank || 9999) - (rewardFor(b.id)?.rank || 9999)).map(motion => { const item = rewardFor(motion.id); return <span key={motion.id}><b>#{item?.rank ?? '–'}</b> {motion.candidate_index ? `Candidate ${motion.candidate_index}` : motion.id}<em>{item ? item.reward.toFixed(3) : 'unscored'}</em></span> })}</div>
            </section>}
            {winnerId && <button className="aura-reset-comparison" onClick={resetTournament}>Compare batch again</button>}
          </div>}
          {!generating && batch.length < 2 && <div className="aura-batch-strip aura-starter-strip">
            <span>PRELOADED AIST++</span>
            <div>{AIST_REFERENCE_MOTIONS.map((motion, index) => <div key={motion.id} className={index === starterWinnerIndex ? 'winner' : index === starterLeftIndex || index === starterRightIndex ? 'active' : ''}><b>{String(index + 1).padStart(2, '0')}</b><span>{motion.label}</span></div>)}</div>
            {starterWinnerIndex !== null && <button className="aura-reset-comparison" onClick={resetStarter}>Rate starter set again</button>}
          </div>}
        </div>
      </div>
    </section>
  )
}
