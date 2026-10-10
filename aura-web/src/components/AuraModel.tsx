import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CharacterPicker } from './CharacterPicker'
import { GeneratedG1RobotPreview } from './GeneratedG1RobotPreview'
import type { AuraModelVizLabel } from './AuraModelViz'
import { ExplorerModelGame, HeroModelGame } from './AuraModelGame'
import { useGamePool, useGameStats, type GamePoolItem } from './auraModelGameState'
import './aura-model.css'

type RewardMeta = {
  version?: string
  kind?: string
  training_paradigm?: string
  trained_at?: number
  training_count?: number
  total_comparison_count?: number
  stored_comparison_count?: number
  unique_pair_count?: number
  duplicate_comparison_count?: number
  rejected_comparison_count?: number
  motion_count?: number
  parameter_count?: number
  sequence_length?: number
  input_dim?: number
  architecture?: { d_model?: number; heads?: number; layers?: number; feedforward?: number }
  train_pair_accuracy?: number | null
  train_pair_log_loss?: number | null
  heldout_pair_accuracy?: number | null
  heldout_pair_log_loss?: number | null
  heldout_pair_count?: number
  dataset_sha256?: string
  model_sha256?: string
  feature_method?: string
  caveat?: string
}

type RewardScore = {
  id: string
  name?: string
  reward: number
  rank: number
  rank_percentile?: number
}

type RewardState = {
  model: RewardMeta | null
  scores: RewardScore[]
  error?: string
}

type TrainingDiagnostics = {
  stored_comparison_count: number
  valid_comparison_count: number
  unique_pair_count: number
  unique_motion_count: number
  duplicate_comparison_count: number
  rejected_comparison_count: number
  rejection_reasons?: Record<string, number>
  reward_min_comparisons: number
  reward_min_motions: number
  reward_trainable: boolean
}

type Motion = {
  id: string
  name: string
  model: string
  created_at: string
  frames: number
  fps: number
  native_file: string
  preview_file: string | null
  preview_error?: string | null
}

type LayerId = 'trajectory' | 'features' | 'projection' | 'position' | 'transformer' | 'pool' | 'head' | 'reward'

type Layer = {
  id: LayerId
  eyebrow: string
  title: string
  shape: string
  description: string
  detail: string
  learnable: boolean
}

const compact = (n?: number | null) => {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat('en', { notation: n >= 10_000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(n)
}

const pct = (n?: number | null) => n == null ? 'Withheld' : `${(n * 100).toFixed(1)}%`
const shortHash = (value?: string) => value ? `${value.slice(0, 10)}…${value.slice(-8)}` : '—'

export function AuraModel() {
  const [reward, setReward] = useState<RewardState>({ model: null, scores: [] })
  const [loading, setLoading] = useState(true)
  const [diagnostics, setDiagnostics] = useState<TrainingDiagnostics | null>(null)
  const [motions, setMotions] = useState<Motion[]>([])
  const [selectedMotionId, setSelectedMotionId] = useState('')
  const [training, setTraining] = useState(false)
  const [message, setMessage] = useState('')
  const [active, setActive] = useState<LayerId>('transformer')
  const [playing, setPlaying] = useState(false)
  const timer = useRef<number | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [rewardResponse, diagnosticsResponse, motionsResponse] = await Promise.all([
        fetch('/aura-api/reward', { cache: 'no-store' }),
        fetch('/aura-api/preferences/diagnostics', { cache: 'no-store' }),
        fetch('/aura-api/motions', { cache: 'no-store' }),
      ])
      if (!rewardResponse.ok) throw new Error(`Aura API returned ${rewardResponse.status}`)
      const data = await rewardResponse.json() as RewardState
      const scores = Array.isArray(data.scores) ? data.scores : []
      setReward({ model: data.model ?? null, scores, error: data.error })
      if (diagnosticsResponse.ok) setDiagnostics(await diagnosticsResponse.json() as TrainingDiagnostics)
      if (motionsResponse.ok) {
        const motionData = await motionsResponse.json() as Motion[]
        const library = Array.isArray(motionData) ? motionData : []
        setMotions(library)
        const previewable = library.filter(item => item.preview_file?.endsWith('.g1.json'))
        setSelectedMotionId(previous => {
          if (previous && previewable.some(item => item.id === previous)) return previous
          const topScored = scores.find(item => previewable.some(motion => motion.id === item.id))?.id
          return topScored ?? previewable[0]?.id ?? ''
        })
      } else {
        setMotions([])
        setSelectedMotionId('')
      }
    } catch (error) {
      setReward({ model: null, scores: [], error: error instanceof Error ? error.message : 'Aura API unavailable' })
      setDiagnostics(null)
      setMotions([])
      setSelectedMotionId('')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    const id = window.setInterval(() => { if (!training) void refresh() }, 15000)
    return () => window.clearInterval(id)
  }, [refresh, training])
  useEffect(() => () => { if (timer.current != null) window.clearTimeout(timer.current) }, [])

  const meta = reward.model
  const dimSeq = meta?.sequence_length ?? 64
  const dimInput = meta?.input_dim ?? 413
  const dimModel = meta?.architecture?.d_model ?? 96
  const dimHeads = meta?.architecture?.heads ?? 4
  const dimBlocks = meta?.architecture?.layers ?? 2
  const dimFf = meta?.architecture?.feedforward ?? 192
  const dims = useMemo(() => ({ seq: dimSeq, input: dimInput, model: dimModel, heads: dimHeads, blocks: dimBlocks, ff: dimFf }), [dimSeq, dimInput, dimModel, dimHeads, dimBlocks, dimFf])

  const layers = useMemo<Layer[]>(() => [
    {
      id: 'trajectory', eyebrow: '01 · Robot trajectory', title: 'G1 motion', shape: 'T × 34 × 3', learnable: false,
      description: 'The model starts from the saved robot-native Unitree G1 trajectory, not from Aura’s handcrafted quality metrics.',
      detail: 'Each motion contains 34 world-space G1 joint positions over time, with global joint rotations when the generated file provides them.'
    },
    {
      id: 'features', eyebrow: '02 · Motion representation', title: 'Trajectory features', shape: `${dims.seq} × ${dims.input}`, learnable: false,
      description: 'Aura converts every trajectory into a fixed-length sequence of pose and dynamics features.',
      detail: 'Per frame: root-relative joint positions, joint velocities, root velocity, root height, 6D global rotations, and a rotation-presence bit. The sequence is resampled to 64 frames and normalized from the training set.'
    },
    {
      id: 'projection', eyebrow: '03 · Learned embedding', title: 'Input projection', shape: `${dims.input} → ${dims.model}`, learnable: true,
      description: 'A learned linear projection compresses each high-dimensional motion frame into Aura’s latent motion space.',
      detail: `Linear(${dims.input}, ${dims.model}) → LayerNorm → GELU. This is the first learned layer and maps heterogeneous trajectory channels into a shared ${dims.model}-dimensional representation.`
    },
    {
      id: 'position', eyebrow: '04 · Time identity', title: 'Positional embedding', shape: `${dims.seq} × ${dims.model}`, learnable: true,
      description: 'Learned positional embeddings tell Aura where each frame occurs in the motion.',
      detail: 'The same pose can mean something different at take-off, mid-motion, or landing. A learned temporal embedding is added before attention.'
    },
    {
      id: 'transformer', eyebrow: '05 · Temporal reasoning', title: `Transformer ×${dims.blocks}`, shape: `${dims.heads} heads · FF ${dims.ff}`, learnable: true,
      description: 'Self-attention lets every frame compare itself with the rest of the trajectory, learning whole-motion coordination and timing.',
      detail: `${dims.blocks} Transformer encoder blocks use ${dims.heads}-head self-attention with a ${dims.ff}-unit feed-forward network. This is where Aura can learn temporal patterns that simple foot-slide or displacement metrics cannot express.`
    },
    {
      id: 'pool', eyebrow: '06 · Motion summary', title: 'Temporal pooling', shape: `${dims.seq} × ${dims.model} → ${dims.model}`, learnable: false,
      description: 'Aura averages the encoded timeline into one representation for the complete motion.',
      detail: 'Mean pooling preserves a compact fixed-size motion embedding while keeping the reward model intentionally small enough for limited preference data.'
    },
    {
      id: 'head', eyebrow: '07 · Preference head', title: 'Reward MLP', shape: `${dims.model} → 64 → 1`, learnable: true,
      description: 'A small neural head converts the motion embedding into one unconstrained scalar reward.',
      detail: `Linear(${dims.model}, 64) → GELU → Dropout → Linear(64, 1). The scalar is meaningful comparatively: higher means the learned model currently ranks that motion above lower-scored motions.`
    },
    {
      id: 'reward', eyebrow: '08 · Model output', title: 'Aura reward', shape: 'r(motion)', learnable: false,
      description: 'The final scalar is Aura’s learned human-preference reward for a G1 trajectory.',
      detail: 'That score now drives the live 3D output browser below: you can inspect the actual saved motion, then retarget the same trajectory onto the Unitree G1 or the fairy body.'
    },
  ], [dims.blocks, dims.ff, dims.heads, dims.input, dims.model, dims.seq])

  const selected = layers.find(layer => layer.id === active) ?? layers[0]
  const rewardById = useMemo(() => new Map(reward.scores.map(item => [item.id, item])), [reward.scores])
  const motionById = useMemo(() => new Map(motions.map(item => [item.id, item])), [motions])
  const previewableMotions = useMemo(() => motions.filter(item => item.preview_file?.endsWith('.g1.json')), [motions])
  const rankedPreviewable = useMemo(() => {
    const ranked = reward.scores.filter(item => motionById.get(item.id)?.preview_file?.endsWith('.g1.json'))
    if (ranked.length) return ranked
    return previewableMotions.map((motion, index) => ({
      id: motion.id,
      name: motion.name,
      reward: 0,
      rank: index + 1,
      rank_percentile: previewableMotions.length <= 1 ? 1 : 1 - index / (previewableMotions.length - 1),
    }))
  }, [motionById, previewableMotions, reward.scores])

  const selectedMotion = motionById.get(selectedMotionId) ?? previewableMotions[0] ?? null
  const selectedScore = selectedMotion ? rewardById.get(selectedMotion.id) ?? null : null

  // Feed-the-model game: real Aura scores when a trained model has scored previewable motions, else demo samples.
  const game = useGameStats()
  const realPool = useMemo<GamePoolItem[]>(() => {
    if (!meta) return []
    const n = reward.scores.length
    return reward.scores.flatMap(item => {
      const file = motionById.get(item.id)?.preview_file
      if (!file?.endsWith('.g1.json')) return []
      const norm = item.rank_percentile ?? (n <= 1 ? 1 : 1 - (item.rank - 1) / (n - 1))
      return [{ id: item.id, name: item.name || motionById.get(item.id)?.name || item.id, file: `/aura-api/files/${encodeURIComponent(file)}`, score: item.reward, norm, demo: false }]
    })
  }, [meta, reward.scores, motionById])
  const gamePool = useGamePool(realPool)
  const [celebrate, setCelebrate] = useState<{ key: number; id: LayerId; title: string } | null>(null)
  const selectLayer = (id: LayerId) => {
    setActive(id)
    if (game.unlock(id)) setCelebrate(prev => ({ key: (prev?.key ?? 0) + 1, id, title: layers.find(l => l.id === id)?.title ?? id }))
  }
  const vizLabels = useMemo<AuraModelVizLabel[]>(() => layers.map(l => ({ id: l.id, number: l.eyebrow.split(' · ')[0], title: l.title, learnable: l.learnable, shape: l.shape, blurb: l.description })), [layers])
  const vizDescription = `3D diagram of the Aura reward model: a ${dims.seq}-frame G1 motion becomes ${dims.seq} × ${dims.input} trajectory features, a learned projection to ${dims.model} dimensions, a learned positional embedding, ${dims.blocks} Transformer block${dims.blocks === 1 ? '' : 's'} with ${dims.heads} attention heads, temporal mean pooling, a ${dims.model} → 64 → 1 reward MLP, and one scalar reward.`

  // Subtle card transition when the selected layer changes (skipped under reduced motion).
  const detailRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const el = detailRef.current
    if (!el || typeof el.animate !== 'function' || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    // starts with the 3D camera glide (~1.05 s) and lands as it settles
    const anim = el.animate([{ opacity: 0.4, transform: 'translateY(10px)' }, { opacity: 1, transform: 'none' }], { duration: 600, delay: 320, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'backwards' })
    return () => anim.cancel()
  }, [active])

  const runForward = () => {
    if (timer.current != null) window.clearTimeout(timer.current)
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduced) { setActive('reward'); return }
    setPlaying(true)
    let index = 0
    const step = () => {
      setActive(layers[index].id)
      index += 1
      if (index >= layers.length) {
        timer.current = window.setTimeout(() => setPlaying(false), 550)
        return
      }
      timer.current = window.setTimeout(step, 1000) // time for the 3D camera to zoom onto each stage
    }
    step()
  }

  const train = async () => {
    if (training) return
    setTraining(true)
    setMessage('Training Aura on the saved human comparisons…')
    try {
      const response = await fetch('/aura-api/reward/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: 40 })
      })
      const data = await response.json().catch(() => ({})) as { error?: string }
      if (!response.ok) throw new Error(data.error || `Training failed (${response.status})`)
      setMessage('Aura model updated from the current preference dataset.')
      await refresh()
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Unable to train Aura.')
    } finally {
      setTraining(false)
    }
  }

  const trainedAt = meta?.trained_at ? new Date(meta.trained_at * 1000).toLocaleString() : 'Not trained yet'

  return (
    <main className="fx-page aura-model-page">
      <section className="aura-model-hero">
        <HeroModelGame dims={dims} labels={vizLabels} description={vizDescription} pool={gamePool} api={game}>
          <span className="aura-model-kicker">Human-preference reward model</span>
          <h1 className="fx-h1">Look inside<br />the Aura model.</h1>
          <p>A compact temporal Transformer learns a scalar reward directly from human A/B choices over G1 motion. Click through the layers, then inspect the live model trained on this machine.</p>
        </HeroModelGame>
      </section>

      <section className="aura-model-section aura-model-output" aria-labelledby="aura-model-output-title">
        <div className="aura-model-section-head aura-model-output-head">
          <div>
            <span className="aura-model-kicker">Live 3D output</span>
            <h2 id="aura-model-output-title">Inspect Aura’s actual motion output.</h2>
          </div>
          <div className="aura-model-output-tools">
            <div className="aura-model-train-cta">
              <button
                type="button"
                className="aura-model-train-button"
                onClick={() => void train()}
                disabled={training || Boolean(diagnostics && !diagnostics.reward_trainable)}
                title={diagnostics && !diagnostics.reward_trainable ? `Needs ${diagnostics.reward_min_comparisons} valid comparisons across ${diagnostics.reward_min_motions} motions` : undefined}
              >
                {training ? 'Training Aura…' : meta ? 'Retrain Aura model' : 'Train Aura model'}
              </button>
              <small>
                {training
                  ? 'Updating the reward model from saved A/B preferences.'
                  : diagnostics?.reward_trainable
                    ? `${diagnostics.valid_comparison_count} valid comparisons ready.`
                    : diagnostics
                      ? `Need ${diagnostics.reward_min_comparisons} valid comparisons across ${diagnostics.reward_min_motions} motions.`
                      : 'Checking training data…'}
              </small>
            </div>
            <CharacterPicker className="aura-model-character-picker" />
          </div>
        </div>
        {message && <p className="aura-model-top-message" role="status">{message}</p>}

        <div className="aura-model-output-shell">
          <div className="aura-model-output-player">
            {selectedMotion?.preview_file ? (
              <>
                <div className="aura-model-output-meta">
                  <div>
                    <span>Selected motion</span>
                    <strong>{selectedMotion.name}</strong>
                  </div>
                  <div>
                    <span>Current Aura score</span>
                    <strong>{selectedScore ? `${selectedScore.reward.toFixed(3)} · #${selectedScore.rank}` : 'Unscored'}</strong>
                  </div>
                </div>
                <GeneratedG1RobotPreview key={`${selectedMotion.id}:${selectedMotion.preview_file}`} file={`/aura-api/files/${encodeURIComponent(selectedMotion.preview_file)}`} />
              </>
            ) : (
              <div className="aura-model-output-empty">
                <strong>No previewable G1 motion yet.</strong>
                <p>Generate or save at least one G1 motion, then Aura can show its output here as a retargeted 3D character instead of only a score table.</p>
              </div>
            )}
          </div>

          <aside className="aura-model-output-sidebar">
            <div className="aura-model-output-copy">
              <strong>What you are seeing</strong>
              <p>This panel renders the same saved G1 trajectory that Aura scores. Switch the character to view it as the native Unitree G1 body or as the retargeted fairy model.</p>
              <small>The model output is not just a number anymore—you can inspect the actual motion bytes Aura is ranking.</small>
            </div>

            {rankedPreviewable.length ? <div className="aura-model-output-list" role="list" aria-label="Previewable saved motions">
              {rankedPreviewable.slice(0, 10).map((item) => {
                const motion = motionById.get(item.id)
                if (!motion) return null
                const isActive = motion.id === selectedMotion?.id
                return (
                  <button
                    key={motion.id}
                    type="button"
                    role="listitem"
                    className={`aura-model-output-item ${isActive ? 'is-active' : ''}`}
                    onClick={() => setSelectedMotionId(motion.id)}
                    aria-pressed={isActive}
                  >
                    <div>
                      <span>{rewardById.has(motion.id) ? `Aura rank #${item.rank}` : 'Saved motion'}</span>
                      <strong>{motion.name}</strong>
                      <small>{motion.frames} frames · {motion.fps} fps · {new Date(motion.created_at).toLocaleString()}</small>
                    </div>
                    <b>{rewardById.has(motion.id) ? item.reward.toFixed(3) : '—'}</b>
                  </button>
                )
              })}
            </div> : <div className="aura-model-output-empty is-compact">
              <strong>No ranked motion outputs yet.</strong>
              <p>Train Aura and save at least one previewable G1 motion to turn this into an interactive 3D output browser.</p>
            </div>}
          </aside>
        </div>
      </section>


      <section className="aura-model-shell" aria-labelledby="aura-model-live-title">
        <div className="aura-model-statusbar">
          <div>
            <span className={meta ? 'aura-model-dot is-active' : 'aura-model-dot'} aria-hidden="true" />
            <div><strong id="aura-model-live-title">{meta ? 'Aura model active' : loading ? 'Checking Aura model…' : 'Aura model untrained'}</strong><small>{meta?.version ?? 'aura-motion-reward-transformer-v1'}</small></div>
          </div>
          <div className="aura-model-actions">
            <button type="button" onClick={() => void refresh()} disabled={loading || training}>Refresh</button>
            <button type="button" className="is-primary" onClick={() => void train()} disabled={training || Boolean(diagnostics && !diagnostics.reward_trainable)} title={diagnostics && !diagnostics.reward_trainable ? `Needs ${diagnostics.reward_min_comparisons} valid comparisons across ${diagnostics.reward_min_motions} motions` : undefined}>{training ? 'Training…' : meta ? 'Update model' : 'Train model'}</button>
          </div>
        </div>
        {message && <p className="aura-model-message" role="status">{message}</p>}
        {reward.error && !meta && <p className="aura-model-message is-muted">Live API unavailable: {reward.error}. The architecture below remains explorable.</p>}

        <div className="aura-model-metrics">
          <div><span>Valid comparisons</span><strong>{compact(diagnostics?.valid_comparison_count ?? meta?.total_comparison_count)}</strong><small>{diagnostics ? `${diagnostics.reward_min_comparisons} required to train` : 'Checking training data…'}</small></div>
          <div><span>Unique motions</span><strong>{compact(diagnostics?.unique_motion_count ?? meta?.motion_count)}</strong><small>{diagnostics ? `${diagnostics.reward_min_motions} required · hash-deduplicated` : 'Generated G1 trajectories'}</small></div>
          <div><span>Duplicate votes</span><strong>{compact(diagnostics?.duplicate_comparison_count ?? meta?.duplicate_comparison_count)}</strong><small>{diagnostics?.rejected_comparison_count ? `${diagnostics.rejected_comparison_count} additional rejected` : 'Duplicates do not train Aura'}</small></div>
          <div><span>Held-out accuracy</span><strong>{pct(meta?.heldout_pair_accuracy)}</strong><small>{meta?.heldout_pair_count ? `${meta.heldout_pair_count} held-out comparisons` : 'Shown once enough unique pair groups exist'}</small></div>
        </div>
        {diagnostics && <div className={`aura-model-data-health ${diagnostics.reward_trainable ? 'is-ready' : ''}`}>
          <strong>{diagnostics.reward_trainable ? 'Training data ready' : 'Training data not ready yet'}</strong>
          <span>{diagnostics.stored_comparison_count} stored · {diagnostics.valid_comparison_count} valid · {diagnostics.unique_pair_count} unique pairs · {diagnostics.unique_motion_count} unique motions</span>
          {(diagnostics.duplicate_comparison_count > 0 || diagnostics.rejected_comparison_count > 0) && <small>{diagnostics.duplicate_comparison_count} duplicate vote(s) and {diagnostics.rejected_comparison_count} rejected comparison(s) are excluded from training.</small>}
        </div>}
      </section>

      <section className="aura-model-section" aria-labelledby="aura-model-architecture-title">
        <div className="aura-model-section-head">
          <div><span className="aura-model-kicker">Interactive architecture</span><h2 id="aura-model-architecture-title">Follow one motion through Aura.</h2></div>
          <button type="button" className="aura-model-run" onClick={runForward} disabled={playing}>{playing ? 'Forward pass…' : 'Animate forward pass'}</button>
        </div>

        <div className="aura-model-explorer">
          <div className="aura-model-pipeline">
            <ExplorerModelGame dims={dims} labels={vizLabels} description={vizDescription} pool={gamePool} api={game}
              active={active} playing={playing} onSelect={selectLayer} onStage={setActive} celebrate={celebrate}>
            <div className="aura-model-layer-chips" role="list" aria-label="Aura reward model layers">
              {layers.map(layer => (
                <button
                  key={layer.id}
                  type="button"
                  role="listitem"
                  className={`aura-model-layer-chip ${active === layer.id ? 'is-active' : ''} ${layer.learnable ? 'is-learned' : 'is-operation'}`}
                  onClick={() => selectLayer(layer.id)}
                  aria-pressed={active === layer.id}
                  title={`${layer.eyebrow} · ${layer.shape}`}
                >
                  <span>{layer.eyebrow.split(' · ')[0]}</span>{layer.title}
                  {game.stats.codex.includes(layer.id) && <i className="aura-model-chip-seen" aria-label="unlocked">✦</i>}
                </button>
              ))}
            </div>
            </ExplorerModelGame>
          </div>

          <aside className="aura-model-layer-detail" aria-live="polite" ref={detailRef}>
            <div className="aura-model-detail-number">{selected.eyebrow.split(' · ')[0]}</div>
            <span>{selected.learnable ? 'Learned layer' : 'Deterministic operation'}</span>
            <h3>{selected.title}</h3>
            <code>{selected.shape}</code>
            <p>{selected.description}</p>
            <p className="is-secondary">{selected.detail}</p>
            <div className="aura-model-detail-foot"><span>{selected.learnable ? 'Weights update from human preferences' : 'No trainable parameters here'}</span></div>
          </aside>
        </div>
      </section>

      <section className="aura-model-training" aria-labelledby="aura-model-training-title">
        <div className="aura-model-training-copy">
          <span className="aura-model-kicker">How human feedback trains it</span>
          <h2 id="aura-model-training-title">One network. Two motions. One choice.</h2>
          <p>Aura runs Motion A and Motion B through the same reward network. Human preference teaches the network which scalar should be higher.</p>
          <div className="aura-model-equation"><span>P(A &gt; B)</span><b>=</b><strong>σ( r(A) − r(B) )</strong></div>
          <small>This is the reward-modeling component used in RLHF systems. Aura does not claim full RLHF because it does not yet optimize a generator or robot policy with reinforcement learning against this reward.</small>
        </div>
        <div className="aura-model-pair" aria-label="Pairwise preference training diagram">
          <div><span>Motion A</span><b>shared Aura model</b><strong>r(A)</strong></div>
          <div className="aura-model-pair-vs">Human chooses</div>
          <div><span>Motion B</span><b>shared Aura model</b><strong>r(B)</strong></div>
          <div className="aura-model-pair-loss"><span>Bradley–Terry preference loss</span><strong>push the chosen motion's reward higher</strong></div>
        </div>
      </section>

      <section className="aura-model-section aura-model-live-data" aria-labelledby="aura-model-data-title">
        <div className="aura-model-section-head">
          <div><span className="aura-model-kicker">Live model state</span><h2 id="aura-model-data-title">What Aura has learned so far.</h2></div>
          <small className="aura-model-trained-at">Last trained<br /><strong>{trainedAt}</strong></small>
        </div>

        {meta ? <>
          <div className="aura-model-validation">
            <div><span>Training pair accuracy</span><strong>{pct(meta.train_pair_accuracy)}</strong><small>Log loss {meta.train_pair_log_loss ?? '—'}</small></div>
            <div><span>Held-out pair accuracy</span><strong>{pct(meta.heldout_pair_accuracy)}</strong><small>Log loss {meta.heldout_pair_log_loss ?? '—'}</small></div>
            <div><span>Sequence length</span><strong>{dims.seq}</strong><small>frames per motion</small></div>
            <div><span>Latent width</span><strong>{dims.model}</strong><small>dimensions</small></div>
          </div>

          <div className="aura-model-ranking-head"><h3>Current motion ranking</h3><span>Raw reward is comparative, not a calibrated probability. Click a row to open that motion in the 3D viewer above.</span></div>
          {reward.scores.length ? <div className="aura-model-ranking">
            {reward.scores.slice(0, 12).map((score) => {
              const maxAbs = Math.max(1, ...reward.scores.map(item => Math.abs(item.reward)))
              const normalized = Math.max(6, Math.min(100, 50 + (score.reward / maxAbs) * 46))
              const canPreview = motionById.get(score.id)?.preview_file?.endsWith('.g1.json')
              return <button type="button" className={`aura-model-rank-row ${canPreview ? 'is-clickable' : ''}`} key={score.id} onClick={() => { if (canPreview) setSelectedMotionId(score.id) }} disabled={!canPreview}>
                <b>#{score.rank}</b><div><span>{score.name || score.id}</span><i><em style={{ width: `${normalized}%` }} /></i></div><strong>{score.reward.toFixed(3)}</strong>
              </button>
            })}
          </div> : <p className="aura-model-empty">The model is trained, but there are no currently scorable library motions.</p>}

          <div className="aura-model-provenance">
            <div><span>Dataset SHA-256</span><code title={meta.dataset_sha256}>{shortHash(meta.dataset_sha256)}</code></div>
            <div><span>Model SHA-256</span><code title={meta.model_sha256}>{shortHash(meta.model_sha256)}</code></div>
          </div>
        </> : <div className="aura-model-untrained">
          <strong>No trained checkpoint yet.</strong>
          <p>{diagnostics ? `Aura currently has ${diagnostics.valid_comparison_count} valid unique comparison${diagnostics.valid_comparison_count === 1 ? '' : 's'} across ${diagnostics.unique_motion_count} unique motion${diagnostics.unique_motion_count === 1 ? '' : 's'}. It needs ${diagnostics.reward_min_comparisons} valid comparisons across at least ${diagnostics.reward_min_motions} unique motions.` : 'Collect at least six generated-motion A/B comparisons covering four distinct motions. Then Aura can train its first reward model.'}</p>
        </div>}
      </section>

      <section className="aura-model-caveat">
        <strong>What the score means</strong>
        <p>{meta?.caveat ?? 'Aura learns a human-preference reward over saved G1 trajectories. It is not physical safety validation, simulator task success, or full RLHF because no generator or robot policy is optimized with reinforcement learning against this reward.'}</p>
      </section>
    </main>
  )
}
