import { useEffect, useRef, useState, type ReactNode } from 'react'
import { prefersReducedMotion } from '../motionPrefs'
import { AURA_MODEL_LAYER_IDS, type AuraModelDims, type AuraModelLayerId } from '../three/auraModelScene'
import { AuraModelViz, type AuraModelVizLabel } from './AuraModelViz'
import { useFeedRound, useJoints, type FeedRound, type GamePoolItem, type GameStatsApi } from './auraModelGameState'

const fmt = (n: number) => (n >= 0 ? '+' : '') + n.toFixed(3)
const TOTAL = AURA_MODEL_LAYER_IDS.length

/** Counts up to `value` (instantly under reduced motion). */
function TickNumber({ value, run }: { value: number; run: number }) {
  const [shown, setShown] = useState(value)
  useEffect(() => {
    if (prefersReducedMotion()) { setShown(value); return }
    let raf = 0
    const t0 = performance.now(), from = 0
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / 750), e = 1 - (1 - k) ** 3
      setShown(from + (value - from) * e)
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value, run])
  return <>{fmt(shown)}</>
}

function ResultPop({ round }: { round: FeedRound }) {
  if (round.phase !== 'result' || !round.current) return null
  const o = round.outcome
  const title = o === 'win' ? 'Nailed it!' : o === 'miss' ? 'Poof!' : o === 'tie' ? 'Dead heat' : o === 'warmup' ? 'The bar is set' : 'Scored'
  return (
    <div className={`aura-model-game-pop is-${o}`} aria-hidden="true">
      <span>{title}</span>
      <strong><TickNumber value={round.current.score} run={round.popKey} /></strong>
      {round.outcome === 'win' && round.streak > 1 && <em>streak ×{round.streak}</em>}
    </div>
  )
}

type VizProps = { dims: AuraModelDims; labels: AuraModelVizLabel[]; description: string; pool: GamePoolItem[]; api: GameStatsApi }

/**
 * Hero: full-screen world with the title on top. It stays on the overview with ambient life only (particle
 * flow, sparkles, fairy props) until the user acts: the stepper or a click on a stage glides the camera there.
 * No autoplay and no game loop here; the game lives in the explorer.
 */
export function HeroModelGame({ dims, labels, description, pool, children }: VizProps & { children?: ReactNode }) {
  const [focus, setFocus] = useState<AuraModelLayerId | null>(null)
  const motion = useJoints(pool[0]?.file ?? null)
  const stageRef = useRef<HTMLDivElement>(null)
  const copyRef = useRef<HTMLDivElement>(null)
  const navRef = useRef<HTMLDivElement>(null)
  const [band, setBand] = useState({ copy: 0.42, nav: 0.14, h: 900 })
  useEffect(() => {
    const st = stageRef.current, copy = copyRef.current, nav = navRef.current
    if (!st || !copy || !nav || typeof ResizeObserver === 'undefined') return
    const measure = () => {
      const r = st.getBoundingClientRect(), c = copy.getBoundingClientRect(), n = nav.getBoundingClientRect()
      if (r.height < 10) return
      const fCopy = Math.round(((c.bottom - r.top + 18) / r.height) * 100) / 100
      const fNav = Math.round(((r.bottom - n.top + 10) / r.height) * 100) / 100
      setBand(p => (p.copy === fCopy && p.nav === fNav && p.h === r.height ? p : { copy: fCopy, nav: fNav, h: r.height }))
    }
    const ro = new ResizeObserver(measure)
    ro.observe(st); ro.observe(copy); ro.observe(nav)
    measure()
    return () => ro.disconnect()
  }, [])
  // While a motion flies (and during its reveal) the hero camera moves exactly like the explorer's:
  // stage by stage, then the reward. Focused or flying, the title fades out and the 3D gets the whole height.
  const view: AuraModelLayerId | null = focus
  const safe = view ? { top: Math.min(0.3, Math.round((118 / band.h) * 100) / 100), bottom: band.nav } : { top: band.copy, bottom: band.nav }
  const idx = focus ? AURA_MODEL_LAYER_IDS.indexOf(focus) : -1
  const step = (d: number) => setFocus(AURA_MODEL_LAYER_IDS[(Math.max(idx, d > 0 ? -1 : 0) + d + TOTAL) % TOTAL])
  const focused = labels.find(l => l.id === focus)
  return (
    <div className={`aura-model-hero-stage ${view ? 'is-focused' : ''}`} ref={stageRef}>
      <AuraModelViz mode="hero" className="aura-model-hero-viz" dims={dims} labels={labels} description={description}
        active={focus ?? 'reward'} playing={false} motion={motion} reward={null}
        view={view} safeArea={safe} onSelect={id => setFocus(id)}>
      </AuraModelViz>
      <div className="aura-model-hero-copy" ref={copyRef}>{children}</div>
      <div className="aura-model-hero-nav" ref={navRef}>
        {focused && <div key={focused.id} className="aura-model-hero-card" aria-live="polite">
          <span>{focused.number} · {focused.learnable ? 'Learned layer' : 'Deterministic operation'}</span>
          <strong>{focused.title}</strong>{focused.shape && <code>{focused.shape}</code>}
          {focused.blurb && <p>{focused.blurb}</p>}
        </div>}
        <div className="aura-model-hero-stepper" role="group" aria-label="Step through the model">
          <button type="button" onClick={() => step(-1)} aria-label="Previous stage">‹</button>
          <span>{focused ? `${focused.number} / 0${TOTAL}` : 'Walk through the model'}</span>
          <button type="button" onClick={() => step(1)} aria-label="Next stage">›</button>
          {focus && <button type="button" className="is-text" onClick={() => setFocus(null)}>Overview</button>}
        </div>
      </div>
    </div>
  )
}

type ExplorerProps = VizProps & {
  active: AuraModelLayerId
  playing: boolean
  onSelect: (id: AuraModelLayerId) => void
  onStage: (id: AuraModelLayerId) => void
  celebrate: { key: number; id: AuraModelLayerId; title: string } | null
  children?: ReactNode
}

/** Explorer: the full game (predict, launch or drag-to-launch, reveal, codex), plus the page's chips as children. */
export function ExplorerModelGame({ dims, labels, description, pool, api, active, playing, onSelect, onStage, celebrate, children }: ExplorerProps) {
  const round = useFeedRound(pool, api, { start: 1, onStage })
  const motion = useJoints(round.current?.file ?? null)
  const vizRef = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<{ x: number; y: number; over: boolean; moved: boolean } | null>(null)
  const dragStart = useRef({ x: 0, y: 0 })
  const flying = round.phase === 'flying'
  const stageTitle = labels.find(l => l.id === round.stage)?.title

  const overViz = (x: number, y: number) => {
    const r = vizRef.current?.getBoundingClientRect()
    return Boolean(r && x >= r.left && x <= r.right && y >= r.top && y <= r.bottom)
  }
  const onDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (round.phase !== 'ready' || e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    dragStart.current = { x: e.clientX, y: e.clientY }
    setDrag({ x: e.clientX, y: e.clientY, over: false, moved: false })
  }
  const onMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!drag) return
    const moved = drag.moved || Math.hypot(e.clientX - dragStart.current.x, e.clientY - dragStart.current.y) > 6
    setDrag({ x: e.clientX, y: e.clientY, over: overViz(e.clientX, e.clientY), moved })
  }
  const onUp = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!drag) return
    const click = !drag.moved, dropped = drag.moved && overViz(e.clientX, e.clientY)
    setDrag(null)
    if (click || dropped) round.launch() // a plain press launches too; keyboard presses go through onClick
  }

  const question = round.phase === 'ready'
    ? round.last ? <>Will Aura score it <b>higher</b> or <b>lower</b> than <code>{fmt(round.last.score)}</code>?</> : <>Warm-up: launch a motion to set the bar.</>
    : flying ? <>The fairy carries it through <b>{stageTitle ?? 'the model'}</b>…</>
    : round.outcome === 'win' ? <>Correct! Streak <b>×{api.stats.streak}</b>.</>
    : round.outcome === 'miss' ? <>Missed it. Streak reset, the fairy shrugs.</>
    : <>Scored <code>{fmt(round.current?.score ?? 0)}</code>. Now predict the next one.</>

  return (
    <>
      <div ref={vizRef} className={`aura-model-game-drop ${drag?.over ? 'is-over' : ''}`}>
        <AuraModelViz mode="explorer" className="aura-model-explorer-viz" dims={dims} labels={labels} description={description}
          active={flying && round.stage ? round.stage : active} playing={flying || playing} motion={motion}
          reward={round.phase === 'result' && round.current ? round.current.norm : null}
          onSelect={id => { if (!flying && !playing) onSelect(id) }}
          pop={round.popKey ? { key: round.popKey, kind: round.outcome === 'win' ? 'win' : round.outcome === 'miss' ? 'miss' : 'neutral' } : null}
          celebrate={celebrate}>
          <ResultPop round={round} />
          {celebrate && <div key={celebrate.key} className="aura-model-game-toast" aria-hidden="true">Codex unlocked · {celebrate.title} <b>{api.stats.codex.length}/{TOTAL}</b></div>}
        </AuraModelViz>
      </div>
      <div className="aura-model-game-bar-row">
        <button type="button" className={`aura-model-game-token ${drag?.moved ? 'is-dragging' : ''}`} disabled={round.phase !== 'ready' || !round.current}
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={() => setDrag(null)}
          onClick={e => { if (e.detail === 0) round.launch() }}
          aria-label={round.current ? `Launch ${round.current.name} into Aura` : 'Loading motions'} title="Drag onto the model, or press to launch">
          <i aria-hidden="true">✦</i><span><small>{round.current?.demo ? 'Demo motion · demo scores' : 'Saved motion'}</small>{round.current?.name ?? 'Loading…'}</span>
        </button>
        <p className="aura-model-game-question">{question}</p>
        <div className="aura-model-game-actions">
          {round.phase === 'ready' && round.last && (['higher', 'lower'] as const).map(g => (
            <button key={g} type="button" className="aura-model-game-guess" aria-pressed={round.guess === g} onClick={() => round.setGuess(round.guess === g ? null : g)}>
              {g === 'higher' ? '▲ Higher' : '▼ Lower'}
            </button>
          ))}
          {round.phase === 'result'
            ? <button type="button" className="aura-model-game-go" onClick={round.next}>Next motion →</button>
            : <button type="button" className="aura-model-game-go" onClick={round.launch} disabled={round.phase !== 'ready' || !round.current}>{flying ? 'Flying…' : 'Launch'}</button>}
        </div>
      </div>
      {drag?.moved && <div className="aura-model-game-ghost" style={{ left: drag.x, top: drag.y }} aria-hidden="true">✦</div>}
      <p className="aura-model-viz__sr" aria-live="polite">{round.announce}</p>
      {children}
      <p className="aura-model-game-codex">Layer codex <b>{api.stats.codex.length}/{TOTAL}</b>{api.stats.codex.length === TOTAL ? ' · Model whisperer ✦' : ' · open each layer to unlock it'}</p>
    </>
  )
}
