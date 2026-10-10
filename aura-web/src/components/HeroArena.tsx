import { useCallback, useEffect, useRef, useState, type FocusEvent, type PointerEvent as ReactPointerEvent } from 'react'
import type { G1Preview } from '../three/g1Rig'
import { prepareClip, type PreparedClip } from '../three/g1Actor'
import { HeroArenaScene, type Choice, type Side, type SideTones } from '../three/heroArenaScene'
import { prefersReducedMotion } from '../motionPrefs'
import { useCharacter } from '../hooks/useCharacter'
import './hero-arena.css'

/*
 * Home-page "Motion A vs Motion B" arena.
 * Research-integrity rules (EXPERIMENT_PROTOCOL.md): no model / prior score and no
 * "correct answer" is ever shown, and nothing rewards agreeing with a model. XP and
 * streaks are for participation only. Hero votes are onboarding data: they stay in
 * this browser's localStorage and are never POSTed to /preferences.
 */

export const HERO_VOTES_KEY = 'aura:hero-arena-votes:v1'
const MAX_STORED_VOTES = 500
const VOTE_XP = 10
const STREAK_BONUS_XP = 2
const STREAK_BONUS_CAP = 5
const LEVEL_XP = 60

type ManifestClip = { id: string; label: string; prompt: string; frames: number; fps: number; file: string }
type Pair = { a: ManifestClip; b: ManifestClip }
type Game = { round: number; xp: number; streak: number; votes: number }
type HeroVote = {
  v: 1; ts: string; round: number; a: string; b: string
  choice: 'A' | 'B' | 'tie'; chosen: string | null; ms: number; input: 'pointer' | 'keyboard' | 'button'
}

const BASE = import.meta.env.BASE_URL || '/'
const clipCache = new Map<string, Promise<PreparedClip>>()

function loadClip(clip: ManifestClip) {
  let p = clipCache.get(clip.id)
  if (!p) {
    p = fetch(`${BASE}demo/hero/${clip.file}`)
      .then(r => { if (!r.ok) throw new Error(`Hero clip request failed (${r.status})`); return r.json() as Promise<G1Preview> })
      .then(prepareClip)
    p.catch(() => clipCache.delete(clip.id))
    clipCache.set(clip.id, p)
  }
  return p
}

function shuffle<T>(xs: T[]) {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] }
  return a
}

/** Two different clips; avoids reusing either clip of the previous round when possible. */
function pickPair(clips: ManifestClip[], prev: Pair | null): Pair {
  const avoid = new Set(prev ? [prev.a.id, prev.b.id] : [])
  const fresh = clips.filter(c => !avoid.has(c.id))
  const pool = fresh.length >= 2 ? fresh : clips
  const [a, b] = shuffle(pool)
  return { a, b }
}

function storeVote(vote: HeroVote) {
  try {
    const raw = window.localStorage.getItem(HERO_VOTES_KEY)
    const list: HeroVote[] = raw ? JSON.parse(raw) : []
    const next = (Array.isArray(list) ? list : []).concat(vote).slice(-MAX_STORED_VOTES)
    window.localStorage.setItem(HERO_VOTES_KEY, JSON.stringify(next))
  } catch { /* storage unavailable (private mode / quota): the game still works */ }
}

function isEditable(target: EventTarget | null) {
  const el = target as HTMLElement | null
  if (!el) return false
  return el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)
}

/** Hero still sits under the fixed nav when its bottom edge is below the nav band. */
const NAV_BAND_PX = 96
const TONE_ATTR = { A: 'data-hero-tone-a', B: 'data-hero-tone-b' } as const
/** Nav parts that follow the half of the hero they sit over (hero-arena.css). */
const NAV_PARTS = '.fx-nav__burger, .fx-nav__pill'
const LIGHT: SideTones = { A: 'light', B: 'light' }

function focusVisible(el: Element) {
  try { return el.matches(':focus-visible') } catch { return true }
}

export function HeroArena() {
  const sectionRef = useRef<HTMLElement>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<HeroArenaScene | null>(null)
  const [reduced] = useState(prefersReducedMotion)
  // Selected body (nav switch / picker). Read through a ref at mount so the scene is never
  // rebuilt for it; the effect below swaps the robots live instead.
  const { character } = useCharacter()
  const characterRef = useRef(character)
  characterRef.current = character
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error' | 'nowebgl'>('loading')
  const [pair, setPair] = useState<Pair | null>(null)
  const [game, setGame] = useState<Game>({ round: 1, xp: 0, streak: 0, votes: 0 })
  const [playing, setPlaying] = useState(!reduced)
  const [busy, setBusy] = useState(true)
  const [announce, setAnnounce] = useState('')
  // Per-half colour scheme reported by the scene (A = left, B = right); drives the DOM controls and the nav.
  const [tones, setTones] = useState<SideTones>(LIGHT)

  const clipsRef = useRef<ManifestClip[]>([])
  const pairRef = useRef<Pair | null>(null)
  /** The next round's pair, chosen (and its clips prefetched) as soon as the current matchup is set. */
  const nextPairRef = useRef<Pair | null>(null)
  const gameRef = useRef(game)
  const busyRef = useRef(true)
  const roundStartRef = useRef(0)
  const inViewRef = useRef(false)
  const pickRef = useRef<(side: Side, input: HeroVote['input']) => void>(() => {})
  const btnFocusRef = useRef<Side | null>(null)
  const btnHoverRef = useRef<Side | null>(null)
  gameRef.current = game
  pairRef.current = pair

  // Pick the next round's pair now and warm just those two clips.
  const prefetchNext = useCallback((cur: Pair) => {
    const next = pickPair(clipsRef.current, cur)
    nextPairRef.current = next
    void loadClip(next.a).catch(() => {}); void loadClip(next.b).catch(() => {})
  }, [])

  // ---------------------------------------------------------------- scene lifecycle
  useEffect(() => {
    const host = hostRef.current, section = sectionRef.current
    if (!host || !section) return
    let disposed = false
    let scene: HeroArenaScene
    try {
      scene = new HeroArenaScene(host, {
        reducedMotion: reduced,
        modelBase: `${BASE}models/g1-native/`,
        callbacks: {
          onPick: side => pickRef.current(side, 'pointer'),
          onPlay: () => setPlaying(true),
          onTone: t => { if (!disposed) setTones(t) },
          onContextLost: () => { if (!disposed) setPhase('nowebgl') },
        },
      })
    } catch {
      setPhase('nowebgl')
      return
    }
    sceneRef.current = scene

    let visible = false
    const sync = () => scene.setActive(visible && !document.hidden)
    const io = typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(([e]) => { visible = e.isIntersecting; inViewRef.current = visible; sync() }, { threshold: 0.05 })
      : null
    if (io) io.observe(section); else { visible = true; inViewRef.current = true; sync() }
    document.addEventListener('visibilitychange', sync)

    ;(async () => {
      const [manifest] = await Promise.all([
        fetch(`${BASE}demo/hero/manifest.json`).then(r => {
          if (!r.ok) throw new Error(`Hero manifest request failed (${r.status})`)
          return r.json() as Promise<{ clips: ManifestClip[] }>
        }),
        scene.init(characterRef.current),
      ])
      if (disposed) return
      const clips = manifest.clips.filter(c => c && c.id && c.file)
      if (clips.length < 2) throw new Error('Hero manifest needs at least two clips')
      clipsRef.current = clips
      const first = pickPair(clips, null)
      const [ca, cb] = await Promise.all([loadClip(first.a), loadClip(first.b)])
      if (disposed) return
      setPair(first)
      setPhase('ready')
      await scene.setMatchup({ clip: ca, label: first.a.label }, { clip: cb, label: first.b.label })
      if (disposed) return
      roundStartRef.current = performance.now()
      busyRef.current = false; setBusy(false)
      prefetchNext(first)
    })().catch(() => { if (!disposed) setPhase('error') })

    return () => {
      disposed = true
      io?.disconnect()
      document.removeEventListener('visibilitychange', sync)
      scene.dispose()
      sceneRef.current = null
      setTones(LIGHT)
    }
  }, [reduced, prefetchNext])

  // Live character swap (no remount; same id as the scene's is a no-op, so the mount
  // pass above is not doubled). Failures leave the current robots in place.
  useEffect(() => {
    sceneRef.current?.setCharacter(character).catch(() => {})
  }, [character.id])

  // Mirror the per-half tones onto <html> while the hero is under the fixed nav, so
  // the nav can follow the split (hero-arena.css). Each nav part is tagged with the
  // half it sits over, and the logo (which can straddle the centre) gets the split
  // position in its own coordinates. Everything is removed when scrolled away / unmounted.
  const tonesRef = useRef(tones)
  tonesRef.current = tones
  const syncDocToneRef = useRef<() => void>(() => {})
  useEffect(() => {
    const root = document.documentElement
    let raf = 0, alive = true, settle = 0
    // Last "hero under the nav" state; the attributes are only touched when it flips.
    let wasUnder: boolean | null = null
    let navEls: HTMLElement[] = []
    const clearNav = () => {
      document.querySelectorAll<HTMLElement>(NAV_PARTS).forEach(el => { delete el.dataset.heroSide })
      root.style.removeProperty('--hero-logo-split')
    }
    const apply = () => {
      raf = 0
      const r = sectionRef.current?.getBoundingClientRect()
      const underNav = !!r && r.top <= 1 && r.bottom > NAV_BAND_PX
      if (!underNav) {
        if (wasUnder !== false) {
          wasUnder = false
          root.removeAttribute(TONE_ATTR.A); root.removeAttribute(TONE_ATTR.B); clearNav()
          navEls = []
        }
        return
      }
      if (wasUnder !== true) { wasUnder = true; navEls = Array.from(document.querySelectorAll<HTMLElement>(NAV_PARTS)) }
      root.setAttribute(TONE_ATTR.A, tonesRef.current.A)
      root.setAttribute(TONE_ATTR.B, tonesRef.current.B)
      const mid = r.left + r.width / 2
      navEls.forEach(el => {
        const b = el.getBoundingClientRect()
        if (b.width) el.dataset.heroSide = b.left + b.width / 2 < mid ? 'a' : 'b'
      })
      const logo = document.querySelector<HTMLElement>('.fx-nav__logo img')
      if (logo) root.style.setProperty('--hero-logo-split', `${Math.round(mid - logo.getBoundingClientRect().left)}px`)
    }
    const schedule = () => { if (alive && !raf) raf = requestAnimationFrame(apply) }
    syncDocToneRef.current = apply
    apply()
    // The nav pill folds on scroll; re-measure after its .35s width transition too.
    const onScroll = () => { schedule(); window.clearTimeout(settle); settle = window.setTimeout(schedule, 400) }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', schedule)
    return () => {
      alive = false
      cancelAnimationFrame(raf)
      window.clearTimeout(settle)
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', schedule)
      syncDocToneRef.current = () => {}
      root.removeAttribute(TONE_ATTR.A); root.removeAttribute(TONE_ATTR.B); clearNav()
    }
  }, [phase])
  useEffect(() => { syncDocToneRef.current() }, [tones])

  // Keep the scene's HUD / play state in step with React state.
  useEffect(() => {
    sceneRef.current?.setHud({
      round: game.round, xp: game.xp, streak: game.streak,
      level: Math.floor(game.xp / LEVEL_XP) + 1, levelProgress: (game.xp % LEVEL_XP) / LEVEL_XP, playing,
    })
  }, [game, playing, phase])
  useEffect(() => { sceneRef.current?.setPlaying(playing) }, [playing, phase])

  // ---------------------------------------------------------------- game loop
  const vote = useCallback(async (choice: Choice, input: HeroVote['input']) => {
    const scene = sceneRef.current, current = pairRef.current
    if (!scene || !current || busyRef.current) return
    busyRef.current = true; setBusy(true)
    const g = gameRef.current
    const counted = choice !== 'skip'
    const gain = counted ? VOTE_XP + STREAK_BONUS_XP * Math.min(g.streak, STREAK_BONUS_CAP) : 0
    const next: Game = { round: g.round + 1, xp: g.xp + gain, streak: counted ? g.streak + 1 : 0, votes: g.votes + (counted ? 1 : 0) }
    if (counted) {
      storeVote({
        v: 1, ts: new Date().toISOString(), round: g.round, a: current.a.id, b: current.b.id,
        choice: choice as HeroVote['choice'],
        chosen: choice === 'A' ? current.a.id : choice === 'B' ? current.b.id : null,
        ms: Math.round(performance.now() - roundStartRef.current), input,
      })
    }
    const said = choice === 'skip' ? 'Skipped.' : choice === 'tie' ? 'Tie recorded.' : `You picked Motion ${choice}.`
    setAnnounce(counted ? `${said} +${gain} XP. Streak ${next.streak}. Loading round ${next.round}.` : `${said} Loading round ${next.round}.`)
    setGame(next)
    try {
      const upcoming = nextPairRef.current ?? pickPair(clipsRef.current, current)
      const loading = Promise.all([loadClip(upcoming.a), loadClip(upcoming.b)])
      await scene.resolve(choice, gain)
      const [ca, cb] = await loading
      if (sceneRef.current !== scene) return
      setPair(upcoming)
      await scene.setMatchup({ clip: ca, label: upcoming.a.label }, { clip: cb, label: upcoming.b.label })
      prefetchNext(upcoming)
      setAnnounce(`Round ${next.round}: Motion A, ${upcoming.a.label}, versus Motion B, ${upcoming.b.label}.`)
      roundStartRef.current = performance.now()
    } catch {
      if (sceneRef.current === scene) setPhase('error')
    } finally {
      if (sceneRef.current === scene) { busyRef.current = false; setBusy(false) }
    }
  }, [prefetchNext])
  const pick = useCallback((side: Side, input: HeroVote['input']) => {
    if (!playing) { setPlaying(true); return } // first interaction starts playback
    void vote(side, input)
  }, [playing, vote])
  pickRef.current = pick

  // Keyboard is the primary way to vote: A / ArrowLeft, B / ArrowRight (T = tie, S = skip),
  // only while the arena is on screen and never while typing in a field. A key vote first
  // flashes its half navy for a beat, then registers (the canvas pointer only orbits now).
  useEffect(() => {
    let voteTimer = 0
    const keyVote = (side: Side) => {
      if (!playing) { pickRef.current(side, 'keyboard'); return } // first press starts playback
      if (busyRef.current || voteTimer) return
      sceneRef.current?.flashSide(side)
      voteTimer = window.setTimeout(() => { voteTimer = 0; pickRef.current(side, 'keyboard') }, 160)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.repeat || e.metaKey || e.ctrlKey || e.altKey || isEditable(e.target)) return
      if (!inViewRef.current || phase !== 'ready') return
      const k = e.key.toLowerCase()
      if (k === 'a' || e.key === 'ArrowLeft') { e.preventDefault(); keyVote('A') }
      else if (k === 'b' || e.key === 'ArrowRight') { e.preventDefault(); keyVote('B') }
      else if (k === 't' && playing) { e.preventDefault(); void vote('tie', 'keyboard') }
      else if (k === 's' && playing) { e.preventDefault(); void vote('skip', 'keyboard') }
    }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); window.clearTimeout(voteTimer) }
  }, [phase, playing, vote])

  // Keyboard focus (focus-visible only, so a tap or click doesn't leave the dark
  // scheme stuck on a focused button) or mouse hover of the A/B buttons.
  const syncSide = () => sceneRef.current?.setFocusSide(btnFocusRef.current ?? btnHoverRef.current)
  const sideHandlers = (side: Side) => ({
    onFocus: (e: FocusEvent<HTMLButtonElement>) => { btnFocusRef.current = focusVisible(e.currentTarget) ? side : null; syncSide() },
    onBlur: () => { btnFocusRef.current = null; syncSide() },
    onPointerEnter: (e: ReactPointerEvent<HTMLButtonElement>) => { if (e.pointerType === 'mouse') { btnHoverRef.current = side; syncSide() } },
    onPointerLeave: () => { if (btnHoverRef.current === side) { btnHoverRef.current = null; syncSide() } },
  })
  const disabled = phase !== 'ready' || busy
  const level = Math.floor(game.xp / LEVEL_XP) + 1

  if (phase === 'nowebgl') return (
    <section className="hero-arena hero-arena--fallback" aria-label="Motion A versus Motion B">
      <div className="hero-arena__fallback">
        <strong>The 3D motion arena needs WebGL.</strong>
        <p>Your browser could not start a WebGL context, so the robots can’t be shown here. Everything else on the page still works.</p>
      </div>
    </section>
  )

  return (
    <section ref={sectionRef} className="hero-arena" data-tone-a={tones.A} data-tone-b={tones.B}
      data-tone-mid={tones.A === tones.B ? tones.A : 'light'} aria-label="Motion A versus Motion B: pick the motion you prefer">
      <div ref={hostRef} className="hero-arena__canvas" aria-hidden="true" />
      <p className="hero-arena__sr" aria-live="polite">{announce}</p>
      <p className="hero-arena__sr">
        {pair ? `Round ${game.round}. Motion A: ${pair.a.label} (${pair.a.prompt}). Motion B: ${pair.b.label} (${pair.b.prompt}). ` : ''}
        {`XP ${game.xp}, level ${level}, streak ${game.streak}. Votes stay in this browser and earn XP for taking part; there is no right answer.`}
      </p>
      {phase !== 'ready' && (
        <div className="hero-arena__status" role="status">
          {phase === 'loading' ? 'Loading arena…' : 'The arena could not load its motions. Reload the page to try again.'}
        </div>
      )}
      <div className="hero-arena__controls" role="group" aria-label="Vote controls">
        <button type="button" className="hero-arena__btn hero-arena__btn--side hero-arena__btn--a" disabled={disabled}
          aria-label={pair ? `Vote for Motion A: ${pair.a.label}` : 'Vote for Motion A'} aria-keyshortcuts="A ArrowLeft"
          onClick={() => pick('A', 'button')} {...sideHandlers('A')}>A</button>
        <div className="hero-arena__mid">
          <button type="button" className="hero-arena__btn" aria-pressed={playing}
            aria-label={playing ? 'Pause both motions' : 'Play both motions'} onClick={() => setPlaying(p => !p)}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <button type="button" className="hero-arena__btn" disabled={disabled || !playing} aria-keyshortcuts="T" onClick={() => void vote('tie', 'button')}>Tie</button>
          <button type="button" className="hero-arena__btn" disabled={disabled || !playing} aria-keyshortcuts="S" onClick={() => void vote('skip', 'button')}>Skip</button>
        </div>
        <button type="button" className="hero-arena__btn hero-arena__btn--side hero-arena__btn--b" disabled={disabled}
          aria-label={pair ? `Vote for Motion B: ${pair.b.label}` : 'Vote for Motion B'} aria-keyshortcuts="B ArrowRight"
          onClick={() => pick('B', 'button')} {...sideHandlers('B')}>B</button>
      </div>
    </section>
  )
}
