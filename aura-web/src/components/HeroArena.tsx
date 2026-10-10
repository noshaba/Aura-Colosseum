import { useCallback, useEffect, useRef, useState, type FocusEvent, type PointerEvent as ReactPointerEvent } from 'react'
import type { G1Preview } from '../three/g1Rig'
import { prepareClip, type PreparedClip } from '../three/g1Actor'
import { HeroArenaScene, type Choice, type Side, type SideTones } from '../three/heroArenaScene'
import { prefersReducedMotion } from '../motionPrefs'
import { useCharacter } from '../hooks/useCharacter'
import './hero-arena.css'

/*
 * Home-page Motion A vs Motion B arena.
 *
 * In live mode this is now Aura's complete collection loop: visitors can rate
 * bundled starter/reference G1 motions immediately, enter a prompt, generate a
 * fresh NVIDIA Kimodo batch, and rate those candidates without leaving the hero.
 * A/B votes are persisted to /preferences. Model/prior scores stay hidden while
 * the person is choosing so the label is not biased by Aura's prediction.
 */

export const HERO_VOTES_KEY = 'aura:hero-arena-votes:v2'
const EVALUATOR_STORAGE = 'aura:generated-evaluator-id:v1'
const MAX_STORED_VOTES = 500
const VOTE_XP = 10
const STREAK_BONUS_XP = 2
const STREAK_BONUS_CAP = 5
const LEVEL_XP = 60

type ManifestClip = { id: string; label: string; prompt: string; frames: number; fps: number; file: string }
type HeroClip = {
  id: string
  preferenceId: string
  label: string
  prompt: string
  frames: number
  fps: number
  url: string
  source: 'starter' | 'generated'
  preferenceGroup: string
  seed?: number
}
type Pair = { a: HeroClip; b: HeroClip }
type Game = { round: number; xp: number; streak: number; votes: number }
type HeroVote = {
  v: 2; ts: string; round: number; a: string; b: string
  choice: 'A' | 'B' | 'tie'; chosen: string | null; ms: number; input: 'pointer' | 'keyboard' | 'button'
  source: 'starter' | 'generated'
}
type GeneratedMotion = {
  id: string
  name: string
  frames: number
  fps: number
  preview_file: string | null
  candidate_index?: number
  generation_seed?: number
  preference_group?: string
  source?: string
}
type GeneratorStatus = {
  model_loaded?: boolean
  busy?: boolean
  cuda_available?: boolean
  cuda_name?: string | null
  default_diffusion_steps?: number
}
type TrainingData = {
  valid_comparison_count: number
  unique_motion_count: number
  reward_trainable: boolean
}
type PreferenceResponse = { id?: number; error?: string; training_data?: TrainingData }

const BASE = import.meta.env.BASE_URL || '/'
const clipCache = new Map<string, Promise<PreparedClip>>()

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

function loadClip(clip: HeroClip) {
  let p = clipCache.get(clip.id)
  if (!p) {
    p = fetch(clip.url, { cache: clip.source === 'generated' ? 'no-store' : 'default' })
      .then(r => { if (!r.ok) throw new Error(`Motion clip request failed (${r.status})`); return r.json() as Promise<G1Preview> })
      .then(prepareClip)
    p.catch(() => clipCache.delete(clip.id))
    clipCache.set(clip.id, p)
  }
  return p
}

function manifestToClip(clip: ManifestClip): HeroClip {
  return {
    id: `starter:${clip.id}`,
    preferenceId: `starter-${clip.id}`,
    label: clip.label,
    prompt: clip.prompt,
    frames: clip.frames,
    fps: clip.fps,
    url: `${BASE}demo/hero/${clip.file}`,
    source: 'starter',
    preferenceGroup: 'starter-general-motion-quality-v1',
  }
}

function shortPromptLabel(prompt: string, index?: number) {
  const cleaned = String(prompt || '')
    .replace(/^a\s+humanoid\s+robot\s+/i, '')
    .replace(/^the\s+humanoid\s+robot\s+/i, '')
    .replace(/^humanoid\s+robot\s+/i, '')
    .replace(/[.!?]+$/g, '')
    .trim()
  const words = cleaned.split(/\s+/).filter(Boolean)
  let label = ''
  for (const word of words) {
    const next = label ? `${label} ${word}` : word
    if (next.length > 34) break
    label = next
  }
  if (!label) label = cleaned.slice(0, 34) || 'Generated motion'
  label = label.charAt(0).toUpperCase() + label.slice(1)
  return index ? `${label} · ${index}` : label
}

function generatedToClip(motion: GeneratedMotion, fallbackPrompt?: string): HeroClip | null {
  if (!motion.preview_file?.endsWith('.g1.json')) return null
  return {
    id: `generated:${motion.id}`,
    preferenceId: motion.id,
    label: shortPromptLabel(motion.name || fallbackPrompt || 'Generated motion', motion.candidate_index),
    prompt: motion.name,
    frames: motion.frames,
    fps: motion.fps,
    url: `/aura-api/files/${encodeURIComponent(motion.preview_file)}`,
    source: 'generated',
    preferenceGroup: String(motion.preference_group || fallbackPrompt || motion.name || '').trim() || `generated:${motion.id}`,
    seed: motion.generation_seed,
  }
}

function shuffle<T>(xs: T[]) {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] }
  return a
}

function localRatedPairKeys() {
  try {
    const raw = window.localStorage.getItem(HERO_VOTES_KEY)
    const list: HeroVote[] = raw ? JSON.parse(raw) : []
    if (!Array.isArray(list)) return new Set<string>()
    return new Set(list.map(v => [v.a, v.b].sort().join('|')))
  } catch { return new Set<string>() }
}

function compatibleGroups(clips: HeroClip[]) {
  const groups = new Map<string, HeroClip[]>()
  for (const clip of clips) {
    const key = clip.preferenceGroup || clip.prompt || clip.label
    const list = groups.get(key) || []
    list.push(clip)
    groups.set(key, list)
  }
  return [...groups.entries()].filter(([, list]) => list.length >= 2)
}

/** Pick a continuous, trainable pair. Pairing stays inside one comparison group. */
function pickPair(clips: HeroClip[], prev: Pair | null, preferredGroup?: string | null): Pair {
  const rated = localRatedPairKeys()
  const prevKey = prev ? [prev.a.preferenceId, prev.b.preferenceId].sort().join('|') : ''
  const groups = compatibleGroups(clips)
  if (!groups.length) throw new Error('Aura needs at least two comparable motions')

  const pairsFor = (group: HeroClip[], unratedOnly: boolean) => {
    const pairs: Pair[] = []
    for (let i = 0; i < group.length; i += 1) {
      for (let j = i + 1; j < group.length; j += 1) {
        const key = [group[i].preferenceId, group[j].preferenceId].sort().join('|')
        if (key === prevKey) continue
        if (unratedOnly && rated.has(key)) continue
        pairs.push({ a: group[i], b: group[j] })
      }
    }
    return pairs
  }

  if (preferredGroup) {
    const preferred = groups.find(([key]) => key === preferredGroup)?.[1]
    if (preferred) {
      const unseen = pairsFor(preferred, true)
      if (unseen.length) return shuffle(unseen)[0]
      const anyPreferred = pairsFor(preferred, false)
      if (anyPreferred.length) return shuffle(anyPreferred)[0]
    }
  }

  for (const [, group] of shuffle(groups)) {
    const unseen = pairsFor(group, true)
    if (unseen.length) return shuffle(unseen)[0]
  }

  // A single evaluator can eventually exhaust every local pair. Keep the arena
  // usable by returning a different compatible pair; the backend still refuses
  // duplicate training labels, so repeated clicks cannot inflate the dataset.
  for (const [, group] of shuffle(groups)) {
    const anyPair = pairsFor(group, false)
    if (anyPair.length) return shuffle(anyPair)[0]
  }
  return { a: groups[0][1][0], b: groups[0][1][1] }
}

function storeVote(vote: HeroVote) {
  try {
    const raw = window.localStorage.getItem(HERO_VOTES_KEY)
    const list: HeroVote[] = raw ? JSON.parse(raw) : []
    const next = (Array.isArray(list) ? list : []).concat(vote).slice(-MAX_STORED_VOTES)
    window.localStorage.setItem(HERO_VOTES_KEY, JSON.stringify(next))
  } catch { /* storage unavailable: API persistence still works in live mode */ }
}

function isEditable(target: EventTarget | null) {
  const el = target as HTMLElement | null
  if (!el) return false
  return el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)
}

const NAV_BAND_PX = 96
const TONE_ATTR = { A: 'data-hero-tone-a', B: 'data-hero-tone-b' } as const
const NAV_PARTS = '.fx-nav__burger'
const LIGHT: SideTones = { A: 'light', B: 'light' }

function focusVisible(el: Element) {
  try { return el.matches(':focus-visible') } catch { return true }
}

export function HeroArena({ live = false }: { live?: boolean }) {
  const sectionRef = useRef<HTMLElement>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<HeroArenaScene | null>(null)
  const [reduced] = useState(prefersReducedMotion)
  const { character } = useCharacter()
  const characterRef = useRef(character)
  characterRef.current = character

  const [phase, setPhase] = useState<'loading' | 'ready' | 'error' | 'nowebgl'>('loading')
  const [pair, setPair] = useState<Pair | null>(null)
  const [mode, setMode] = useState<'starter' | 'generated'>('starter')
  const [game, setGame] = useState<Game>({ round: 1, xp: 0, streak: 0, votes: 0 })
  const [playing, setPlaying] = useState(!reduced)
  const [busy, setBusy] = useState(true)
  const [announce, setAnnounce] = useState('')
  const [tones, setTones] = useState<SideTones>(LIGHT)

  const [prompt, setPrompt] = useState('A humanoid robot takes several careful steps forward while maintaining balance.')
  const candidateCount = 2
  const [generating, setGenerating] = useState(false)
  const [generatorStatus, setGeneratorStatus] = useState<GeneratorStatus | null>(null)
  const [generationMessage, setGenerationMessage] = useState('')
  const [generationError, setGenerationError] = useState('')
  const [poolCount, setPoolCount] = useState(0)
  const [evaluatorId] = useState(localEvaluatorId)

  const clipsRef = useRef<HeroClip[]>([])
  const pairRef = useRef<Pair | null>(null)
  const nextPairRef = useRef<Pair | null>(null)
  const preferredGroupRef = useRef<string | null>(null)
  const gameRef = useRef(game)
  const busyRef = useRef(true)
  const roundStartRef = useRef(0)
  const inViewRef = useRef(false)
  const pickRef = useRef<(side: Side, input: HeroVote['input']) => void>(() => {})
  const btnFocusRef = useRef<Side | null>(null)
  const btnHoverRef = useRef<Side | null>(null)
  gameRef.current = game
  pairRef.current = pair

  const prefetchNext = useCallback((cur: Pair) => {
    if (!clipsRef.current.length) return
    const next = pickPair(clipsRef.current, cur)
    nextPairRef.current = next
    void loadClip(next.a).catch(() => {}); void loadClip(next.b).catch(() => {})
  }, [])

  const refreshGeneratorStatus = useCallback(async () => {
    if (!live) return
    try {
      const response = await fetch('/aura-api/generator/status', { cache: 'no-store' })
      if (response.ok) setGeneratorStatus(await response.json() as GeneratorStatus)
    } catch { setGeneratorStatus(null) }
  }, [live])

  useEffect(() => {
    if (!live) return
    void refreshGeneratorStatus()
    const id = window.setInterval(() => void refreshGeneratorStatus(), 3000)
    return () => window.clearInterval(id)
  }, [live, refreshGeneratorStatus])

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
      const [manifest, library] = await Promise.all([
        fetch(`${BASE}demo/hero/manifest.json`).then(r => {
          if (!r.ok) throw new Error(`Hero manifest request failed (${r.status})`)
          return r.json() as Promise<{ clips: ManifestClip[] }>
        }),
        live
          ? fetch('/aura-api/motions', { cache: 'no-store' })
              .then(async r => r.ok ? await r.json() as GeneratedMotion[] : [])
              .catch(() => [] as GeneratedMotion[])
          : Promise.resolve([] as GeneratedMotion[]),
        scene.init(characterRef.current),
      ])
      if (disposed) return
      const starters = manifest.clips.filter(c => c && c.id && c.file).map(manifestToClip)
      if (starters.length < 2) throw new Error('Hero manifest needs at least two clips')
      const existing = library
        .filter(motion => motion && !String(motion.id || '').startsWith('starter-') && motion.source !== 'starter_reference')
        .map(motion => generatedToClip(motion))
        .filter((clip): clip is HeroClip => Boolean(clip))
      const all = [...starters]
      const seen = new Set(starters.map(clip => clip.preferenceId))
      for (const clip of existing) {
        if (!seen.has(clip.preferenceId)) { seen.add(clip.preferenceId); all.push(clip) }
      }
      clipsRef.current = all
      setPoolCount(all.length)
      const first = pickPair(all, null, 'starter-general-motion-quality-v1')
      const [ca, cb] = await Promise.all([loadClip(first.a), loadClip(first.b)])
      if (disposed) return
      setPair(first)
      setMode(first.a.source === 'starter' ? 'starter' : 'generated')
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
  }, [reduced, prefetchNext, live])

  useEffect(() => {
    sceneRef.current?.setCharacter(character).catch(() => {})
  }, [character.id])

  // Mirror per-half tones onto <html> while the fixed nav is over the arena.
  const tonesRef = useRef(tones)
  tonesRef.current = tones
  const syncDocToneRef = useRef<() => void>(() => {})
  useEffect(() => {
    const root = document.documentElement
    let raf = 0, alive = true, settle = 0
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

  useEffect(() => {
    sceneRef.current?.setHud({
      round: game.round, xp: game.xp, streak: game.streak,
      level: Math.floor(game.xp / LEVEL_XP) + 1, levelProgress: (game.xp % LEVEL_XP) / LEVEL_XP, playing,
    })
  }, [game, playing, phase])
  useEffect(() => { sceneRef.current?.setPlaying(playing) }, [playing, phase])

  const persistPreference = useCallback(async (current: Pair, choice: 'A' | 'B') => {
    if (!live) return null
    const winner = choice === 'A' ? current.a : current.b
    const response = await fetch('/aura-api/preferences', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        left_id: current.a.preferenceId,
        right_id: current.b.preferenceId,
        winner_id: winner.preferenceId,
        evaluator_id: evaluatorId,
        context: current.a.source === 'starter'
          ? 'Starter/reference comparison: prefer the motion with better overall motion quality. This is not prompt-compliance labeling.'
          : 'Generated batch comparison: prefer the motion that better satisfies the shared prompt.',
      }),
    })
    const data = await response.json().catch(() => ({})) as PreferenceResponse
    if (!response.ok) throw new Error(data.error || 'Could not save Aura preference.')
    if (data.training_data?.reward_trainable) {
      void Promise.allSettled([
        fetch('/aura-api/reward/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }),
        fetch('/aura-api/prior/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }),
      ])
    }
    return data
  }, [evaluatorId, live])

  const installStarterPair = useCallback(async () => {
    const scene = sceneRef.current
    if (!scene || clipsRef.current.length < 2 || busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      const next = pickPair(clipsRef.current, pairRef.current, 'starter-general-motion-quality-v1')
      const [ca, cb] = await Promise.all([loadClip(next.a), loadClip(next.b)])
      nextPairRef.current = null
      setMode('starter')
      setPair(next)
      await scene.setMatchup({ clip: ca, label: next.a.label }, { clip: cb, label: next.b.label })
      prefetchNext(next)
      roundStartRef.current = performance.now()
      setGenerationMessage('Starter/reference motions active. Generated and saved motions remain in the continuous queue.')
    } finally {
      busyRef.current = false; setBusy(false)
    }
  }, [prefetchNext])

  const generate = useCallback(async () => {
    const scene = sceneRef.current
    if (!live || !scene || phase !== 'ready' || generating || !prompt.trim()) return
    const requestedPrompt = prompt.trim()
    setGenerating(true)
    setGenerationError('')
    setGenerationMessage(generatorStatus?.model_loaded
      ? `Aura is generating ${candidateCount} motions in the background. Keep rating the current motions.`
      : `Aura is loading NVIDIA Kimodo, then generating ${candidateCount} motions. Keep rating while it runs.`)
    try {
      const response = await fetch('/aura-api/generator/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: requestedPrompt, count: candidateCount }),
      })
      const data = await response.json() as { motions?: GeneratedMotion[]; motion?: GeneratedMotion; error?: string }
      const motions = data.motions || (data.motion ? [data.motion] : [])
      const clips = motions.map(motion => generatedToClip(motion, requestedPrompt)).filter((clip): clip is HeroClip => Boolean(clip))
      if (!response.ok || clips.length < 2) throw new Error(data.error || 'Aura did not receive at least two previewable generated motions.')

      const existingIds = new Set(clipsRef.current.map(clip => clip.preferenceId))
      const added = clips.filter(clip => !existingIds.has(clip.preferenceId))
      clipsRef.current = clipsRef.current.concat(added)
      setPoolCount(clipsRef.current.length)
      preferredGroupRef.current = clips[0].preferenceGroup
      nextPairRef.current = null

      // Warm the first generated matchup without interrupting the pair currently on screen.
      const generatedPair = pickPair(clipsRef.current, pairRef.current, clips[0].preferenceGroup)
      void Promise.all([loadClip(generatedPair.a), loadClip(generatedPair.b)]).catch(() => {})

      setGenerationMessage(`${added.length} new unique motion${added.length === 1 ? '' : 's'} added. Keep choosing: the generated motions are now in the continuous comparison queue and will appear next.`)
      window.dispatchEvent(new CustomEvent('aura:motion-generated', { detail: { ids: clips.map(clip => clip.preferenceId) } }))
      void refreshGeneratorStatus()
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : 'Aura generation failed.')
      setGenerationMessage('')
    } finally {
      setGenerating(false)
    }
  }, [generating, generatorStatus?.model_loaded, live, phase, prompt, refreshGeneratorStatus])

  // ---------------------------------------------------------------- game loop
  const vote = useCallback(async (choice: Choice, input: HeroVote['input']) => {
    const scene = sceneRef.current, current = pairRef.current
    if (!scene || !current || busyRef.current) return

    // Lock exactly once per visible matchup. The preference write is intentionally
    // NOT on the visual critical path: a slow local API must never make A/B feel dead.
    busyRef.current = true; setBusy(true)
    const g = gameRef.current
    const counted = choice !== 'skip'
    const gain = counted ? VOTE_XP + STREAK_BONUS_XP * Math.min(g.streak, STREAK_BONUS_CAP) : 0
    const nextGame: Game = { round: g.round + 1, xp: g.xp + gain, streak: counted ? g.streak + 1 : 0, votes: g.votes + (counted ? 1 : 0) }

    if (counted) {
      storeVote({
        v: 2, ts: new Date().toISOString(), round: g.round,
        a: current.a.preferenceId, b: current.b.preferenceId,
        choice: choice as HeroVote['choice'],
        chosen: choice === 'A' ? current.a.preferenceId : choice === 'B' ? current.b.preferenceId : null,
        ms: Math.round(performance.now() - roundStartRef.current), input,
        source: current.a.source,
      })
    }

    // Start persistence immediately, but let the animation and next-motion load run
    // at the same time. Any backend error is surfaced without swallowing the vote UX.
    const savePromise: Promise<PreferenceResponse | null> = (choice === 'A' || choice === 'B')
      ? persistPreference(current, choice).catch(error => {
          setGenerationError(error instanceof Error ? error.message : 'Could not save preference.')
          return null
        })
      : Promise.resolve(null)

    const said = choice === 'skip' ? 'Skipped.' : choice === 'tie' ? 'Tie recorded locally; no pairwise training label was added.' : `You picked Motion ${choice}.`
    setAnnounce(counted ? `${said} +${gain} XP.` : said)
    setGame(nextGame)

    try {
      // Decide and preload the next matchup before the celebration finishes. This
      // removes the dead gap that used to appear after clicking A/B.
      const preferred = preferredGroupRef.current
      const upcoming = preferred ? pickPair(clipsRef.current, current, preferred) : (nextPairRef.current ?? pickPair(clipsRef.current, current))
      if (preferred && upcoming.a.preferenceGroup === preferred) preferredGroupRef.current = null
      nextPairRef.current = null
      const nextClipsPromise = Promise.all([loadClip(upcoming.a), loadClip(upcoming.b)])

      // A paused arena should still accept A/B. Resume playback for the transition,
      // but do not require a separate first click just to resume.
      if (!playing) setPlaying(true)
      if (choice === 'A' || choice === 'B') scene.flashSide(choice)
      const [, [ca, cb]] = await Promise.all([scene.resolve(choice, gain), nextClipsPromise])
      if (sceneRef.current !== scene) return

      setPair(upcoming)
      const nextMode = upcoming.a.source === 'starter' ? 'starter' : 'generated'
      setMode(nextMode)
      await scene.setMatchup({ clip: ca, label: upcoming.a.label }, { clip: cb, label: upcoming.b.label })
      prefetchNext(upcoming)
      setAnnounce(`Round ${nextGame.round}: Motion A, ${upcoming.a.label}, versus Motion B, ${upcoming.b.label}.`)
      roundStartRef.current = performance.now()

      // Update training diagnostics when the save finishes, without holding the UI.
      void savePromise.then(saved => {
        if (!saved?.training_data) return
        setGenerationMessage(`${generating ? 'Generation continues in the background · ' : ''}${saved.training_data.valid_comparison_count} valid Aura comparisons across ${saved.training_data.unique_motion_count} unique motions. ${poolCount || clipsRef.current.length} motions are available in the picker.`)
      })
    } catch {
      if (sceneRef.current === scene) setPhase('error')
    } finally {
      if (sceneRef.current === scene) { busyRef.current = false; setBusy(false) }
    }
  }, [generating, persistPreference, playing, poolCount, prefetchNext])

  const pick = useCallback((side: Side, input: HeroVote['input']) => {
    // A/B always means "vote A/B". If playback is paused, the same click resumes
    // the motion and records the vote instead of consuming the click as Play.
    if (!playing) setPlaying(true)
    void vote(side, input)
  }, [playing, vote])
  pickRef.current = pick

  useEffect(() => {
    let voteTimer = 0
    const keyVote = (side: Side) => {
      if (busyRef.current || voteTimer) return
      sceneRef.current?.flashSide(side)
      voteTimer = window.setTimeout(() => { voteTimer = 0; pickRef.current(side, 'keyboard') }, 120)
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
        <p>Your browser could not start a WebGL context, so the motions can’t be shown here. Everything else on the page still works.</p>
      </div>
    </section>
  )

  return (
    <section ref={sectionRef} className="hero-arena" data-tone-a={tones.A} data-tone-b={tones.B}
      data-tone-mid={tones.A === tones.B ? tones.A : 'light'} aria-label="Motion A versus Motion B: pick the motion you prefer">
      <div ref={hostRef} className="hero-arena__canvas" aria-hidden="true" />

      {live && <div className="hero-arena__generator" role="group" aria-label="Generate motions from text">
        <div className="hero-arena__generator-main">
          <label htmlFor="hero-motion-prompt">Generate a motion</label>
          <input
            id="hero-motion-prompt"
            type="text"
            value={prompt}
            disabled={generating}
            onChange={event => { setPrompt(event.target.value); setGenerationError('') }}
            onKeyDown={event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') void generate() }}
            placeholder="Describe what the Unitree G1 should do…"
          />
          <button type="button" onClick={() => void generate()} disabled={generating || phase !== 'ready' || !prompt.trim()}>
            {generating ? 'Aura is generating…' : `Generate ${candidateCount}`}
          </button>
        </div>
        <div className="hero-arena__generator-meta">
          <span className="hero-arena__candidate-limit">2 motions per generation</span>
          <span className={generatorStatus?.cuda_available ? 'is-ready' : ''}>{generating ? 'AURA IS GENERATING' : generatorStatus?.cuda_available ? 'NVIDIA KIMODO READY' : 'GENERATOR OFFLINE'}</span>
          {mode === 'generated' && <button type="button" className="hero-arena__starter-button" onClick={() => void installStarterPair()} disabled={busy}>Starter motions</button>}
        </div>
        {(generationMessage || generationError) && <div className={generationError ? 'hero-arena__generator-message is-error' : 'hero-arena__generator-message'}>{generationError || generationMessage}</div>}
      </div>}

      {generating && <div className="hero-arena__generating" aria-live="polite"><strong>AURA IS GENERATING</strong><span>{candidateCount} unique candidates · fresh seeds</span></div>}

      <div className="hero-arena__source" aria-hidden="true">
        <span>{mode === 'starter' ? 'STARTER / REFERENCE' : 'GENERATED / SAVED'}</span>
        <strong>{pair ? `${pair.a.label}  vs  ${pair.b.label}` : 'Loading motions…'}</strong>
        <small>{poolCount} motions in continuous queue{generating ? ' · generating in background' : ''}</small>
      </div>

      <p className="hero-arena__sr" aria-live="polite">{announce}</p>
      <p className="hero-arena__sr">
        {pair ? `Round ${game.round}. Motion A: ${pair.a.label} (${pair.a.prompt}). Motion B: ${pair.b.label} (${pair.b.prompt}). ` : ''}
        {`XP ${game.xp}, level ${level}, streak ${game.streak}. In live mode A/B choices are persisted as Aura preference data; model scores are hidden while you vote.`}
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
