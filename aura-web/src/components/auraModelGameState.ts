/**
 * "Feed the model" mini-game state for the Aura model page.
 *
 * Loop: a motion waits at the gate; the player may bet whether Aura scores it
 * HIGHER or LOWER than the previous motion, then launches it. The forward pass
 * steps through the eight stages (~1 s per stage, so the camera can zoom onto each one),
 * the orb pops with the score, and a correct bet builds a streak (XP is tallied in storage only;
 * the XP HUD belongs to the home arena).
 *
 * Scores are real Aura rewards when a trained model has scored at least two
 * previewable motions; otherwise the bundled sample motions get deterministic
 * pseudo-scores and everything is labelled as demo. Stats (XP, streak, codex)
 * persist in localStorage under their own key, separate from the arena.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { prefersReducedMotion } from '../motionPrefs'
import { AURA_MODEL_LAYER_IDS, type AuraModelLayerId, type AuraModelMotion } from '../three/auraModelScene'

export type GamePoolItem = { id: string; name: string; prompt?: string; file: string | null; score: number; norm: number; demo: boolean }
export type GameStats = { xp: number; streak: number; best: number; rounds: number; codex: AuraModelLayerId[] }
export type GameOutcome = 'win' | 'miss' | 'tie' | 'warmup' | 'free'
export type Guess = 'higher' | 'lower'

const STORAGE_KEY = 'aura:model-game:v1'
const STEP_MS = 1000

const EMPTY: GameStats = { xp: 0, streak: 0, best: 0, rounds: 0, codex: [] }

function readStats(): GameStats {
  try {
    const raw = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || 'null') as Partial<GameStats> | null
    if (!raw || typeof raw !== 'object') return EMPTY
    const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.floor(v) : 0)
    const codex = Array.isArray(raw.codex) ? raw.codex.filter((id): id is AuraModelLayerId => (AURA_MODEL_LAYER_IDS as readonly string[]).includes(id as string)) : []
    return { xp: num(raw.xp), streak: num(raw.streak), best: num(raw.best), rounds: num(raw.rounds), codex: [...new Set(codex)] }
  } catch { return EMPTY }
}

function writeStats(s: GameStats) {
  try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s)) } catch { /* private mode / blocked storage: keep in memory */ }
}

export type GameStatsApi = {
  stats: GameStats
  /** Scores a finished launch; returns the XP gained. */
  record(outcome: GameOutcome): number
  /** First visit of a stage; returns true when it was newly unlocked. */
  unlock(id: AuraModelLayerId): boolean
}

export function xpFor(outcome: GameOutcome, streakBefore: number) {
  if (outcome === 'win') return 10 + 5 * Math.min(streakBefore, 4)
  return outcome === 'miss' ? 0 : 2
}

export function useGameStats(): GameStatsApi {
  const [stats, setStats] = useState<GameStats>(() => (typeof window === 'undefined' ? EMPTY : readStats()))
  const ref = useRef(stats)
  const commit = useCallback((next: GameStats) => { ref.current = next; setStats(next); writeStats(next) }, [])
  const record = useCallback((outcome: GameOutcome) => {
    const s = ref.current
    const gain = xpFor(outcome, s.streak)
    const streak = outcome === 'win' ? s.streak + 1 : outcome === 'miss' ? 0 : s.streak
    commit({ ...s, xp: s.xp + gain, streak, best: Math.max(s.best, streak), rounds: s.rounds + 1 })
    return gain
  }, [commit])
  const unlock = useCallback((id: AuraModelLayerId) => {
    const s = ref.current
    if (s.codex.includes(id)) return false
    const codex = [...s.codex, id]
    commit({ ...s, codex, xp: s.xp + 5 + (codex.length === AURA_MODEL_LAYER_IDS.length ? 25 : 0) })
    return true
  }, [commit])
  return { stats, record, unlock }
}

// ---------------------------------------------------------------- pool
/** Deterministic demo score in [-1.5, 1.5] from the motion id (FNV-1a). */
export function demoScore(id: string) {
  let h = 0x811c9dc5
  for (let i = 0; i < id.length; i++) { h ^= id.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0 }
  return Math.round(((h % 3001) / 1000 - 1.5) * 1000) / 1000
}

type DemoMotion = { id: string; name: string; prompt?: string; preview_file?: string | null }
let demoPromise: Promise<GamePoolItem[]> | null = null
const FALLBACK_NAMES = ['Stumble recovery', 'Reach and turn', 'Side shuffle', 'Hop forward']

function loadDemoPool(): Promise<GamePoolItem[]> {
  demoPromise ??= fetch('/demo/motions.json')
    .then(r => (r.ok ? r.json() as Promise<{ motions?: DemoMotion[] }> : { motions: [] }))
    .then(d => (Array.isArray(d.motions) ? d.motions : []))
    .catch(() => [] as DemoMotion[])
    .then(list => {
      const items = list.length
        ? list.map(m => ({ id: m.id, name: m.name, prompt: m.prompt, file: m.preview_file?.endsWith('.g1.json') ? m.preview_file : null }))
        : FALLBACK_NAMES.map((name, i) => ({ id: `demo-${i}`, name, prompt: undefined, file: null }))
      const scores = items.map(m => demoScore(m.id))
      const lo = Math.min(...scores), hi = Math.max(...scores)
      return items.map((m, i) => ({ ...m, score: scores[i], norm: hi > lo ? (scores[i] - lo) / (hi - lo) : 0.5, demo: true }))
    })
  return demoPromise
}

/** Real scored motions when available (>= 2), else the bundled demo pool. */
export function useGamePool(real: GamePoolItem[]) {
  const [demo, setDemo] = useState<GamePoolItem[]>([])
  const useReal = real.length >= 2
  useEffect(() => {
    if (useReal) return
    let live = true
    void loadDemoPool().then(items => { if (live) setDemo(items) })
    return () => { live = false }
  }, [useReal])
  return useReal ? real : demo
}

// ---------------------------------------------------------------- joints cache
const jointCache = new Map<string, Promise<AuraModelMotion | null>>()
export function loadJoints(file: string): Promise<AuraModelMotion | null> {
  let p = jointCache.get(file)
  if (!p) {
    p = fetch(file)
      .then(r => (r.ok ? r.json() as Promise<{ positions?: number[][][]; parents?: number[] }> : null))
      .then(d => (d?.positions?.length ? { positions: d.positions, parents: d.parents } : null))
      .catch(() => null)
    jointCache.set(file, p)
    void p.then(v => { if (!v) jointCache.delete(file) }) // retry later after a failure
  }
  return p
}

export function useJoints(file: string | null) {
  const [motion, setMotion] = useState<AuraModelMotion | null>(null)
  useEffect(() => {
    if (!file) { setMotion(null); return }
    let live = true
    void loadJoints(file).then(m => { if (live) setMotion(m) })
    return () => { live = false }
  }, [file])
  return motion
}

// ---------------------------------------------------------------- round
export type RoundPhase = 'ready' | 'flying' | 'result'
export type RoundState = {
  phase: RoundPhase
  index: number
  guess: Guess | null
  stage: AuraModelLayerId | null
  last: { item: GamePoolItem; score: number } | null
  outcome: GameOutcome | null
  gained: number
  popKey: number
  announce: string
  /** Streak after this round (for the reveal). */
  streak: number
}

export type FeedRound = RoundState & {
  current: GamePoolItem | null
  setGuess(g: Guess | null): void
  launch(): void
  next(): void
}

const fmt = (n: number) => (n >= 0 ? '+' : '') + n.toFixed(3)

/**
 * One feed-the-model round loop. `auto` (hero): launches by itself after a pause and
 * moves on after the reveal; any guess launches immediately.
 */
export function useFeedRound(pool: GamePoolItem[], api: GameStatsApi, opts: { auto?: boolean; start?: number; onStage?: (id: AuraModelLayerId) => void } = {}): FeedRound {
  const [st, setSt] = useState<RoundState>({ phase: 'ready', index: opts.start ?? 0, guess: null, stage: null, last: null, outcome: null, gained: 0, popKey: 0, announce: '', streak: 0 })
  const timers = useRef<number[]>([])
  const stRef = useRef(st)
  stRef.current = st
  const onStage = useRef(opts.onStage)
  onStage.current = opts.onStage
  const current = pool.length ? pool[st.index % pool.length] : null
  const curRef = useRef(current)
  curRef.current = current

  const clear = () => { timers.current.forEach(id => window.clearTimeout(id)); timers.current = [] }
  useEffect(() => clear, [])

  const resolve = useCallback(() => {
    const s = stRef.current, item = curRef.current
    if (!item) return
    let outcome: GameOutcome
    if (!s.last) outcome = 'warmup'
    else if (!s.guess) outcome = 'free'
    else if (item.score === s.last.score) outcome = 'tie'
    else outcome = (item.score > s.last.score) === (s.guess === 'higher') ? 'win' : 'miss'
    const streakBefore = api.stats.streak
    const gained = api.record(outcome)
    const streak = outcome === 'win' ? streakBefore + 1 : outcome === 'miss' ? 0 : streakBefore
    const demo = item.demo ? ' (demo score)' : ''
    const announce = outcome === 'win' ? `Correct! Aura scored ${item.name} ${fmt(item.score)}${demo}, ${s.guess} than ${fmt(s.last!.score)}. Streak ${streak}.`
      : outcome === 'miss' ? `Not quite. Aura scored ${item.name} ${fmt(item.score)}${demo}, ${s.guess === 'higher' ? 'lower' : 'higher'} than ${fmt(s.last!.score)}. Streak reset.`
      : `Aura scored ${item.name} ${fmt(item.score)}${demo}.`
    setSt(p => ({ ...p, phase: 'result', stage: 'reward', outcome, gained, streak, popKey: p.popKey + 1, announce }))
  }, [api])

  const launch = useCallback(() => {
    const s = stRef.current
    if (s.phase !== 'ready' || !curRef.current) return
    clear()
    if (prefersReducedMotion()) {
      setSt(p => ({ ...p, phase: 'flying', stage: 'reward', announce: '' }))
      onStage.current?.('reward')
      resolve()
      return
    }
    setSt(p => ({ ...p, phase: 'flying', announce: `Launching ${curRef.current?.name ?? 'motion'} into Aura.` }))
    AURA_MODEL_LAYER_IDS.forEach((id, k) => {
      timers.current.push(window.setTimeout(() => { setSt(p => ({ ...p, stage: id })); onStage.current?.(id) }, k * STEP_MS))
    })
    timers.current.push(window.setTimeout(resolve, AURA_MODEL_LAYER_IDS.length * STEP_MS + 120))
  }, [resolve])

  const next = useCallback(() => {
    clear()
    const item = curRef.current
    setSt(p => ({ ...p, phase: 'ready', index: p.index + 1, guess: null, stage: null, outcome: null, gained: 0, announce: '',
      last: item && p.phase === 'result' ? { item, score: item.score } : p.last }))
  }, [])

  const setGuess = useCallback((g: Guess | null) => {
    if (stRef.current.phase !== 'ready') return
    setSt(p => ({ ...p, guess: g }))
  }, [])

  // Hero autoplay: launch after a pause (immediately once a guess is made), move on after the reveal.
  useEffect(() => {
    if (!opts.auto || !pool.length) return
    let id = 0
    if (st.phase === 'ready') id = window.setTimeout(launch, st.guess ? 250 : st.last ? 7000 : 3200)
    else if (st.phase === 'result') id = window.setTimeout(next, 4200)
    return () => window.clearTimeout(id)
  }, [opts.auto, pool.length, st.phase, st.guess, st.last, launch, next])

  return useMemo(() => ({ ...st, current, setGuess, launch, next }), [st, current, setGuess, launch, next])
}
