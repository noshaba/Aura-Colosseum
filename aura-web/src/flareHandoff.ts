/**
 * Fairy-flare handoff between the loading screen (2D canvas overlay) and the
 * hero arena (WebGL). One continuous move:
 *
 *   loader figure-eight  ->  eased departure to the top of the hero's A|B
 *   ribbon  ->  hero split sweep down the ribbon  ->  sparkle dissolve.
 *
 * The hero registers where its sweep starts (viewport CSS px), the velocity the
 * sweep starts with and the head radius. The loader curves into that point with
 * exactly that velocity and radius, then calls `handOffFlare(t)` with the rAF
 * timestamp of arrival; the hero starts its sweep at that same timestamp, so
 * position and velocity are continuous. If either side is missing (no hero,
 * reduced motion, WebGL failure) nothing waits: the loader just fades and the
 * hero plays its own sweep on the first matchup.
 */
export type FlareEntry = {
  /** Sweep start, viewport CSS px. */
  x: number
  y: number
  /** Velocity at the start of the sweep, CSS px / s. */
  vx: number
  vy: number
  /** Halo radius of the head, CSS px. */
  radius: number
}

/** Split sweep timing, shared so both sides agree on the velocity profile. */
export const FLARE_SWEEP_SECONDS = 2.4
/** Start/end slope of the sweep's Hermite ease (1 = linear; >1 starts fast, slows past the VS, speeds out). */
export const FLARE_SWEEP_SLOPE = 1.5
/** Loader departure: seconds from leaving the figure-eight to reaching the ribbon top. */
export const FLARE_DEPART_SECONDS = 0.95

/** Sweep progress 0..1 for normalised time u (cubic Hermite, slope S at both ends). */
export function flareSweepEase(u: number) {
  const x = Math.min(1, Math.max(0, u)), S = FLARE_SWEEP_SLOPE
  const x2 = x * x, x3 = x2 * x
  return (x3 - 2 * x2 + x) * S + (-2 * x3 + 3 * x2) + (x3 - x2) * S
}

let loaderRunning = false
let target: (() => FlareEntry | null) | null = null
const listeners = new Set<(t: number | null) => void>()

export function setLoaderFlareRunning(running: boolean) { loaderRunning = running }
export function isLoaderFlareRunning() { return loaderRunning }

/** Hero: provide the sweep entry. Returns an unregister function. */
export function registerFlareTarget(fn: () => FlareEntry | null) {
  target = fn
  return () => { if (target === fn) target = null }
}
export function getFlareTarget(): FlareEntry | null { return target ? target() : null }

/**
 * Hero: called with the rAF timestamp at which the flare arrives at the entry,
 * or with null when the loader finished without a handoff (play your own sweep).
 */
export function onFlareHandoff(cb: (t: number | null) => void) {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}

/** Loader: the flare has arrived at the entry at rAF time `t` (null: loader ended without a handoff). */
export function handOffFlare(t: number | null) {
  if (!loaderRunning) return
  loaderRunning = false
  listeners.forEach(cb => cb(t))
}
