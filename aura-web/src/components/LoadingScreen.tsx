import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { prefersReducedMotion } from '../motionPrefs'
import { FLARE_DEPART_SECONDS, getFlareTarget, handOffFlare, setLoaderFlareRunning, type FlareEntry } from '../flareHandoff'

const MIN_SHOW_MS = 2000
const FADE_MS = 700
/** Seconds of tapered trail behind the head. */
const TRAIL = 0.7
/** After the handoff the loader keeps drawing its receding trail + falling dust this long. */
const TAIL_S = 1.5

type Mote = { x: number; y: number; vx: number; vy: number; life: number; max: number; size: number; star: boolean; fill: string; twinkle: number }

// The flare canvas sits outside the fading overlay so it stays crisp while the
// background and mark fade; it is a fixed full-viewport layer above the overlay.
const FLARE_LAYER: CSSProperties = { position: 'fixed', inset: 0, zIndex: 101, width: '100%', height: '100%', pointerEvents: 'none' }

/**
 * Intro overlay: the Aura mark floats in the center while a fairy-like flare
 * loops around it on a figure-eight, shedding a trail of sparkling dust.
 * Stays up for at least MIN_SHOW_MS and until the window has loaded. Then the
 * overlay fades while the flare darts off to the top of the hero's A|B ribbon
 * and hands over to the hero's split sweep (see flareHandoff.ts), arriving with
 * the sweep's exact position, velocity and size.
 */
export function LoadingScreen() {
  const base = import.meta.env.BASE_URL || '/'
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const [phase, setPhase] = useState<'show' | 'fade' | 'done'>('show')
  const [flareFades, setFlareFades] = useState(false) // no hero to hand off to: the flare fades with the overlay
  const leaving = useRef(false)
  const flareActive = useRef(false)

  useEffect(() => {
    const reduced = prefersReducedMotion()
    if (!reduced) setLoaderFlareRunning(true)
    const started = performance.now()
    let timer = 0, safety = 0
    const finish = () => {
      const wait = Math.max(0, (reduced ? 600 : MIN_SHOW_MS) - (performance.now() - started))
      timer = window.setTimeout(() => {
        setPhase('fade')
        leaving.current = true
        // The flare loop ends the loader itself after its handoff; without one, plain fade.
        if (!flareActive.current) timer = window.setTimeout(() => setPhase('done'), FADE_MS)
        // Never hang (e.g. rAF paused in a background tab).
        safety = window.setTimeout(() => setPhase('done'), FADE_MS + (FLARE_DEPART_SECONDS + TAIL_S) * 1000 + 1500)
      }, wait)
    }
    if (document.readyState === 'complete') finish()
    else window.addEventListener('load', finish, { once: true })
    const prevOverflow = document.documentElement.style.overflow
    document.documentElement.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('load', finish)
      window.clearTimeout(timer)
      window.clearTimeout(safety)
      document.documentElement.style.overflow = prevOverflow
      handOffFlare(null) // no-op after a real handoff; otherwise the hero plays its own sweep
    }
  }, [])

  useEffect(() => {
    if (phase !== 'done') return
    document.documentElement.style.overflow = ''
    handOffFlare(null)
  }, [phase])

  useEffect(() => {
    const el = canvas.current
    if (!el || prefersReducedMotion()) return
    const ctx = el.getContext('2d')
    if (!ctx) return
    flareActive.current = true

    let w = 0, h = 0, dpr = 1
    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = el.clientWidth; h = el.clientHeight
      el.width = Math.round(w * dpr); el.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const motes: Mote[] = []
    let raf = 0
    let last = performance.now()
    const t0 = last
    // Departure state (seconds since t0).
    let leaveAt = -1, handAt = -1, ended = false, stopAt = -1
    const p0 = { x: 0, y: 0, vx: 0, vy: 0, depth: 0 }
    let tgt: FlareEntry | null = null

    // Soft halo, drawn at unit radius around the origin and scaled/translated per frame.
    const flareGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, 1)
    flareGrad.addColorStop(0, 'rgba(255, 250, 230, 1)')
    flareGrad.addColorStop(0.18, 'rgba(242, 204, 143, .95)')
    flareGrad.addColorStop(0.5, 'rgba(224, 168, 95, .35)')
    flareGrad.addColorStop(1, 'rgba(224, 122, 95, 0)')

    const star = (x: number, y: number, r: number) => {
      // four-point twinkle
      ctx.beginPath()
      ctx.moveTo(x, y - r); ctx.quadraticCurveTo(x, y, x + r, y)
      ctx.quadraticCurveTo(x, y, x, y + r); ctx.quadraticCurveTo(x, y, x - r, y)
      ctx.quadraticCurveTo(x, y, x, y - r)
      ctx.fill()
    }

    // Figure-eight around the mark with a slow wobble, like a fairy darting about.
    // Sampled by time (not per frame) so the trail stays smooth at any frame rate.
    const loop = (time: number) => {
      const cx = w / 2, cy = h / 2
      const R = Math.min(w, h) * 0.24
      const k = time * 1.6
      return {
        x: cx + Math.sin(k) * R * 1.15 + Math.sin(k * 3.1) * R * 0.06,
        y: cy + Math.sin(k * 2) * R * 0.42 + Math.cos(k * 1.3) * R * 0.08,
        depth: (Math.cos(k) + 1) / 2, // 1 = in front of the logo, 0 = behind it
      }
    }
    const headRadius = (depth: number) => 18 + depth * 16
    // Full path: the loop, then a cubic Hermite departure that lands on the hero's
    // sweep entry with the sweep's start velocity, then parked there (the hero has it).
    const at = (time: number) => {
      if (leaveAt < 0 || time <= leaveAt || !tgt) {
        const p = loop(time)
        return { ...p, r: headRadius(p.depth) }
      }
      const T = FLARE_DEPART_SECONDS
      const s = Math.min(1, (time - leaveAt) / T)
      const s2 = s * s, s3 = s2 * s
      const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2
      const e = s * s * (3 - 2 * s)
      return {
        x: h00 * p0.x + h10 * T * p0.vx + h01 * tgt.x + h11 * T * tgt.vx,
        y: h00 * p0.y + h10 * T * p0.vy + h01 * tgt.y + h11 * T * tgt.vy,
        depth: p0.depth + (1 - p0.depth) * e,
        r: headRadius(p0.depth) + (tgt.radius - headRadius(p0.depth)) * e,
      }
    }

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const t = (now - t0) / 1000

      // Start the departure once the overlay starts leaving (if the hero is there to take over).
      if (leaving.current && leaveAt < 0 && !ended) {
        tgt = getFlareTarget()
        if (tgt) {
          const a = loop(t), b = loop(t - 1 / 120), c = loop(t + 1 / 120)
          p0.x = a.x; p0.y = a.y; p0.depth = a.depth
          p0.vx = (c.x - b.x) * 60; p0.vy = (c.y - b.y) * 60
          leaveAt = t
        } else {
          ended = true
          stopAt = t + FADE_MS / 1000
          setFlareFades(true)
        }
      }
      if (leaveAt >= 0 && handAt < 0) {
        tgt = getFlareTarget() ?? tgt // follow the hero if it re-lays out mid-flight
        if (t - leaveAt >= FLARE_DEPART_SECONDS) {
          handAt = leaveAt + FLARE_DEPART_SECONDS
          handOffFlare(t0 + handAt * 1000) // rAF time of arrival: the hero sweep starts exactly there
        }
      }
      // Path time is clamped at the handoff: the head is the hero's from then on.
      const pathT = (time: number) => (handAt >= 0 ? Math.min(time, handAt) : time)
      const live = handAt < 0
      const head = at(pathT(t))

      // Dust: ~300 motes/s, spread along the stretch flown since the last frame.
      if (live) {
        const spawn = Math.round(dt * 300)
        for (let i = 0; i < spawn; i++) {
          const src = at(t - dt * Math.random())
          const a = Math.random() * Math.PI * 2
          const s = 8 + Math.random() * 22
          motes.push({
            x: src.x + (Math.random() - 0.5) * 6, y: src.y + (Math.random() - 0.5) * 6,
            vx: Math.cos(a) * s, vy: Math.sin(a) * s + 10,
            life: 0, max: 1.1 + Math.random() * 1.2,
            size: (1 + Math.random() * 2.2) * (0.6 + src.depth * 0.6),
            star: Math.random() < 0.18, fill: `hsl(${36 + Math.random() * 14} 85% 58%)`, twinkle: Math.random() * Math.PI * 2,
          })
        }
      }

      ctx.clearRect(0, 0, w, h)
      // After the handoff the loader's leftovers (receding trail, dust) fade out.
      const tailFade = handAt < 0 ? 1 : 1 - Math.min(1, Math.max(0, (t - handAt - 0.5) / (TAIL_S - 0.5)))

      // Ribbon: tapered golden streak along the last 0.7s of the flight path.
      // After the handoff it recedes into the hero's sweep entry.
      const STEPS = 40
      const tEnd = pathT(t), tStart = Math.min(tEnd, Math.max(t - TRAIL, 0))
      if (tEnd > tStart) {
        ctx.lineCap = 'round'
        ctx.strokeStyle = 'rgb(236, 190, 120)'
        let prev = at(tStart)
        for (let i = 1; i <= STEPS; i++) {
          const q = i / STEPS
          const tq = tStart + (tEnd - tStart) * q
          const cur = at(tq)
          const age = 1 - (t - tq) / TRAIL // 1 at the head, 0 at 0.7 s old
          ctx.globalAlpha = Math.max(0, age) * Math.max(0, age) * 0.55 * tailFade
          ctx.lineWidth = 1 + Math.max(0, age) * 5
          ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(cur.x, cur.y); ctx.stroke()
          prev = cur
        }
      }
      for (let i = motes.length - 1; i >= 0; i--) {
        const m = motes[i]
        m.life += dt
        if (m.life >= m.max) { motes[i] = motes[motes.length - 1]; motes.pop(); continue } // swap-remove (order is irrelevant)
        m.vx *= 0.97; m.vy = m.vy * 0.97 + 14 * dt // drift down like falling dust
        m.x += m.vx * dt; m.y += m.vy * dt
        const p = 1 - m.life / m.max
        const twinkle = 0.65 + 0.35 * Math.sin(now / 70 + m.twinkle)
        ctx.globalAlpha = p * twinkle * tailFade
        ctx.fillStyle = m.fill
        if (m.star) star(m.x, m.y, m.size * 2.6 * p + 1)
        else { ctx.beginPath(); ctx.arc(m.x, m.y, m.size * p, 0, Math.PI * 2); ctx.fill() }
      }

      // The flare itself: soft halo + bright core, smaller and dimmer when behind the mark.
      if (live) {
        const { x: hx, y: hy, depth, r } = head
        ctx.globalAlpha = 0.35 + depth * 0.65
        ctx.save()
        ctx.translate(hx, hy); ctx.scale(r, r)
        ctx.fillStyle = flareGrad
        ctx.beginPath(); ctx.arc(0, 0, 1, 0, Math.PI * 2); ctx.fill()
        ctx.restore()
        ctx.fillStyle = 'rgba(255, 252, 240, .95)'
        star(hx, hy, r * 0.35 + Math.sin(now / 90) * 1.5)
      }
      ctx.globalAlpha = 1

      if ((handAt >= 0 && t - handAt > TAIL_S) || (stopAt >= 0 && t > stopAt)) { setPhase('done'); return }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => { flareActive.current = false; cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])

  if (phase === 'done') return null
  return (
    <>
      <div className={phase === 'fade' ? 'fx-loader is-leaving' : 'fx-loader'} role="status" aria-label="Loading Aura">
        <img className="fx-loader__mark" src={`${base}brand/aura-mark.png`} alt="" aria-hidden="true" />
      </div>
      <canvas
        className="fx-loader__flare"
        ref={canvas}
        aria-hidden="true"
        style={{ ...FLARE_LAYER, opacity: flareFades ? 0 : 1, transition: `opacity ${FADE_MS}ms` }}
      />
    </>
  )
}
