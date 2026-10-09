/**
 * Robot flare: a small fairy flare that rides on a robot's motion.
 *
 * Tracking: every frame we measure the world-space speed of a few end effectors
 * (hands, feet, head), smooth each speed, and follow the fastest one. Switching
 * needs a clear margin (30 % faster + 0.15 m/s) and a short dwell, so the flare
 * doesn't flicker between joints; when it does switch, the head glides over on a
 * fast critically-damped follow, which reads as the fairy darting to the limb
 * that's moving. Brightness, trail length and sparkle emission all scale with
 * that limb's speed: a still robot leaves an almost invisible flare and no
 * sparkles; a fast gesture leaves a bright, tapered trail.
 *
 * Everything lives in world space (trail samples, sparkles) and is projected by
 * the camera, so the trail stays attached to the body when the camera moves.
 * It is ONE Points draw with a fixed pool and no per-frame allocation; it uses
 * the shared flare look (flareShared.ts) and belongs on AURA_OVERLAY_LAYER so
 * the ink pass never outlines it. Hidden (visible = false) while idle.
 */
import * as THREE from 'three'
import { FLARE_COLORS, FLARE_COLS, FLARE_KIND, GLOW_BLEND, flarePointFrag, uploadLiveRange } from './flareShared'

export type FlareEffector = { object: THREE.Object3D; offset?: THREE.Vector3 }
export type RobotFlareOptions = {
  /** World units per metre (1 for a real-scale G1, ~2.1 for the 2.7-unit stage robot). */
  scale?: number
  /** Overall strength 0..1; keep it subtle where motion quality is being judged. */
  strength?: number
  layer?: number
}

const TRAIL_SAMPLES = 48 // ring buffer of head positions (~0.8 s at 60 fps)
const TRAIL_BLOBS = 220
const SPARKS = 140
const MAX_POINTS = TRAIL_BLOBS + 1 + SPARKS

/** Picks the meshes/bones whose names match `re`, offset to their geometry's centre when available. */
export function effectorsByName(root: THREE.Object3D, re: RegExp): FlareEffector[] {
  const out: FlareEffector[] = []
  root.traverse(o => {
    if (!re.test(o.name)) return
    const mesh = o as THREE.Mesh
    let offset: THREE.Vector3 | undefined
    if (mesh.isMesh && mesh.geometry) {
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox()
      offset = mesh.geometry.boundingBox!.getCenter(new THREE.Vector3())
    }
    out.push({ object: o, offset })
  })
  return out
}

export class RobotFlare {
  readonly points: THREE.Points
  private geo = new THREE.BufferGeometry()
  private mat: THREE.ShaderMaterial
  private effectors: FlareEffector[] = []
  private scale: number
  private strength: number
  private attrs: THREE.BufferAttribute[]
  // Effector tracking.
  private pos: THREE.Vector3[] = []
  private prev: THREE.Vector3[] = []
  private speed = new Float32Array(0)
  private active = 0
  private dwell = 0
  private initialised = false
  private head = new THREE.Vector3()
  private headVel = new THREE.Vector3()
  private intensity = 0
  private time = 0
  // Trail ring buffer (world space).
  private trail = new Float32Array(TRAIL_SAMPLES * 3)
  private trailT = new Float64Array(TRAIL_SAMPLES)
  private trailHead = 0
  private trailCount = 0
  // Sparkles (world space).
  private sx = new Float32Array(SPARKS * 3)
  private sv = new Float32Array(SPARKS * 3)
  private slife = new Float32Array(SPARKS)
  private smax = new Float32Array(SPARKS)
  private ssize = new Float32Array(SPARKS)
  private skind = new Uint8Array(SPARKS)
  private scol = new Uint8Array(SPARKS)
  private sphase = new Float32Array(SPARKS)
  private sCount = 0
  private glintAcc = 0
  private dustAcc = 0
  // Scratch.
  private fresh = new Set<THREE.Object3D>() // ancestors already updated this frame (see refreshMatrices)
  private chain: THREE.Object3D[] = []
  private a = new THREE.Vector3()
  private b = new THREE.Vector3()
  private c = new THREE.Vector3()

  constructor(effectors: FlareEffector[], opts: RobotFlareOptions = {}) {
    this.scale = opts.scale ?? 1
    this.strength = opts.strength ?? 0.8
    this.geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3))
    this.geo.setAttribute('msize', new THREE.BufferAttribute(new Float32Array(MAX_POINTS), 1))
    this.geo.setAttribute('malpha', new THREE.BufferAttribute(new Float32Array(MAX_POINTS), 1))
    this.geo.setAttribute('mstar', new THREE.BufferAttribute(new Float32Array(MAX_POINTS), 1))
    this.geo.setAttribute('mcol', new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3))
    this.geo.setDrawRange(0, 0)
    this.attrs = ['position', 'msize', 'malpha', 'mstar', 'mcol'].map(n => this.geo.getAttribute(n) as THREE.BufferAttribute)
    this.mat = new THREE.ShaderMaterial({
      name: 'AuraRobotFlare',
      uniforms: { ...FLARE_COLS(), dpr: { value: 1 } },
      vertexShader: `attribute float msize;
attribute float malpha;
attribute float mstar;
attribute vec3 mcol;
uniform float dpr;
varying vec3 vCol;
varying float vA;
varying float vStar;
void main() {
  vCol = mcol; vA = malpha; vStar = mstar;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  gl_PointSize = msize * dpr;
}`,
      fragmentShader: flarePointFrag(0.9),
      ...GLOW_BLEND,
    })
    this.points = new THREE.Points(this.geo, this.mat)
    this.points.name = 'AuraRobotFlare'
    this.points.frustumCulled = false
    this.points.matrixAutoUpdate = false // world-space data: keep at identity under the scene root
    this.points.renderOrder = 960
    this.points.visible = false
    this.points.raycast = () => {}
    if (opts.layer !== undefined) this.points.layers.set(opts.layer)
    this.setEffectors(effectors)
  }

  setEffectors(effectors: FlareEffector[]) {
    this.effectors = effectors
    this.pos = effectors.map(() => new THREE.Vector3())
    this.prev = effectors.map(() => new THREE.Vector3())
    this.speed = new Float32Array(effectors.length)
    this.active = 0
    this.reset()
  }

  /** Forget motion history (new clip, teleport, loop wrap). */
  reset() {
    this.initialised = false
    this.trailCount = 0
    this.intensity = 0
    this.speed.fill(0)
  }

  /**
   * Advance and upload. Call after the robot's pose (and its parents) are set for
   * this frame and before rendering. `gain` dims the flare (e.g. a dimmed loser).
   * `matricesFresh`: the caller already ran updateMatrixWorld over the robot (and its
   * parents) after posing it this frame, so the effectors' matrixWorld is read as is;
   * otherwise the flare refreshes each effector's ancestor chain once (shared ancestors
   * are computed a single time).
   */
  update(dt: number, dpr: number, camera: THREE.Camera, viewHeightPx: number, gain = 1, matricesFresh = false) {
    if (!this.effectors.length || dt <= 0) return
    if (!matricesFresh) this.refreshMatrices()
    dt = Math.min(dt, 0.1)
    this.time += dt
    const n = this.effectors.length
    const S = this.scale
    let teleport = false
    for (let i = 0; i < n; i++) {
      const e = this.effectors[i]
      const p = this.pos[i]
      if (e.offset) p.copy(e.offset); else p.set(0, 0, 0)
      p.applyMatrix4(e.object.matrixWorld)
      if (!this.initialised) this.prev[i].copy(p)
      const raw = p.distanceTo(this.prev[i]) / dt / S // m/s
      if (raw > 9) teleport = true // clip swap / loop wrap: not motion
      else this.speed[i] += (raw - this.speed[i]) * (1 - Math.exp(-dt / 0.1))
      this.prev[i].copy(p)
    }
    if (!this.initialised || teleport) {
      this.initialised = true
      this.head.copy(this.pos[this.active])
      this.headVel.set(0, 0, 0)
      this.trailCount = 0
    }

    // Follow the fastest effector, with hysteresis.
    this.dwell += dt
    let best = this.active
    for (let i = 0; i < n; i++) if (this.speed[i] > this.speed[best]) best = i
    if (best !== this.active && this.dwell > 0.22 && this.speed[best] > this.speed[this.active] * 1.3 + 0.15) {
      this.active = best; this.dwell = 0
    }
    // Critically damped follow (smooth-damp): glides to a newly chosen limb, sticks to the current one.
    const omega = 2 / 0.06, x = omega * dt
    const ex = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x)
    const target3 = this.pos[this.active]
    this.a.copy(this.head).sub(target3) // change
    this.b.copy(this.headVel).addScaledVector(this.a, omega).multiplyScalar(dt) // temp
    this.headVel.addScaledVector(this.b, -omega).multiplyScalar(ex)
    this.head.copy(target3).add(this.a.add(this.b).multiplyScalar(ex))
    if (this.head.distanceTo(this.pos[this.active]) > 1.5 * S) this.head.copy(this.pos[this.active])

    // Intensity from the followed limb's speed: fast attack, slower release.
    const target = THREE.MathUtils.smoothstep(this.speed[this.active], 0.3, 1.7) * gain
    const k = target > this.intensity ? 1 - Math.exp(-dt / 0.08) : 1 - Math.exp(-dt / 0.45)
    this.intensity += (target - this.intensity) * k
    const I = this.intensity

    // Trail sample (world).
    this.trailHead = (this.trailHead + 1) % TRAIL_SAMPLES
    this.trail[this.trailHead * 3] = this.head.x; this.trail[this.trailHead * 3 + 1] = this.head.y; this.trail[this.trailHead * 3 + 2] = this.head.z
    this.trailT[this.trailHead] = this.time
    this.trailCount = Math.min(TRAIL_SAMPLES, this.trailCount + 1)

    // Emit sparkles from the head, carried a little by the limb's motion.
    this.glintAcc += dt * 30 * I * I
    this.dustAcc += dt * 90 * I
    for (; this.glintAcc >= 1; this.glintAcc--) this.spawn(1)
    for (; this.dustAcc >= 1; this.dustAcc--) this.spawn(0)

    // ---- upload
    const [pos, size, alpha, kind, col] = this.attrs
    let m = 0

    // Trail: soft blobs along the last (0.12 .. 0.6 s) of the head path, spaced by screen distance.
    const trailLen = 0.12 + 0.48 * I
    if (I > 0.01 && this.trailCount > 1) {
      const pxPerNdc = viewHeightPx / 2
      let idx = this.trailHead
      for (let s = 0; s < this.trailCount - 1 && m < TRAIL_BLOBS; s++) {
        const j = (idx - 1 + TRAIL_SAMPLES) % TRAIL_SAMPLES
        const age0 = this.time - this.trailT[idx], age1 = this.time - this.trailT[j]
        if (age0 > trailLen) break
        this.a.fromArray(this.trail, idx * 3)
        this.b.fromArray(this.trail, j * 3)
        // screen distance of this segment -> blob count
        this.c.copy(this.a).project(camera)
        const ax = this.c.x, ay = this.c.y
        this.c.copy(this.b).project(camera)
        const distPx = Math.hypot(this.c.x - ax, this.c.y - ay) * pxPerNdc
        const steps = Math.min(16, Math.max(1, Math.ceil(distPx / 2.5)))
        for (let t = 0; t < steps && m < TRAIL_BLOBS; t++) {
          const f = t / steps
          const age = age0 + (age1 - age0) * f
          const q = 1 - age / trailLen
          if (q <= 0) break
          this.c.copy(this.a).lerp(this.b, f)
          pos.setXYZ(m, this.c.x, this.c.y, this.c.z)
          size.setX(m, (2.5 + 9 * q) * (0.55 + 0.45 * I))
          alpha.setX(m, q * q * I * 0.75)
          kind.setX(m, FLARE_KIND.trail)
          col.setXYZ(m, 1, 1, 1)
          m++
        }
        idx = j
      }
    }

    // Head.
    if (I > 0.01) {
      pos.setXYZ(m, this.head.x, this.head.y, this.head.z)
      size.setX(m, (12 + 18 * I) * (1 + 0.06 * Math.sin(this.time * 11)))
      alpha.setX(m, I * 0.9)
      kind.setX(m, FLARE_KIND.head)
      col.setXYZ(m, 1, 1, 1)
      m++
    }

    // Sparkles: age, drift, twinkle (swap-remove).
    const damp = Math.pow(0.96, dt * 60)
    for (let i = 0; i < this.sCount;) {
      this.slife[i] += dt
      if (this.slife[i] >= this.smax[i]) { this.removeSpark(i); continue }
      const o = i * 3
      this.sv[o] *= damp; this.sv[o + 1] = this.sv[o + 1] * damp - 0.5 * S * dt; this.sv[o + 2] *= damp
      this.sx[o] += this.sv[o] * dt; this.sx[o + 1] += this.sv[o + 1] * dt; this.sx[o + 2] += this.sv[o + 2] * dt
      const l = this.slife[i] / this.smax[i]
      const env = THREE.MathUtils.smoothstep(l, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(l, 0.5, 1))
      const tw = 0.5 + 0.5 * Math.sin(this.sphase[i] + this.time * 14)
      const glint = this.skind[i] === 1
      pos.setXYZ(m, this.sx[o], this.sx[o + 1], this.sx[o + 2])
      size.setX(m, glint ? this.ssize[i] * (0.55 + 0.45 * env) * (0.8 + 0.2 * tw) : this.ssize[i])
      alpha.setX(m, glint ? env * (0.3 + 0.7 * tw) : env * (0.75 + 0.25 * tw))
      kind.setX(m, this.skind[i])
      const cc = FLARE_COLORS[this.scol[i]]
      col.setXYZ(m, cc.r, cc.g, cc.b)
      m++
      i++
    }

    uploadLiveRange(this.attrs, m)
    this.geo.setDrawRange(0, m)
    this.mat.uniforms.dpr.value = dpr
    this.points.visible = m > 0
  }

  /**
   * updateWorldMatrix(true, false) per effector would recompute the shared ancestors
   * (root, hips, spine, ...) once per effector. Walk each chain up to the first node
   * already refreshed this frame, then update top-down from there.
   */
  private refreshMatrices() {
    const fresh = this.fresh, chain = this.chain
    fresh.clear()
    for (const e of this.effectors) {
      chain.length = 0
      for (let o: THREE.Object3D | null = e.object; o && !fresh.has(o); o = o.parent) chain.push(o)
      for (let k = chain.length - 1; k >= 0; k--) { chain[k].updateWorldMatrix(false, false); fresh.add(chain[k]) }
    }
    chain.length = 0
  }

  private spawn(kind: 0 | 1) {
    if (this.sCount >= SPARKS) return
    const i = this.sCount++, o = i * 3, S = this.scale, r = Math.random
    const spread = 0.035 * S
    this.sx[o] = this.head.x + (r() - 0.5) * spread * 2
    this.sx[o + 1] = this.head.y + (r() - 0.5) * spread * 2
    this.sx[o + 2] = this.head.z + (r() - 0.5) * spread * 2
    // drift: a share of the limb's velocity plus a small random scatter
    const th = r() * Math.PI * 2, ph = Math.acos(2 * r() - 1), sp = (0.05 + r() * 0.2) * S
    this.sv[o] = this.headVel.x * 0.18 + Math.sin(ph) * Math.cos(th) * sp
    this.sv[o + 1] = this.headVel.y * 0.18 + Math.cos(ph) * sp
    this.sv[o + 2] = this.headVel.z * 0.18 + Math.sin(ph) * Math.sin(th) * sp
    this.slife[i] = 0
    this.smax[i] = kind ? 0.35 + r() * 0.5 : 0.6 + r() * 0.8
    this.ssize[i] = kind ? 5 + r() * 7 : 1.3 + r() * 1.5
    this.skind[i] = kind
    this.scol[i] = kind ? 1 + Math.floor(r() * 1.8) : 2 + Math.floor(r() * 1.7)
    this.sphase[i] = r() * Math.PI * 2
  }

  private removeSpark(i: number) {
    const j = --this.sCount
    if (i === j) return
    const o = i * 3, p = j * 3
    this.sx[o] = this.sx[p]; this.sx[o + 1] = this.sx[p + 1]; this.sx[o + 2] = this.sx[p + 2]
    this.sv[o] = this.sv[p]; this.sv[o + 1] = this.sv[p + 1]; this.sv[o + 2] = this.sv[p + 2]
    this.slife[i] = this.slife[j]; this.smax[i] = this.smax[j]; this.ssize[i] = this.ssize[j]
    this.skind[i] = this.skind[j]; this.scol[i] = this.scol[j]; this.sphase[i] = this.sphase[j]
  }

  dispose() {
    this.points.removeFromParent()
    this.geo.dispose()
    this.mat.dispose()
  }
}
