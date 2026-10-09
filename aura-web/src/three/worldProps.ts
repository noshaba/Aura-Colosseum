/**
 * Quiet fairy-world props for the Aura world (inspired by how the Abeto world's
 * plants and critters answer the player; our own shapes and shaders):
 *
 *  - grass + flower tufts: sway in the wind and part softly around the pointer
 *  - butterflies: wander near home, scatter when the pointer comes close, settle back
 *  - toadstool clusters (and the odd fairy ring): squish on hover / tap and puff a few glowing spores
 *  - fireflies (wisps): drift near the ground, glow, and drift curiously toward a
 *    resting pointer; a tap on empty ground answers with a small sparkle burst
 *
 * Solid props use the world's language: flat palette fills with a hard,
 * hue-shifted shade, written as world surfaces (alpha 0) so the composite inks
 * them, and the ground's per-half scheme colours (AW_SPLIT) so the hero's A | B
 * halves keep working. Glows (wisps, spores, tap sparkles) use the fairy flare's
 * point shader (flareShared.ts) on the overlay layer, so the world shares one
 * sparkle language with the flares.
 *
 * Cost: four draws (tufts, toadstools, butterflies, glow points), fixed pools,
 * no per-frame allocation, sway/flap in the vertex shader, and one ray vs the
 * ground plane per frame for the pointer (no mesh raycasts). Listeners are
 * passive and never preventDefault / stopPropagation; a "tap" is press + release
 * without dragging, so orbit drags never trigger it. Reduced motion: static
 * props, static faint wisps, no reactions.
 */
import * as THREE from 'three'
import { FLARE_COLORS, FLARE_COLS, FLARE_KIND, GLOW_BLEND, flarePointFrag, uploadLiveRange } from './flareShared'

export type WorldPropsOptions = {
  /** 0..1, scales prop counts (small viewers use fewer). Default 1. */
  density?: number
  /** Keep-out radius around the focus (stage / walking area), world units. */
  keepOut?: number
  /** Keep the taller props and the glows out of the camera-front sector (+z): the hero's fixed camera. */
  clearFront?: boolean
  /** Return true when the pointer (NDC) is over something the app owns (a robot, a button); props then ignore it. */
  pointerBlocked?: (ndcX: number, ndcY: number) => boolean
}

/** Things auraWorld hands over: shared shader chunks, the ground's (split-aware) uniforms, setup flags. */
export type WorldPropsContext = {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  group: THREE.Group
  overlayLayer: number
  scale: number
  groundY: number
  reduced: boolean
  direct: boolean
  glslCommon: string
  glslSplit: string
  /** Ground uniforms (uBase/uPatch/uInk/uAccent/uHaze and their *B twins, uSplitPts, uSplitCount, uFogDensity). Shared by reference. */
  ground: Record<string, THREE.IUniform>
  sunDir: THREE.Vector3
  palette: { terra: THREE.Color; sunset: THREE.Color; eggshell: THREE.Color; navy: THREE.Color }
  /** Optional sector to keep clear (the hero's sightline behind the robots). */
  sightline?: { yaw: number; halfAngle: number }
}

export type WorldProps = {
  readonly materials: THREE.ShaderMaterial[]
  /** Re-centre and re-scatter around (x, z) with a new keep-out radius. */
  layout(x: number, z: number, keepOut?: number): void
  update(camera: THREE.Camera, dt: number): void
  dispose(): void
}

const MAX_TUFTS = 260
const MAX_SHROOMS = 28
const MAX_FLIES = 8
const MAX_WISPS = 24
const MAX_SPARKS = 90
const MAX_GLOW = MAX_WISPS + MAX_SPARKS

/** Seeded PRNG (mulberry32), 0..1. Also seeds the world backdrop (auraWorld.ts). */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------- geometry
/** A tuft: 5 bent blades + 2 small flower diamonds. `kind`: 0 blade, 1 flower. */
function tuftGeometry() {
  const pos: number[] = [], kind: number[] = []
  const blade = (ang: number, lean: number, h: number, w: number) => {
    const c = Math.cos(ang), s = Math.sin(ang)
    const bx = -s * w, bz = c * w
    pos.push(-bx, 0, -bz, bx, 0, bz, c * lean, h, s * lean); kind.push(0, 0, 0)
  }
  const R = mulberry32(3)
  for (let i = 0; i < 5; i++) blade(i * 1.26 + R() * 0.5, 0.05 + R() * 0.06, 0.16 + R() * 0.12, 0.032)
  const flower = (x: number, y: number, z: number, r: number) => {
    pos.push(x - r, y, z, x + r, y, z, x, y + r * 1.4, z, x - r, y, z, x, y - r * 0.9, z, x + r, y, z); kind.push(1, 1, 1, 1, 1, 1)
    pos.push(x, y, z - r, x, y, z + r, x, y + r * 1.4, z, x, y, z - r, x, y - r * 0.9, z, x, y, z + r); kind.push(1, 1, 1, 1, 1, 1)
  }
  flower(0.03, 0.22, 0.01, 0.022)
  flower(-0.05, 0.17, -0.03, 0.018)
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('kind', new THREE.Float32BufferAttribute(kind, 1))
  return g
}

/** A toadstool (about 0.27 m tall at scale 1): stem + domed cap. `part`: 0 stem, 1 cap. */
function toadstoolGeometry() {
  const stem = new THREE.CylinderGeometry(0.035, 0.05, 0.2, 7, 1); stem.translate(0, 0.1, 0)
  const cap = new THREE.SphereGeometry(0.11, 9, 4, 0, Math.PI * 2, 0, Math.PI / 2); cap.scale(1, 0.75, 1); cap.translate(0, 0.19, 0)
  const flat = [stem, cap].map(geo => { const f = geo.toNonIndexed(); geo.dispose(); f.deleteAttribute('uv'); f.deleteAttribute('normal'); return f })
  const n0 = flat[0].getAttribute('position').count, n1 = flat[1].getAttribute('position').count
  const pos = new Float32Array((n0 + n1) * 3)
  pos.set(flat[0].getAttribute('position').array as Float32Array, 0)
  pos.set(flat[1].getAttribute('position').array as Float32Array, n0 * 3)
  const part = new Float32Array(n0 + n1); part.fill(1, n0)
  flat.forEach(geo => geo.dispose())
  const out = new THREE.BufferGeometry()
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  out.setAttribute('part', new THREE.BufferAttribute(part, 1))
  return out
}

/** Butterfly: a tiny body + two wing quads hinged on the body axis (z). `wing`: -1 left, +1 right, 0 body. */
function butterflyGeometry() {
  const pos: number[] = [], wing: number[] = []
  const quad = (side: number) => {
    const x1 = side * 0.09
    pos.push(0, 0, 0.03, x1, 0, 0.045, x1 * 0.85, 0, -0.005); wing.push(side, side, side)
    pos.push(0, 0, 0.0, x1 * 0.8, 0, -0.012, x1 * 0.45, 0, -0.05); wing.push(side, side, side)
  }
  quad(-1); quad(1)
  pos.push(-0.006, 0, 0.035, 0.006, 0, 0.035, 0, 0, -0.04); wing.push(0, 0, 0)
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('wing', new THREE.Float32BufferAttribute(wing, 1))
  return g
}

// ---------------------------------------------------------------------------- shaders
const SPLIT_UNIFORMS = `
  uniform vec3 uBase, uPatch, uInk, uAccent, uHaze;
  #ifdef AW_SPLIT
  uniform vec3 uBaseB, uPatchB, uInkB, uAccentB, uHazeB;
  #endif`
const SPLIT_PICK = `
  #ifdef AW_SPLIT
  float awSide = aw_side(gl_FragCoord.xy);
  vec3 sBase = AW_SIDE(uBase, uBaseB), sPatch = AW_SIDE(uPatch, uPatchB), sInk = AW_SIDE(uInk, uInkB);
  vec3 sAccent = AW_SIDE(uAccent, uAccentB), sHaze = AW_SIDE(uHaze, uHazeB);
  #else
  vec3 sBase = uBase, sPatch = uPatch, sInk = uInk, sAccent = uAccent, sHaze = uHaze;
  #endif`
/** Flat fill with a hard, hue-shifted shade on faces turned from the sun, then distance fog. */
const SHADE_FOG = `
  vec3 nrm = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  float lit = aw_step(0.05, dot(nrm, normalize(uSun)) * sign(dot(nrm, cameraPosition - vWorld)) + 0.25);
  col = mix(aw_shade(col, 0.75), col, lit);
  float dist = length(vWorld - cameraPosition);
  col = mix(col, sHaze, clamp(1.0 - exp(-pow(dist * uFogDensity, 2.0)), 0.0, 1.0));
  gl_FragColor = vec4(col, 0.0);
  #ifdef AW_DIRECT
  gl_FragColor.a = 1.0;
  #endif
  #include <colorspace_fragment>`

export function createWorldProps(ctx: WorldPropsContext, opts: WorldPropsOptions = {}): WorldProps {
  const S = ctx.scale
  const density = THREE.MathUtils.clamp(opts.density ?? 1, 0, 1)
  const nTufts = Math.round(MAX_TUFTS * density)
  const nShrooms = Math.max(6, Math.round(MAX_SHROOMS * density))
  const nFlies = Math.max(2, Math.round(MAX_FLIES * density))
  const nWisps = Math.max(6, Math.round(MAX_WISPS * density))
  let keepOut = opts.keepOut ?? 3 * S
  const g = ctx.ground

  const root = new THREE.Group()
  root.name = 'AuraWorldProps'
  root.position.y = ctx.groundY
  ctx.group.add(root)

  const uTime = { value: 0 }
  const uPointer = { value: new THREE.Vector3(1e5, 0, 1e5) } // ground point under the pointer (props-local)
  const uPush = { value: 0 }
  const uSun = { value: ctx.sunDir }
  const base = (extra: Record<string, THREE.IUniform>) => ({
    uBase: g.uBase, uPatch: g.uPatch, uInk: g.uInk, uAccent: g.uAccent, uHaze: g.uHaze,
    uBaseB: g.uBaseB, uPatchB: g.uPatchB, uInkB: g.uInkB, uAccentB: g.uAccentB, uHazeB: g.uHazeB,
    uSplitPts: g.uSplitPts, uSplitCount: g.uSplitCount, uFogDensity: g.uFogDensity,
    uSun, uTime, uScale: { value: S }, ...extra,
  })
  const defines = ctx.direct ? { AW_DIRECT: '' } : {}
  const pal = () => ({ uTerra: { value: ctx.palette.terra }, uSunset: { value: ctx.palette.sunset }, uEgg: { value: ctx.palette.eggshell } })

  // -------------------------------------------------------------- tufts
  const tuftGeo = tuftGeometry()
  const tuftPhase = new THREE.InstancedBufferAttribute(new Float32Array(MAX_TUFTS), 1)
  const tuftFlower = new THREE.InstancedBufferAttribute(new Float32Array(MAX_TUFTS), 1)
  tuftGeo.setAttribute('aPhase', tuftPhase)
  tuftGeo.setAttribute('aFlower', tuftFlower)
  const tuftMat = new THREE.ShaderMaterial({
    name: 'AuraPropsTufts',
    defines: { ...defines },
    uniforms: base({ uPointer, uPush, ...pal() }),
    vertexShader: /* glsl */ `
      attribute float kind;
      attribute float aPhase;
      attribute float aFlower;
      uniform float uTime, uPush, uScale;
      uniform vec3 uPointer;
      varying vec3 vWorld;
      varying float vKind, vFlower, vH;
      void main() {
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        float h = clamp(position.y / 0.26, 0.0, 1.0);
        float bend = h * h;
        float gust = sin(uTime * 1.4 + aPhase + wp.x * 0.35 / uScale) * 0.6 + sin(uTime * 2.3 + aPhase * 1.7 + wp.z * 0.5 / uScale) * 0.4;
        wp.x += gust * 0.035 * uScale * bend;
        wp.z += gust * 0.015 * uScale * bend;
        vec2 d = wp.xz - uPointer.xz;
        float r = length(d);
        float push = uPush * (1.0 - smoothstep(0.15 * uScale, 0.75 * uScale, r));
        wp.xz += (r > 1e-4 ? d / r : vec2(0.0)) * push * 0.13 * uScale * bend;
        wp.y -= push * 0.04 * uScale * bend;
        vWorld = wp.xyz; vKind = kind; vFlower = aFlower; vH = h;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      ${ctx.glslCommon}
      ${ctx.glslSplit}
      ${SPLIT_UNIFORMS}
      uniform vec3 uSun, uTerra, uSunset, uEgg;
      uniform float uFogDensity;
      varying vec3 vWorld;
      varying float vKind, vFlower, vH;
      void main() {
        ${SPLIT_PICK}
        vec3 col = mix(mix(sBase, sPatch, 0.55), mix(sPatch, sAccent, 0.12), vH);
        if (vKind > 0.5) {
          if (vFlower < 0.5) discard;
          col = vFlower < 1.5 ? uTerra : vFlower < 2.5 ? uSunset : uEgg;
        }
        ${SHADE_FOG}
      }`,
    side: THREE.DoubleSide,
    toneMapped: false,
    blending: THREE.NoBlending,
  })
  const tufts = new THREE.InstancedMesh(tuftGeo, tuftMat, MAX_TUFTS)
  tufts.name = 'AuraPropsTufts'
  tufts.count = nTufts
  root.add(tufts)

  // -------------------------------------------------------------- toadstools
  const shroomGeo = toadstoolGeometry()
  const shroomSquish = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SHROOMS), 1)
  shroomSquish.setUsage(THREE.DynamicDrawUsage)
  const shroomCap = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SHROOMS), 1)
  shroomGeo.setAttribute('aSquish', shroomSquish)
  shroomGeo.setAttribute('aCap', shroomCap)
  const shroomMat = new THREE.ShaderMaterial({
    name: 'AuraPropsToadstools',
    defines: { ...defines },
    uniforms: base({ ...pal() }),
    vertexShader: /* glsl */ `
      attribute float part;
      attribute float aSquish;
      attribute float aCap;
      varying vec3 vWorld, vLocal;
      varying float vPart, vCap;
      void main() {
        vec3 p = position;
        // squish: shorter and wider about the base (aSquish oscillates and decays on the CPU)
        p.y *= 1.0 - 0.3 * aSquish;
        p.xz *= 1.0 + 0.22 * aSquish * smoothstep(0.0, 0.2, position.y);
        vec4 wp = modelMatrix * instanceMatrix * vec4(p, 1.0);
        vWorld = wp.xyz; vLocal = position; vPart = part; vCap = aCap;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      ${ctx.glslCommon}
      ${ctx.glslSplit}
      ${SPLIT_UNIFORMS}
      uniform vec3 uSun, uTerra, uSunset, uEgg;
      uniform float uFogDensity;
      varying vec3 vWorld, vLocal;
      varying float vPart, vCap;
      void main() {
        ${SPLIT_PICK}
        vec3 col = mix(uEgg, sPatch, 0.25); // stem
        if (vPart > 0.5) {
          col = vCap < 0.5 ? uTerra : vCap < 1.5 ? uSunset : mix(uTerra, uSunset, 0.5);
          float spot = aw_step(0.78, aw_noise(vLocal * 38.0 + vCap * 3.1)); // eggshell spots
          col = mix(col, uEgg, spot);
        }
        ${SHADE_FOG}
      }`,
    toneMapped: false,
    blending: THREE.NoBlending,
  })
  const shrooms = new THREE.InstancedMesh(shroomGeo, shroomMat, MAX_SHROOMS)
  shrooms.name = 'AuraPropsToadstools'
  shrooms.count = nShrooms
  root.add(shrooms)

  // -------------------------------------------------------------- butterflies
  const flyGeo = butterflyGeometry()
  const flyPhase = new THREE.InstancedBufferAttribute(new Float32Array(MAX_FLIES), 1)
  const flyRate = new THREE.InstancedBufferAttribute(new Float32Array(MAX_FLIES), 1)
  const flyCol = new THREE.InstancedBufferAttribute(new Float32Array(MAX_FLIES), 1)
  flyGeo.setAttribute('aPhase', flyPhase)
  flyGeo.setAttribute('aRate', flyRate)
  flyGeo.setAttribute('aCol', flyCol)
  const flyMat = new THREE.ShaderMaterial({
    name: 'AuraPropsButterflies',
    defines: { ...defines },
    uniforms: base({ ...pal() }),
    vertexShader: /* glsl */ `
      attribute float wing;
      attribute float aPhase;
      attribute float aRate;
      attribute float aCol;
      uniform float uTime, uScale;
      varying vec3 vWorld;
      varying float vWing, vCol;
      void main() {
        vec3 p = position * uScale;
        float ang = wing * (0.25 + 1.05 * abs(sin(uTime * aRate + aPhase)));
        float c = cos(ang), s = sin(ang);
        p = vec3(p.x * c, abs(p.x) * s, p.z);
        vec4 wp = modelMatrix * instanceMatrix * vec4(p, 1.0);
        vWorld = wp.xyz; vWing = abs(wing); vCol = aCol;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      ${ctx.glslCommon}
      ${ctx.glslSplit}
      ${SPLIT_UNIFORMS}
      uniform vec3 uSun, uTerra, uSunset, uEgg;
      uniform float uFogDensity;
      varying vec3 vWorld;
      varying float vWing, vCol;
      void main() {
        ${SPLIT_PICK}
        vec3 col = vCol < 0.5 ? uTerra : vCol < 1.5 ? uSunset : uEgg;
        col = mix(sInk, col, step(0.5, vWing));
        ${SHADE_FOG}
      }`,
    side: THREE.DoubleSide,
    toneMapped: false,
    blending: THREE.NoBlending,
  })
  const flies = new THREE.InstancedMesh(flyGeo, flyMat, MAX_FLIES)
  flies.name = 'AuraPropsButterflies'
  flies.count = nFlies
  flies.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  root.add(flies)

  // -------------------------------------------------------------- glow points (wisps + spores + tap sparkles)
  const glowGeo = new THREE.BufferGeometry()
  glowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_GLOW * 3), 3))
  glowGeo.setAttribute('msize', new THREE.BufferAttribute(new Float32Array(MAX_GLOW), 1))
  glowGeo.setAttribute('malpha', new THREE.BufferAttribute(new Float32Array(MAX_GLOW), 1))
  glowGeo.setAttribute('mstar', new THREE.BufferAttribute(new Float32Array(MAX_GLOW), 1))
  glowGeo.setAttribute('mcol', new THREE.BufferAttribute(new Float32Array(MAX_GLOW * 3), 3))
  glowGeo.setDrawRange(0, 0)
  const glowAttrs = ['position', 'msize', 'malpha', 'mstar', 'mcol'].map(n => glowGeo.getAttribute(n) as THREE.BufferAttribute)
  const glowMat = new THREE.ShaderMaterial({
    name: 'AuraPropsGlow',
    uniforms: { ...FLARE_COLS(), dpr: { value: 1 } },
    vertexShader: /* glsl */ `
      attribute float msize;
      attribute float malpha;
      attribute float mstar;
      attribute vec3 mcol;
      uniform float dpr;
      varying vec3 vCol;
      varying float vA;
      varying float vStar;
      void main() {
        vCol = mcol; vA = malpha; vStar = mstar;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = msize * dpr;
      }`,
    fragmentShader: flarePointFrag(0.85),
    ...GLOW_BLEND,
  })
  const glow = new THREE.Points(glowGeo, glowMat)
  glow.name = 'AuraPropsGlow'
  glow.frustumCulled = false
  glow.matrixAutoUpdate = false // positions are written in world space
  glow.layers.set(ctx.overlayLayer) // after the ink pass, like the flares
  glow.renderOrder = 955
  glow.raycast = () => {}
  ctx.scene.add(glow) // a direct scene child, so the world's overlay pass picks it up

  for (const m of [tufts, shrooms, flies]) { m.frustumCulled = false; m.castShadow = false; m.receiveShadow = false; m.raycast = () => {} }

  // -------------------------------------------------------------- placement + state
  const dummy = new THREE.Object3D()
  const shroomPos = new Float32Array(MAX_SHROOMS * 3)
  const shroomAmp = new Float32Array(MAX_SHROOMS)
  const shroomT = new Float32Array(MAX_SHROOMS).fill(99)
  const flyHome = new Float32Array(MAX_FLIES * 3)
  const flyPos = new Float32Array(MAX_FLIES * 3)
  const flyFlee = new Float32Array(MAX_FLIES)
  const flyYaw = new Float32Array(MAX_FLIES)
  const wispHome = new Float32Array(MAX_WISPS * 3)
  const wispPos = new Float32Array(MAX_WISPS * 3)
  const wispPhase = new Float32Array(MAX_WISPS)
  const sx = new Float32Array(MAX_SPARKS * 3), sv = new Float32Array(MAX_SPARKS * 3)
  const slife = new Float32Array(MAX_SPARKS), smax = new Float32Array(MAX_SPARKS), ssize = new Float32Array(MAX_SPARKS)
  const skind = new Uint8Array(MAX_SPARKS), scol = new Uint8Array(MAX_SPARKS), sphase = new Float32Array(MAX_SPARKS)
  let sCount = 0

  const blocked = (ang: number, tall: boolean) => {
    const sl = ctx.sightline
    if (sl) {
      const d = Math.abs(Math.atan2(Math.sin(ang - sl.yaw), Math.cos(ang - sl.yaw)))
      if (d < sl.halfAngle + 0.12) return true
    }
    if (tall && opts.clearFront) {
      const d = Math.abs(Math.atan2(Math.sin(ang - Math.PI / 2), Math.cos(ang - Math.PI / 2)))
      if (d < 0.75) return true
    }
    return false
  }
  const ringPoint = (R: () => number, inner: number, band: number, tall: boolean, out: THREE.Vector3) => {
    for (let k = 0; k < 24; k++) {
      const ang = R() * Math.PI * 2
      if (blocked(ang, tall)) continue
      const r = inner + Math.pow(R(), 1.4) * band
      out.set(Math.cos(ang) * r, 0, Math.sin(ang) * r)
      return true
    }
    out.set(0, -100, 0)
    return false
  }

  const writeFlies = () => {
    for (let i = 0; i < nFlies; i++) {
      dummy.position.set(flyPos[i * 3], flyPos[i * 3 + 1], flyPos[i * 3 + 2])
      dummy.rotation.set(0, flyYaw[i], 0)
      dummy.scale.setScalar(1)
      dummy.updateMatrix()
      flies.setMatrixAt(i, dummy.matrix)
    }
    flies.instanceMatrix.needsUpdate = true
  }

  const spark = (x: number, y: number, z: number, vx: number, vy: number, vz: number, max: number, size: number, kind: 0 | 1, col: number) => {
    if (sCount >= MAX_SPARKS) return
    const i = sCount++, o = i * 3
    sx[o] = x; sx[o + 1] = y; sx[o + 2] = z; sv[o] = vx; sv[o + 1] = vy; sv[o + 2] = vz
    slife[i] = 0; smax[i] = max; ssize[i] = size; skind[i] = kind; scol[i] = col; sphase[i] = Math.random() * 6.28
  }
  const burst = (x: number, y: number, z: number, n: number, speed: number, glints: number) => {
    const r = Math.random
    for (let k = 0; k < n; k++) {
      const th = r() * Math.PI * 2, up = 0.4 + r() * 0.8, sp = (0.3 + r() * 0.7) * speed
      const glint = r() < glints
      spark(x, y, z, Math.cos(th) * sp, up * sp, Math.sin(th) * sp, glint ? 0.45 + r() * 0.5 : 0.7 + r() * 0.7,
        glint ? 6 + r() * 6 : 1.4 + r() * 1.4, glint ? 1 : 0, glint ? (r() < 0.5 ? 0 : 1) : 2 + Math.floor(r() * 1.7))
    }
  }

  /** Upload wisps + sparkles (world space). */
  const writeGlow = (dt: number) => {
    const [pos, size, alpha, kind, col] = glowAttrs
    const t = uTime.value
    const ox = root.position.x, oy = root.position.y, oz = root.position.z
    let m = 0
    for (let i = 0; i < nWisps; i++, m++) {
      const pulse = ctx.reduced ? 0.6 : 0.45 + 0.55 * Math.max(0, Math.sin(t * 1.3 + wispPhase[i] * 3.0))
      pos.setXYZ(m, wispPos[i * 3] + ox, wispPos[i * 3 + 1] + oy, wispPos[i * 3 + 2] + oz)
      size.setX(m, 9 + 5 * pulse)
      alpha.setX(m, 0.25 + 0.55 * pulse)
      kind.setX(m, FLARE_KIND.trail) // soft round glow with a hot core
      col.setXYZ(m, 1, 1, 1)
    }
    const damp = Math.pow(0.95, dt * 60)
    for (let i = 0; i < sCount;) {
      slife[i] += dt
      if (slife[i] >= smax[i]) {
        const j = --sCount, o = i * 3, p = j * 3
        sx[o] = sx[p]; sx[o + 1] = sx[p + 1]; sx[o + 2] = sx[p + 2]; sv[o] = sv[p]; sv[o + 1] = sv[p + 1]; sv[o + 2] = sv[p + 2]
        slife[i] = slife[j]; smax[i] = smax[j]; ssize[i] = ssize[j]; skind[i] = skind[j]; scol[i] = scol[j]; sphase[i] = sphase[j]
        continue
      }
      const o = i * 3
      sv[o] *= damp; sv[o + 1] = sv[o + 1] * damp - 0.35 * S * dt; sv[o + 2] *= damp
      sx[o] += sv[o] * dt; sx[o + 1] += sv[o + 1] * dt; sx[o + 2] += sv[o + 2] * dt
      const l = slife[i] / smax[i]
      const env = THREE.MathUtils.smoothstep(l, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(l, 0.5, 1))
      const tw = 0.5 + 0.5 * Math.sin(sphase[i] + t * 14)
      const glint = skind[i] === 1
      pos.setXYZ(m, sx[o] + ox, sx[o + 1] + oy, sx[o + 2] + oz)
      size.setX(m, glint ? ssize[i] * (0.55 + 0.45 * env) * (0.8 + 0.2 * tw) : ssize[i])
      alpha.setX(m, glint ? env * (0.3 + 0.7 * tw) : env * (0.75 + 0.25 * tw))
      kind.setX(m, skind[i])
      const c = FLARE_COLORS[scol[i]]
      col.setXYZ(m, c.r, c.g, c.b)
      m++; i++
    }
    uploadLiveRange(glowAttrs, m)
    glowGeo.setDrawRange(0, m)
    glowMat.uniforms.dpr.value = ctx.renderer.getPixelRatio()
    glow.visible = m > 0
  }

  const tmpV = new THREE.Vector3()
  const layout = (x: number, z: number, nextKeepOut?: number) => {
    if (nextKeepOut !== undefined) keepOut = nextKeepOut
    root.position.set(x, ctx.groundY, z)
    const R = mulberry32(11)
    // tufts in clumps
    const clumps = Math.max(6, Math.round(34 * density))
    let t = 0
    for (let c = 0; c < clumps && t < nTufts; c++) {
      ringPoint(R, keepOut + 0.2 * S, 7 * S, false, tmpV)
      const cx = tmpV.x, cz = tmpV.z
      const per = Math.ceil(nTufts / clumps)
      for (let i = 0; i < per && t < nTufts; i++, t++) {
        const a = R() * Math.PI * 2, rr = Math.sqrt(R()) * 0.55 * S
        dummy.position.set(cx + Math.cos(a) * rr, 0, cz + Math.sin(a) * rr)
        if (Math.hypot(dummy.position.x, dummy.position.z) < keepOut) dummy.position.y = -100
        dummy.rotation.set(0, R() * Math.PI * 2, 0)
        dummy.scale.setScalar(S * (0.75 + R() * 0.7))
        dummy.updateMatrix()
        tufts.setMatrixAt(t, dummy.matrix)
        tuftPhase.setX(t, R() * 6.28)
        const f = R()
        tuftFlower.setX(t, f < 0.55 ? 0 : f < 0.75 ? 1 : f < 0.9 ? 2 : 3)
      }
    }
    tufts.instanceMatrix.needsUpdate = true; tuftPhase.needsUpdate = true; tuftFlower.needsUpdate = true
    // toadstools in small clusters (3-4), sometimes a little fairy ring of 6
    let m = 0
    while (m < nShrooms) {
      ringPoint(R, keepOut + 0.5 * S, 5 * S, true, tmpV)
      const cx = tmpV.x, cz = tmpV.z, ring = R() < 0.25
      const n = ring ? 6 : 3 + Math.floor(R() * 2)
      for (let i = 0; i < n && m < nShrooms; i++, m++) {
        const a = ring ? (i / n) * Math.PI * 2 : R() * Math.PI * 2
        const rr = ring ? 0.32 * S : Math.sqrt(R()) * 0.22 * S
        dummy.position.set(cx + Math.cos(a) * rr, tmpV.y, cz + Math.sin(a) * rr)
        dummy.rotation.set((R() - 0.5) * 0.25, R() * 6.28, (R() - 0.5) * 0.25)
        dummy.scale.setScalar(S * (ring ? 0.6 + R() * 0.2 : 0.6 + R() * 0.8))
        dummy.updateMatrix()
        shrooms.setMatrixAt(m, dummy.matrix)
        shroomPos[m * 3] = dummy.position.x; shroomPos[m * 3 + 1] = dummy.position.y; shroomPos[m * 3 + 2] = dummy.position.z
        shroomCap.setX(m, Math.floor(R() * 2.99))
      }
    }
    shrooms.instanceMatrix.needsUpdate = true; shroomCap.needsUpdate = true
    for (let i = 0; i < nFlies; i++) {
      ringPoint(R, keepOut + 0.6 * S, 3.5 * S, true, tmpV)
      const h = (0.25 + R() * 0.55) * S
      flyHome[i * 3] = tmpV.x; flyHome[i * 3 + 1] = tmpV.y + h; flyHome[i * 3 + 2] = tmpV.z
      flyPos[i * 3] = tmpV.x; flyPos[i * 3 + 1] = tmpV.y + h; flyPos[i * 3 + 2] = tmpV.z
      flyFlee[i] = 0; flyYaw[i] = R() * 6.28
      flyPhase.setX(i, R() * 6.28); flyRate.setX(i, 9 + R() * 5); flyCol.setX(i, Math.floor(R() * 2.99))
    }
    flyPhase.needsUpdate = true; flyRate.needsUpdate = true; flyCol.needsUpdate = true
    for (let i = 0; i < nWisps; i++) {
      ringPoint(R, keepOut + 0.3 * S, 5 * S, true, tmpV)
      const h = (0.12 + R() * 0.6) * S
      wispHome[i * 3] = tmpV.x; wispHome[i * 3 + 1] = tmpV.y + h; wispHome[i * 3 + 2] = tmpV.z
      wispPos[i * 3] = tmpV.x; wispPos[i * 3 + 1] = tmpV.y + h; wispPos[i * 3 + 2] = tmpV.z
      wispPhase[i] = R() * 6.28
    }
    sCount = 0
    writeFlies()
    writeGlow(0)
  }

  // -------------------------------------------------------------- pointer
  const el = ctx.renderer.domElement
  const ndc = new THREE.Vector2()
  let pointerIn = false
  let moved = 99 // seconds since last pointer motion
  let downX = 0, downY = 0, downT = 0, downId = -1
  let tapPending = false
  const setNdc = (e: PointerEvent) => {
    const r = el.getBoundingClientRect()
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
  }
  const onMove = (e: PointerEvent) => { setNdc(e); pointerIn = true; moved = 0 }
  const onLeave = () => { pointerIn = false }
  const onDown = (e: PointerEvent) => { downX = e.clientX; downY = e.clientY; downT = performance.now(); downId = e.pointerId }
  const onUp = (e: PointerEvent) => {
    if (e.pointerId !== downId) return
    downId = -1
    if (performance.now() - downT < 350 && Math.hypot(e.clientX - downX, e.clientY - downY) < 8) { setNdc(e); tapPending = true }
  }
  if (!ctx.reduced) {
    el.addEventListener('pointermove', onMove, { passive: true })
    el.addEventListener('pointerleave', onLeave, { passive: true })
    el.addEventListener('pointerdown', onDown, { passive: true })
    el.addEventListener('pointerup', onUp, { passive: true })
  }

  const raycaster = new THREE.Raycaster()
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
  const hit = new THREE.Vector3()
  const local = new THREE.Vector3()
  let hasHit = false
  let restFor = 0

  const update = (camera: THREE.Camera, dt: number) => {
    if (ctx.reduced) return // static props and wisps (uploaded once in layout), no reactions
    dt = Math.min(dt, 0.1)
    uTime.value += dt
    moved += dt

    // ground point under the pointer (one ray vs the ground plane), ignored over the app's own targets
    hasHit = false
    if ((pointerIn && moved < 2.5) || tapPending) {
      if (!(opts.pointerBlocked && opts.pointerBlocked(ndc.x, ndc.y))) {
        raycaster.setFromCamera(ndc, camera)
        plane.constant = -ctx.groundY
        if (raycaster.ray.intersectPlane(plane, hit)) {
          local.copy(hit).sub(root.position)
          hasHit = Math.hypot(local.x, local.z) > keepOut * 0.9
        }
      }
    }
    const moving = hasHit && moved < 0.6
    restFor = hasHit && !moving ? restFor + dt : 0
    if (moving) {
      uPointer.value.lerp(local, uPush.value < 0.05 ? 1 : 1 - Math.exp(-dt * 18))
      uPush.value += (Math.min(1, 1 - moved / 0.6) - uPush.value) * (1 - Math.exp(-dt * 10))
    } else uPush.value *= Math.exp(-dt * 3)

    // tap on empty ground: a small sparkle answer + a push puff in the grass
    if (tapPending) {
      tapPending = false
      if (hasHit) {
        burst(local.x, 0.05 * S, local.z, 16, 0.9 * S, 0.45)
        uPointer.value.copy(local); uPush.value = 1
        for (let f = 0; f < nFlies; f++) {
          const dx = flyPos[f * 3] - local.x, dz = flyPos[f * 3 + 2] - local.z
          if (dx * dx + dz * dz < (1.6 * S) ** 2) flyFlee[f] = Math.max(flyFlee[f], 1.4)
        }
        for (let i = 0; i < nShrooms; i++) {
          const dx = shroomPos[i * 3] - local.x, dz = shroomPos[i * 3 + 2] - local.z
          if (dx * dx + dz * dz < (0.5 * S) ** 2) { shroomAmp[i] = 1; shroomT[i] = 0; burst(shroomPos[i * 3], 0.22 * S, shroomPos[i * 3 + 2], 7, 0.45 * S, 0.3) }
        }
      }
    }

    // toadstools: a moving pointer within reach squishes them and puffs a few spores
    for (let i = 0; i < nShrooms; i++) {
      if (moving && shroomT[i] > 0.6) {
        const dx = shroomPos[i * 3] - local.x, dz = shroomPos[i * 3 + 2] - local.z
        if (dx * dx + dz * dz < (0.3 * S) ** 2) {
          shroomAmp[i] = 0.8; shroomT[i] = 0
          burst(shroomPos[i * 3], 0.22 * S, shroomPos[i * 3 + 2], 5, 0.35 * S, 0.25)
        }
      }
      shroomT[i] += dt
      const k = shroomAmp[i] * Math.exp(-shroomT[i] * 5) * Math.cos(shroomT[i] * 16)
      shroomSquish.setX(i, shroomT[i] < 1.2 ? Math.max(-0.4, k) : 0)
    }
    shroomSquish.needsUpdate = true

    // butterflies: wander near home; scatter from the pointer; drift back
    for (let i = 0; i < nFlies; i++) {
      const o = i * 3
      const px = flyPos[o], py = flyPos[o + 1], pz = flyPos[o + 2]
      if (moving) {
        const dx = px - local.x, dz = pz - local.z
        if (dx * dx + dz * dz < (1.0 * S) ** 2) flyFlee[i] = Math.max(flyFlee[i], 1.2)
      }
      const t = uTime.value + i * 7.3
      let tx = flyHome[o] + Math.sin(t * 0.37) * 0.35 * S + Math.sin(t * 0.91) * 0.12 * S
      let ty = flyHome[o + 1] + Math.sin(t * 0.73) * 0.08 * S
      let tz = flyHome[o + 2] + Math.cos(t * 0.29) * 0.35 * S
      if (flyFlee[i] > 0) {
        flyFlee[i] -= dt
        const ax = px - local.x, az = pz - local.z, len = Math.hypot(ax, az) || 1
        tx = px + (ax / len) * 1.2 * S; tz = pz + (az / len) * 1.2 * S; ty = flyHome[o + 1] + 0.6 * S
      }
      const k = 1 - Math.exp(-dt * (flyFlee[i] > 0 ? 3.2 : 1.1))
      const nx = px + (tx - px) * k, nz = pz + (tz - pz) * k
      if (Math.hypot(nx - px, nz - pz) > 1e-4 * S) {
        const yaw = Math.atan2(nx - px, nz - pz)
        flyYaw[i] += Math.atan2(Math.sin(yaw - flyYaw[i]), Math.cos(yaw - flyYaw[i])) * (1 - Math.exp(-dt * 6))
      }
      flyPos[o] = nx; flyPos[o + 1] = py + (ty - py) * k; flyPos[o + 2] = nz
    }
    writeFlies()

    // wisps: slow drift around home; a resting pointer nearby draws the closest ones in, curious
    for (let i = 0; i < nWisps; i++) {
      const o = i * 3, t = uTime.value * 0.35 + wispPhase[i] * 5
      let tx = wispHome[o] + Math.sin(t) * 0.4 * S, ty = wispHome[o + 1] + Math.sin(t * 1.7) * 0.1 * S, tz = wispHome[o + 2] + Math.cos(t * 0.8) * 0.4 * S
      if (hasHit && restFor > 0.3) {
        const dx = local.x - wispPos[o], dz = local.z - wispPos[o + 2], d = Math.hypot(dx, dz)
        if (d < 1.6 * S && d > 0.2 * S) { tx = local.x - (dx / d) * 0.25 * S; tz = local.z - (dz / d) * 0.25 * S; ty = 0.25 * S }
      }
      const k = 1 - Math.exp(-dt * 0.9)
      wispPos[o] += (tx - wispPos[o]) * k; wispPos[o + 1] += (ty - wispPos[o + 1]) * k; wispPos[o + 2] += (tz - wispPos[o + 2]) * k
    }
    writeGlow(dt)
  }

  layout(0, 0)

  return {
    materials: [tuftMat, shroomMat, flyMat],
    layout,
    update,
    dispose() {
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerleave', onLeave)
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      root.removeFromParent()
      glow.removeFromParent()
      tuftGeo.dispose(); shroomGeo.dispose(); flyGeo.dispose(); glowGeo.dispose()
      tuftMat.dispose(); shroomMat.dispose(); flyMat.dispose(); glowMat.dispose()
      tufts.dispose(); shrooms.dispose(); flies.dispose()
    },
  }
}
