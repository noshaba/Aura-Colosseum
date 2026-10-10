/**
 * Aura model scene: the reward model's forward pass as one glowing diorama.
 *
 * Layout (world units): stages run left -> right along +X; inside every
 * sequence stage the 64 resampled frames run along Z (time) and the feature
 * width along Y, so frame i of the input skeleton ribbon, feature column i and
 * token column i all line up. Every count (frames, feature rows, latent rows,
 * blocks, heads, MLP widths) comes from the live model dims.
 *
 *   trajectory  64 skeleton slices of the motion (point cloud + key skeletons + joint trails)
 *   features    plain glass slab, a tall column of feature dots per frame (height ~ input_dim)
 *   projection  learned tapered glass prism, 413 -> 96 (height ~ d_model), token sheet out
 *   position    learned slab, a sunset wave sweeps along time over every later token
 *   transformer one learned slab per block; attention arcs between frames, one plane / colour per head
 *   pool        plain glass wedge funnelling the token sheet into one bright point
 *   head        learned plate with a d_model -> 64 -> 1 neuron fan
 *   reward      a toon-iridescent orb with a flare halo and sparkles; size / glow ~ reward percentile
 *
 * Draws: one glass mesh per slab (~8), one instanced glow-line mesh for every
 * static line (arcs, trails, fans), one for the playhead skeleton, one static
 * Points draw (tokens, joints, halos), one dynamic flare Points pool (flow
 * particles, sparkles, orb halo) and the orb. ~13 draw calls, no post pass.
 *
 * Learned stages get terra / sunset rims, deterministic ops stay eggshell glass.
 * The flare look (point sprite shader, colour stops, blend) is shared with the
 * site's fairy flares (flareShared.ts). No per-frame allocations; rendering
 * pauses off-screen / in hidden tabs; reduced motion renders a static frame on demand.
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { AURA_OVERLAY_LAYER, AURA_PALETTE, createAuraRenderer, createAuraWorld, type AuraWorld } from './auraWorld'
import { FLARE_COLORS, FLARE_COLS, FLARE_HEX, FLARE_KIND, GLOW_BLEND, flarePointFrag, premultiply, uploadLiveRange } from './flareShared'
import { mulberry32 } from './worldProps'
import { prefersReducedMotion } from '../motionPrefs'

export const AURA_MODEL_LAYER_IDS = ['trajectory', 'features', 'projection', 'position', 'transformer', 'pool', 'head', 'reward'] as const
export type AuraModelLayerId = typeof AURA_MODEL_LAYER_IDS[number]
export type AuraModelDims = { seq: number; input: number; model: number; heads: number; blocks: number; ff: number }
/** World-space G1 joints, positions[frame][joint][xyz] (y up, +z forward), optional parent list. */
export type AuraModelMotion = { positions: number[][][]; parents?: number[] }
export type AuraModelSceneMode = 'hero' | 'explorer'
export type AuraModelSceneOptions = {
  mode: AuraModelSceneMode
  dims: AuraModelDims
  /** Explorer: a stage was clicked. */
  onSelect?: (id: AuraModelLayerId) => void
  /** Optional overlay; its `[data-layer]` children are positioned under each stage every frame. */
  labels?: HTMLElement | null
}

const NSTAGE = AURA_MODEL_LAYER_IDS.length
const STAGE: Record<AuraModelLayerId, number> = { trajectory: 0, features: 1, projection: 2, position: 3, transformer: 4, pool: 5, head: 6, reward: 7 }
const LEARNED = [false, false, true, true, true, false, true, false]
const SPAN = 3.0 // time axis depth
const HEAD_HIDDEN = 64 // reward MLP hidden width (fixed in the model: d_model -> 64 -> 1)
const GROUND_Y = -1.25 // the meadow sits under the floating stages
const KEEP_OUT = 8.2 // fairy props stay outside this radius around the pipeline
/** The diorama is ~13 units long: scale the world (fog, mesas, props) to match, as the arena does for its stage. */
const WORLD_SCALE = 3.2
/** Result of a scored launch, drives the orb pop. */
export type AuraModelPop = 'win' | 'miss' | 'neutral'

/** Unitree G1 34-joint parents (Aura's g1-joints preview format). */
const G1_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32]
/** A neutral G1 standing pose (metres), the base of the procedural fallback motion. */
const G1_REST: readonly (readonly [number, number, number])[] = [
  [0.02, 0.79, 0.05], [0.08, 0.69, 0.06], [0.14, 0.66, 0.05], [0.14, 0.54, 0.1], [0.15, 0.35, 0.05], [0.16, 0.06, 0], [0.16, 0.04, 0], [0.18, 0.01, 0.14],
  [-0.04, 0.68, 0.06], [-0.1, 0.65, 0.05], [-0.11, 0.53, 0.09], [-0.12, 0.35, 0.04], [-0.13, 0.05, -0.02], [-0.14, 0.04, -0.02], [-0.16, 0, 0.11],
  [0.02, 0.79, 0.05], [0.02, 0.83, 0.04], [0.02, 0.83, 0.04],
  [0.12, 1.08, 0.07], [0.16, 1.07, 0.07], [0.19, 0.97, 0.07], [0.2, 0.89, 0.08], [0.2, 0.8, 0.11], [0.2, 0.76, 0.12], [0.21, 0.72, 0.14], [0.21, 0.62, 0.16],
  [-0.08, 1.08, 0.06], [-0.12, 1.08, 0.06], [-0.15, 0.98, 0.05], [-0.16, 0.9, 0.06], [-0.16, 0.8, 0.08], [-0.16, 0.76, 0.1], [-0.16, 0.72, 0.11], [-0.17, 0.62, 0.13],
]

const clamp = THREE.MathUtils.clamp
const col = (hex: string) => new THREE.Color(hex)
const PAL = {
  egg: col(AURA_PALETTE.eggshell), navy: col(AURA_PALETTE.navy), sunset: col(AURA_PALETTE.sunset),
  sage: col(AURA_PALETTE.sage), terra: col(AURA_PALETTE.terra), gold: col(FLARE_HEX.gold), hot: col(FLARE_HEX.hot),
}
const HEAD_COLORS = [PAL.sunset, PAL.terra, PAL.sage, PAL.egg, PAL.gold]
/** Sparkle tints: the four flare stops (0 hot .. 3 terra), then sage and eggshell for the miss poof. */
const SPARK_COLORS: readonly THREE.Color[] = [...FLARE_COLORS, PAL.sage.clone().lerp(PAL.egg, 0.35), PAL.egg]

/**
 * Stylized fallback motion (judge mode, no saved motion): a swinging, bobbing
 * walk-dance on the G1 joint layout, sagittal rotations about hip / knee /
 * shoulder / elbow pivots. Deterministic.
 */
export function proceduralG1Motion(frames = 96): AuraModelMotion {
  const out: number[][][] = []
  const rot = (p: number[], pivot: number[], a: number) => {
    const dy = p[1] - pivot[1], dz = p[2] - pivot[2], c = Math.cos(a), s = Math.sin(a)
    p[1] = pivot[1] + dy * c - dz * s
    p[2] = pivot[2] + dy * s + dz * c
  }
  const chain = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, k) => from + k)
  for (let f = 0; f < frames; f++) {
    const ph = (f / frames) * Math.PI * 4
    const pose = G1_REST.map(p => [p[0], p[1], p[2]])
    const swing = 0.5 * Math.sin(ph)
    const legs: [number[], number, number][] = [[chain(2, 7), 1, -swing], [chain(9, 14), 8, swing]]
    for (const [js, pivot, a] of legs) {
      const pv = pose[pivot].slice()
      for (const j of js) rot(pose[j], pv, a)
      const knee = pose[js[2]].slice()
      const bend = Math.max(0, Math.sin(ph + (a > 0 ? 0 : Math.PI) + 0.6)) * 0.9
      for (const j of js.slice(3)) rot(pose[j], knee, bend)
    }
    const arms: [number[], number, number][] = [[chain(19, 25), 18, swing * 0.9], [chain(27, 33), 26, -swing * 0.9]]
    for (const [js, pivot, a] of arms) {
      const pv = pose[pivot].slice()
      for (const j of js) rot(pose[j], pv, a - 0.15)
      const elbow = pose[js[2]].slice()
      for (const j of js.slice(3)) rot(pose[j], elbow, -0.5 - 0.35 * Math.sin(ph * 0.5 + (a > 0 ? 0 : 1)))
    }
    const bob = 0.035 * Math.cos(ph * 2), sway = 0.03 * Math.sin(ph)
    for (const p of pose) { p[1] += bob; p[0] += sway; p[2] += f * 0.012 }
    out.push(pose)
  }
  return { positions: out, parents: G1_PARENTS }
}

// ------------------------------------------------------------------ shaders
const PAL_GLSL = `uniform vec3 pEgg;
uniform vec3 pSun;
uniform vec3 pTerra;
uniform vec3 pSage;
vec3 auraPal( float h ) {
  h = fract( h ) * 4.0;
  return h < 1.0 ? mix( pEgg, pSun, h ) : h < 2.0 ? mix( pSun, pTerra, h - 1.0 ) : h < 3.0 ? mix( pTerra, pSage, h - 2.0 ) : mix( pSage, pEgg, h - 3.0 );
}
`
const palUniforms = () => ({ pEgg: { value: PAL.egg.clone() }, pSun: { value: PAL.sunset.clone() }, pTerra: { value: PAL.terra.clone() }, pSage: { value: PAL.sage.clone() } })

const POINT_VERT = `attribute vec3 aCol;
attribute vec4 aP; // size (world), alpha, flare kind, stage (-1 = always full)
attribute vec2 aQ; // time 0..1, fx (1 playhead, 2 playhead + position wave, 3 twinkle)
uniform float uGlow[ ${NSTAGE} ];
uniform float uTime;
uniform float uPlay;
uniform float uWave;
uniform float uPx;
uniform vec3 uWaveCol;
varying vec3 vCol;
varying float vA;
varying float vStar;
void main() {
  vec4 mv = modelViewMatrix * vec4( position, 1.0 );
  float g = aP.w < -0.5 ? 1.0 : uGlow[ int( aP.w + 0.5 ) ];
  float a = aP.y * g;
  float s = aP.x * ( 0.72 + 0.28 * g );
  vec3 c = aCol;
  if ( aQ.y > 0.5 && aQ.y < 2.5 ) {
    float hl = exp( -pow( ( aQ.x - uPlay ) * 24.0, 2.0 ) );
    if ( aQ.y > 1.5 ) {
      float w = 0.5 + 0.5 * sin( aQ.x * 18.85 - uTime * 1.7 );
      c = mix( c, uWaveCol, w * w * uWave );
      a *= 1.0 + w * uWave * 0.35;
    }
    c = mix( c, vec3( 1.0, 0.97, 0.88 ), hl * 0.55 );
    a *= 1.0 + hl * 1.3;
    s *= 1.0 + hl * 0.55;
  } else if ( aQ.y > 2.5 ) {
    a *= 0.45 + 0.55 * ( 0.5 + 0.5 * sin( uTime * 1.9 + aQ.x * 61.0 ) );
  }
  vCol = c; vA = a; vStar = aP.z;
  gl_Position = projectionMatrix * mv;
  gl_PointSize = s * uPx / max( -mv.z, 0.05 );
}`

const LINE_VERT = `attribute vec3 aA;
attribute vec3 aB;
attribute vec3 aCol;
attribute vec3 aM; // width (world), alpha, stage
attribute vec4 aK; // kind, phase, t0, t1
uniform float uGlow[ ${NSTAGE} ];
uniform float uArcs;
varying vec3 vCol;
varying float vSide;
varying float vT;
varying float vA;
varying float vKind;
varying float vPhase;
void main() {
  vec4 a = viewMatrix * vec4( aA, 1.0 );
  vec4 b = viewMatrix * vec4( aB, 1.0 );
  vec2 d = b.xy - a.xy;
  float len = length( d );
  vec2 dir = len > 1e-6 ? d / len : vec2( 1.0, 0.0 );
  vec4 p = mix( a, b, position.x );
  p.xy += vec2( -dir.y, dir.x ) * position.y * aM.x + dir * ( position.x * 2.0 - 1.0 ) * aM.x * 0.6;
  float g = aM.z < -0.5 ? 1.0 : uGlow[ int( aM.z + 0.5 ) ];
  vA = aM.y * g * ( aK.x > 0.5 && aK.x < 1.5 ? uArcs : 1.0 );
  vCol = aCol; vSide = position.y; vT = mix( aK.z, aK.w, position.x ); vKind = aK.x; vPhase = aK.y;
  gl_Position = projectionMatrix * p;
}`

const LINE_FRAG = `uniform float uTime;
uniform float uPlay;
uniform vec3 cHot;
varying vec3 vCol;
varying float vSide;
varying float vT;
varying float vA;
varying float vKind;
varying float vPhase;
void main() {
  float core = exp( -vSide * vSide * 3.2 );
  float hot = exp( -vSide * vSide * 14.0 );
  float a = vA * core;
  vec3 c = vCol;
  if ( vKind > 0.5 && vKind < 1.5 ) {
    // attention arc: a light packet runs key -> query; each arc breathes in and out
    float head = fract( uTime * 0.42 + vPhase );
    float pulse = exp( -pow( ( vT - head ) * 6.5, 2.0 ) );
    float cyc = 0.5 + 0.5 * sin( uTime * 0.75 + vPhase * 6.2832 );
    cyc = smoothstep( 0.15, 0.85, cyc );
    a *= ( 0.16 + 1.1 * pulse ) * ( 0.25 + 0.75 * cyc );
    c = mix( c, cHot, pulse * hot * 0.85 );
  } else if ( vKind > 1.5 && vKind < 2.5 ) {
    c = mix( c, cHot, hot * 0.7 );
  } else if ( vKind > 2.5 && vKind < 3.5 ) {
    float hl = exp( -pow( ( vT - uPlay ) * 11.0, 2.0 ) );
    a *= 0.3 + 1.1 * hl;
    c = mix( c, cHot, hl * hot * 0.6 );
  } else if ( vKind > 3.5 ) {
    float p = fract( vT - uTime * 0.55 + vPhase );
    float pulse = smoothstep( 0.72, 0.97, p ) * ( 1.0 - smoothstep( 0.97, 1.0, p ) );
    a *= 0.32 + 1.2 * pulse;
    c = mix( c, cHot, pulse * hot * 0.7 );
  }
  if ( a < 0.003 ) discard;
  gl_FragColor = vec4( c, clamp( a, 0.0, 1.0 ) );
  #include <colorspace_fragment>
  ${premultiply(0.7)}
}`

const GLASS_VERT = `attribute vec3 aBox;
varying vec3 vN;
varying vec3 vV;
varying vec3 vBox;
void main() {
  vec4 mv = modelViewMatrix * vec4( position, 1.0 );
  vN = normalize( normalMatrix * normal );
  vV = -mv.xyz;
  vBox = aBox;
  gl_Position = projectionMatrix * mv;
}`

const GLASS_FRAG = `${PAL_GLSL}
uniform vec3 uTint;
uniform vec3 uRim;
uniform vec3 uHalf;
uniform vec3 cHot;
uniform float uR;
uniform float uGlow;
uniform float uTime;
uniform float uSeed;
uniform float uLearned;
varying vec3 vN;
varying vec3 vV;
varying vec3 vBox;
void main() {
  vec3 N = normalize( vN );
  if ( !gl_FrontFacing ) N = -N;
  vec3 V = normalize( vV );
  float f = 1.0 - abs( dot( N, V ) );
  // toon: fresnel posterized into two anti-aliased bands
  float b1 = smoothstep( 0.30, 0.36, f );
  float b2 = smoothstep( 0.66, 0.72, f );
  // inked frame along the slab edges (second-nearest face distance)
  vec3 d = uHalf - abs( vBox );
  float mn = min( d.x, min( d.y, d.z ) ), mx = max( d.x, max( d.y, d.z ) );
  float second = d.x + d.y + d.z - mn - mx;
  float edge = 1.0 - smoothstep( uR * 0.35, uR * 1.15, second );
  // iridescent film: palette hue walks with view angle and position
  float h = f * 1.15 + dot( vBox, vec3( 0.18, 0.42, 0.11 ) ) + uTime * 0.035 + uSeed;
  vec3 irid = auraPal( h );
  // a soft sheen sweeps diagonally across the glass now and then
  vec3 q = vBox / uHalf;
  float sp = ( q.y * 0.7 + q.z * 0.3 + q.x * 0.2 ) * 0.5 + 0.5;
  float sw = fract( uTime * 0.07 + uSeed ) * 2.2 - 0.6;
  float sheen = exp( -pow( ( sp - sw ) * 7.0, 2.0 ) );
  // Toon glass for the world composite: flat palette fill, iridescent fresnel band, hard rim
  // band, rim-coloured frame (learned = terra/sunset, op = eggshell) and a passing sheen.
  // Written premultiplied at a fixed alpha of 0.5, which the composite reads as a "painted
  // character": exact palette, no tone mapping, solid ink silhouettes.
  vec3 c = mix( uTint, irid, 0.22 + 0.4 * b1 );
  c = mix( c, uRim, 0.55 * b2 + ( 0.75 + 0.2 * uLearned ) * edge );
  c += cHot * sheen * ( 0.18 + 0.3 * uGlow );
  c *= mix( 0.66, 1.1, clamp( uGlow, 0.0, 1.0 ) ) + 0.3 * max( uGlow - 1.0, 0.0 );
  gl_FragColor = vec4( c * 0.5, 0.5 );
}`

const ORB_VERT = `varying vec3 vN;
varying vec3 vV;
varying vec3 vP;
void main() {
  vec4 mv = modelViewMatrix * vec4( position, 1.0 );
  vN = normalize( normalMatrix * normal );
  vV = -mv.xyz;
  vP = position;
  gl_Position = projectionMatrix * mv;
}`

const ORB_FRAG = `${PAL_GLSL}
uniform vec3 uCore;
uniform vec3 uShade;
uniform vec3 cHot;
uniform float uTime;
uniform float uLevel;
uniform float uGlow;
varying vec3 vN;
varying vec3 vV;
varying vec3 vP;
void main() {
  vec3 N = normalize( vN );
  vec3 V = normalize( vV );
  float ndv = max( dot( N, V ), 0.0 );
  float f = 1.0 - ndv;
  // hard two-tone key light (toon), from the upper left in view space
  float key = dot( N, normalize( vec3( -0.45, 0.65, 0.6 ) ) );
  float lit = smoothstep( 0.05, 0.12, key );
  vec3 base = mix( uShade, uCore, lit );
  // painted highlight blob
  float spec = smoothstep( 0.86, 0.9, dot( N, normalize( vec3( -0.35, 0.55, 0.75 ) ) ) );
  // iridescent swirl
  float h = vP.y * 0.9 + atan( vP.z, vP.x ) * 0.32 + uTime * 0.12 + f * 0.8;
  vec3 irid = auraPal( h );
  vec3 c = mix( base, irid, 0.22 + 0.3 * smoothstep( 0.35, 0.75, f ) );
  c = mix( c, cHot, spec * 0.9 );
  float rim = smoothstep( 0.62, 0.68, f );
  c = mix( c, mix( pEgg, cHot, uLevel ), rim * 0.85 );
  c *= 0.84 + 0.26 * min( uGlow, 1.4 );
  gl_FragColor = vec4( c, 0.5 ); // painted character tag for the world composite
}`

// ------------------------------------------------------------------ buffers
class PointBuf {
  pos: number[] = []; col: number[] = []; p: number[] = []; q: number[] = []
  add(x: number, y: number, z: number, c: THREE.Color, size: number, alpha: number, kind: number, stage: number, t = 0, fx = 0) {
    this.pos.push(x, y, z); this.col.push(c.r, c.g, c.b); this.p.push(size, alpha, kind, stage); this.q.push(t, fx)
  }
  geometry() {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3))
    g.setAttribute('aCol', new THREE.Float32BufferAttribute(this.col, 3))
    g.setAttribute('aP', new THREE.Float32BufferAttribute(this.p, 4))
    g.setAttribute('aQ', new THREE.Float32BufferAttribute(this.q, 2))
    return g
  }
}

const QUAD = [0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]
const LINE_KIND = { plain: 0, arc: 1, bone: 2, trail: 3, flow: 4 } as const

class LineBuf {
  a: number[] = []; b: number[] = []; c: number[] = []; m: number[] = []; k: number[] = []
  add(ax: number, ay: number, az: number, bx: number, by: number, bz: number, c: THREE.Color, width: number, alpha: number, stage: number, kind: number, phase = 0, t0 = 0, t1 = 1) {
    this.a.push(ax, ay, az); this.b.push(bx, by, bz); this.c.push(c.r, c.g, c.b); this.m.push(width, alpha, stage); this.k.push(kind, phase, t0, t1)
  }
  get count() { return this.m.length / 3 }
  geometry(capacity = this.count, dynamic = false) {
    const g = new THREE.InstancedBufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(QUAD, 3))
    g.setIndex([0, 1, 2, 0, 2, 3])
    const attr = (src: number[], size: number) => {
      const arr = new Float32Array(Math.max(1, capacity) * size)
      arr.set(src.slice(0, arr.length))
      const at = new THREE.InstancedBufferAttribute(arr, size)
      if (dynamic) at.setUsage(THREE.DynamicDrawUsage)
      return at
    }
    g.setAttribute('aA', attr(this.a, 3)); g.setAttribute('aB', attr(this.b, 3)); g.setAttribute('aCol', attr(this.c, 3))
    g.setAttribute('aM', attr(this.m, 3)); g.setAttribute('aK', attr(this.k, 4))
    g.instanceCount = this.count
    return g
  }
}

type Stage = { id: AuraModelLayerId; cx: number; cy: number; box: THREE.Box3; label: THREE.Vector3 }
type Knot = { x: number; h: number; d: number }
type Built = {
  group: THREE.Group
  geos: THREE.BufferGeometry[]
  mats: THREE.Material[]
  glass: { mat: THREE.ShaderMaterial; stage: number; mesh: THREE.Mesh }[]
  stages: Stage[]
  knots: Knot[]
  xStart: number
  xEnd: number
  orbPos: THREE.Vector3
  bounds: THREE.Box3
  /** resampled, scene-space playhead skeleton frames: seq * (34 + 1 head) * 3 */
  skel: Float32Array
  skelJoints: number
  bones: [number, number][]
  trajCx: number
}

const FLOW_MAX = { hero: 150, explorer: 240 }
const SPARK_MAX = 140

export class AuraModelScene {
  readonly mode: AuraModelSceneMode
  private container: HTMLElement
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(26, 1, 0.1, 200)
  private clock = new THREE.Clock(false)
  private reduced = prefersReducedMotion()
  private dims: AuraModelDims
  private motion: AuraModelMotion | null = null
  private fallback: AuraModelMotion | null = null
  private reward: number | null = null
  private onSelect?: (id: AuraModelLayerId) => void
  private labels: HTMLElement | null
  private labelEls: { el: HTMLElement; stage: number; x: number; y: number; on: string }[] = []

  private active = STAGE.transformer
  private hovered = -1
  private playing = false
  private world: AuraWorld
  private pulse = new Float32Array(NSTAGE) // stage juice, 1 on arrival, decays
  private popT = 9 // seconds since the last orb pop
  private orbCss = { x: NaN, y: NaN }
  private heroFocus = -1 // hero: stage the camera zooms to (-1 = overview)
  private safeTop = 0 // fractions of the canvas height kept clear (hero text on top, stepper below)
  private safeBottom = 0
  private built: Built | null = null
  private disposed = false
  private raf = 0
  private inView = true
  private pageVisible = true
  private io: IntersectionObserver | null = null
  private ro: ResizeObserver | null = null
  private time = 0
  private width = 1
  private height = 1

  // Shared uniforms.
  private glow = new Float32Array(NSTAGE).fill(0.6)
  private glowTarget = new Float32Array(NSTAGE).fill(0.6)
  private uTime = { value: 0 }
  private uPlay = { value: 0 }
  private uWave = { value: 0.35 }
  private uArcs = { value: 0.7 }
  private uPx = { value: 500 }
  private uGlow = { value: this.glow }

  private pointMat: THREE.ShaderMaterial
  private lineMat: THREE.ShaderMaterial
  private orbMat: THREE.ShaderMaterial
  private orbGeo = new THREE.SphereGeometry(1, 48, 32)
  private orb: THREE.Mesh
  private orbLevel = 0.5
  private orbScale = 0.3

  // Playhead skeleton (dynamic lines) and flare pool (dynamic points).
  private boneGeo: THREE.InstancedBufferGeometry
  private boneMesh: THREE.Mesh
  private dynGeo = new THREE.BufferGeometry()
  private dynPts: THREE.Points
  private dynAttrs: THREE.BufferAttribute[]
  private flowN: number
  private fs: Float32Array // progress 0..1 along the pipeline
  private fu: Float32Array // time lane -1..1
  private fv: Float32Array // feature lane -1..1
  private fspd: Float32Array
  private flag: Float32Array // lag inside the forward-pass packet (0..1), < 0 = free flow
  private fkind: Uint8Array
  private fcol: Uint8Array
  private packetS = 0
  private passStage = -1
  // Sparkles.
  private sx = new Float32Array(SPARK_MAX * 3)
  private sv = new Float32Array(SPARK_MAX * 3)
  private slife = new Float32Array(SPARK_MAX)
  private smax = new Float32Array(SPARK_MAX)
  private ssize = new Float32Array(SPARK_MAX)
  private skind = new Uint8Array(SPARK_MAX)
  private scol = new Uint8Array(SPARK_MAX)
  private sCount = 0
  private sparkAcc = 0
  private rand = mulberry32(7)

  // Camera state.
  private camGoal = new THREE.Vector3()
  private camDistGoal = 20
  private camSnap = true
  /**
   * One camera controller. Channels: target x/y/z, distance, yaw, pitch, projection shift (safe band).
   * A selection change starts a cubic Hermite glide from the current pose AND velocity toward the live
   * goal (ease-in-out when starting at rest; interrupts retarget without a jump), plus a small arc.
   */
  private cam = {
    cur: new Float64Array(7), from: new Float64Array(7), v0: new Float64Array(7), goal: new Float64Array(7),
    t: 1, dur: 1, lift: 0, swing: 0,
    focus: -99, passing: false, safeT: -1, safeB: -1, build: -1,
  }
  private buildCount = 0
  private focusScale = new Float32Array(NSTAGE).fill(1)
  private pointer = new THREE.Vector2()
  private parallax = new THREE.Vector2()
  // Scratch.
  private v = new THREE.Vector3()
  private v2 = new THREE.Vector3()
  private ndc = new THREE.Vector2()
  private fitCam = new THREE.PerspectiveCamera()
  private fitBox = new THREE.Box3()
  private fitTarget = new THREE.Vector3()
  private fitDir = new THREE.Vector3()
  private ray = new THREE.Raycaster()

  constructor(container: HTMLElement, opts: AuraModelSceneOptions) {
    this.container = container
    this.mode = opts.mode
    this.dims = { ...opts.dims }
    this.onSelect = opts.onSelect
    this.labels = opts.labels ?? null
    this.flowN = FLOW_MAX[opts.mode]

    // The Aura world: sky, mesas, meadow and the toon/ink composite (same as the arena and viewers).
    this.renderer = createAuraRenderer({ powerPreference: 'high-performance' })
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.06
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.mode === 'hero' ? 1.5 : 2))
    const canvas = this.renderer.domElement
    canvas.setAttribute('aria-hidden', 'true')
    canvas.className = 'aura-model-viz__canvas'
    container.prepend(canvas)

    // The camera looks from -x/+z toward +x/-z: keep the near mesa ring out of the pipeline's backdrop.
    const back = Math.atan2(-Math.cos(-0.5), -Math.sin(-0.5))
    this.world = createAuraWorld(this.scene, this.renderer, {
      scale: WORLD_SCALE, groundY: GROUND_Y, ring: false, msaa: 'auto', reducedMotion: this.reduced,
      sightline: { yaw: back, halfAngle: THREE.MathUtils.degToRad(32) },
      props: { density: this.mode === 'hero' ? 0.45 : 0.7, keepOut: KEEP_OUT, pointerBlocked: (x, y) => this.pickNdc(x, y) >= 0 },
    })
    this.world.setFocus(0, 0, KEEP_OUT)
    this.camera.far = 240
    const sharedGlow = { uGlow: this.uGlow, uTime: this.uTime, uPlay: this.uPlay }
    this.pointMat = new THREE.ShaderMaterial({
      name: 'AuraModelPoints',
      uniforms: { ...FLARE_COLS(), ...sharedGlow, uWave: this.uWave, uPx: this.uPx, uWaveCol: { value: PAL.sunset.clone().lerp(PAL.hot, 0.2) } },
      vertexShader: POINT_VERT,
      fragmentShader: flarePointFrag(0.72),
      ...GLOW_BLEND,
      depthTest: true,
    })
    this.lineMat = new THREE.ShaderMaterial({
      name: 'AuraModelLines',
      uniforms: { ...sharedGlow, uArcs: this.uArcs, cHot: { value: PAL.hot.clone() } },
      vertexShader: LINE_VERT,
      fragmentShader: LINE_FRAG,
      ...GLOW_BLEND,
      depthTest: true,
    })
    this.orbMat = new THREE.ShaderMaterial({
      name: 'AuraModelOrb',
      uniforms: {
        ...palUniforms(), uCore: { value: new THREE.Color() }, uShade: { value: new THREE.Color() }, cHot: { value: PAL.hot.clone() },
        uTime: this.uTime, uLevel: { value: 0.5 }, uGlow: { value: 0.6 },
      },
      vertexShader: ORB_VERT,
      fragmentShader: ORB_FRAG,
    })
    this.orb = new THREE.Mesh(this.orbGeo, this.orbMat)
    this.orb.renderOrder = 1
    this.scene.add(this.orb)

    // Playhead skeleton: up to 40 bones, rewritten every frame.
    this.boneGeo = new LineBuf().geometry(40, true)
    this.boneGeo.instanceCount = 0
    this.boneMesh = new THREE.Mesh(this.boneGeo, this.lineMat)
    this.boneMesh.frustumCulled = false
    this.boneMesh.renderOrder = 4
    this.boneMesh.layers.set(AURA_OVERLAY_LAYER)
    this.scene.add(this.boneMesh)

    // Flare pool: flow particles + sparkles + orb halo + playhead joints.
    const maxDyn = this.flowN + SPARK_MAX + 4 + 40 + 8
    this.dynGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxDyn * 3), 3))
    this.dynGeo.setAttribute('aCol', new THREE.BufferAttribute(new Float32Array(maxDyn * 3), 3))
    this.dynGeo.setAttribute('aP', new THREE.BufferAttribute(new Float32Array(maxDyn * 4), 4))
    this.dynGeo.setAttribute('aQ', new THREE.BufferAttribute(new Float32Array(maxDyn * 2), 2))
    this.dynAttrs = ['position', 'aCol', 'aP', 'aQ'].map(n => {
      const a = this.dynGeo.getAttribute(n) as THREE.BufferAttribute
      a.setUsage(THREE.DynamicDrawUsage)
      return a
    })
    this.dynGeo.setDrawRange(0, 0)
    this.dynPts = new THREE.Points(this.dynGeo, this.pointMat)
    this.dynPts.frustumCulled = false
    this.dynPts.renderOrder = 6
    this.dynPts.layers.set(AURA_OVERLAY_LAYER) // glows are drawn after the ink composite, like the arena's flares
    this.scene.add(this.dynPts)

    const n = this.flowN
    this.fs = new Float32Array(n); this.fu = new Float32Array(n); this.fv = new Float32Array(n); this.fspd = new Float32Array(n)
    this.flag = new Float32Array(n).fill(-1); this.fkind = new Uint8Array(n); this.fcol = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      const r = this.rand
      this.fs[i] = r(); this.fu[i] = r() * 2 - 1; this.fv[i] = r() * 2 - 1
      this.fspd[i] = 0.045 + r() * 0.03
      this.fkind[i] = r() < 0.16 ? FLARE_KIND.glint : FLARE_KIND.trail
      this.fcol[i] = Math.floor(r() * 3)
    }

    this.build()
    this.setReward(null)

    // Resize, visibility, input.
    this.ro = new ResizeObserver(() => this.resize())
    this.ro.observe(container)
    this.resize()
    if (typeof IntersectionObserver !== 'undefined') {
      this.io = new IntersectionObserver(entries => {
        for (const e of entries) this.inView = e.isIntersecting
        this.updateLoop()
      }, { rootMargin: '80px' })
      this.io.observe(container)
    }
    document.addEventListener('visibilitychange', this.onVisibility)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerleave', this.onPointerLeave)
    canvas.addEventListener('click', this.onClick)
    this.updateLoop()
  }

  // ---------------------------------------------------------------- public API
  setActive(id: AuraModelLayerId) {
    const i = STAGE[id]
    if (i === undefined || i === this.active) return
    const prev = this.active
    this.active = i
    if (this.playing) this.arrive(i, prev)
    else if (i === STAGE.reward) this.burst(26)
    this.invalidate()
  }

  setPlaying(on: boolean) {
    if (on === this.playing) return
    this.playing = on
    if (on) this.startPacket()
    else this.endPacket()
    this.invalidate()
  }

  setDims(d: AuraModelDims) {
    const k = (x: AuraModelDims) => `${x.seq}|${x.input}|${x.model}|${x.heads}|${x.blocks}|${x.ff}`
    if (k(d) === k(this.dims)) return
    this.dims = { ...d }
    this.build()
    this.invalidate()
  }

  setMotion(m: AuraModelMotion | null) {
    const valid = m && Array.isArray(m.positions) && m.positions.length > 1 && m.positions[0]?.length === 34 ? m : null
    if (valid === this.motion) return
    this.motion = valid
    this.build()
    this.invalidate()
  }

  /** Hero: zoom the camera to one stage (bigger, like the explorer), or back to the overview with null. */
  setView(id: AuraModelLayerId | null) {
    const i = id ? STAGE[id] : -1
    if (i === this.heroFocus) return
    this.heroFocus = i ?? -1
    if (i !== undefined && i >= 0) this.active = i
    this.invalidate()
  }

  /**
   * Keep the top / bottom fraction of the canvas clear (full-screen hero: the title sits on the
   * world's sky). The projection is shifted so the diorama is framed in the band between.
   */
  setSafeArea(top: number, bottom: number) {
    const t = clamp(top, 0, 0.75), b = clamp(bottom, 0, 0.4)
    if (Math.abs(t - this.safeTop) < 0.004 && Math.abs(b - this.safeBottom) < 0.004) return
    this.safeTop = t; this.safeBottom = b
    this.invalidate() // the camera glides to the new band (see updateCamera)
  }

  /** Projection shift that centres the safe band: 2c - 1, c = band centre as a fraction from the top. */
  private safeShift() { return this.safeTop + (1 - this.safeBottom) - 1 }

  private static shiftView(cam: THREE.PerspectiveCamera, w: number, h: number, sh: number) {
    if (Math.abs(sh) < 0.002) cam.clearViewOffset()
    else cam.setViewOffset(w, h * (1 + Math.abs(sh)), 0, sh < 0 ? h * -sh : 0, w, h)
  }

  private applyViewOffset() {
    AuraModelScene.shiftView(this.camera, this.width, this.height, this.cam.cur[6])
    this.camera.updateProjectionMatrix()
  }

  /** Normalised reward 0..1 (rank percentile), or null for an untrained / unscored model (neutral orb). */
  setReward(norm: number | null) {
    this.reward = norm == null || !Number.isFinite(norm) ? null : clamp(norm, 0, 1)
    const lv = this.reward ?? 0.5
    this.orbLevel = lv
    this.orbScale = this.reward == null ? 0.38 : 0.3 + 0.26 * lv
    const u = this.orbMat.uniforms
    const core = u.uCore.value as THREE.Color, shade = u.uShade.value as THREE.Color
    if (this.reward == null) {
      core.copy(PAL.egg).lerp(PAL.sage, 0.35)
      shade.copy(PAL.sage).lerp(PAL.navy, 0.35)
    } else {
      core.copy(PAL.sunset).lerp(PAL.hot, lv * 0.55).lerp(PAL.terra, (1 - lv) * 0.35)
      shade.copy(PAL.terra).lerp(PAL.navy, 0.25 + (1 - lv) * 0.25)
    }
    u.uLevel.value = lv
    this.invalidate()
  }

  /**
   * The reward orb pops after a launch: a bounce plus a gold sparkle burst for a correct
   * prediction, a soft sage "poof" for a miss, a small burst otherwise. Reduced motion: no particles.
   */
  pop(kind: AuraModelPop) {
    const b = this.built
    this.popT = 0
    this.pulse[STAGE.reward] = 1
    if (!b || this.reduced) { this.invalidate(); return }
    const o = b.orbPos
    if (kind === 'miss') for (let k = 0; k < 34; k++) this.spawn(o.x, o.y, o.z, this.orbScale * 0.9, 2)
    else for (let k = 0; k < (kind === 'win' ? 56 : 22); k++) this.spawn(o.x, o.y, o.z, this.orbScale, this.rand() < 0.6 ? 1 : 0)
  }

  /** Codex unlock: the stage pulses and sparkles. */
  celebrate(id: AuraModelLayerId) {
    const i = STAGE[id], b = this.built
    if (i === undefined || !b) return
    this.pulse[i] = 1
    if (!this.reduced) {
      const box = b.stages[i].box, r = this.rand
      for (let k = 0; k < 26; k++) this.spawn(box.min.x + r() * (box.max.x - box.min.x), box.min.y + r() * (box.max.y - box.min.y), box.min.z + r() * (box.max.z - box.min.z), 0.04, 1)
    }
    this.invalidate()
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    cancelAnimationFrame(this.raf)
    this.raf = 0
    this.io?.disconnect()
    this.ro?.disconnect()
    document.removeEventListener('visibilitychange', this.onVisibility)
    const canvas = this.renderer.domElement
    canvas.removeEventListener('pointermove', this.onPointerMove)
    canvas.removeEventListener('pointerleave', this.onPointerLeave)
    canvas.removeEventListener('click', this.onClick)
    this.disposeBuilt()
    this.orbGeo.dispose(); this.orbMat.dispose()
    this.boneGeo.dispose(); this.dynGeo.dispose()
    this.pointMat.dispose(); this.lineMat.dispose()
    this.world.dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
    canvas.remove()
    for (const l of this.labelEls) { l.el.style.transform = ''; delete l.el.dataset.on }
    this.labelEls = []
  }

  // ---------------------------------------------------------------- build
  private disposeBuilt() {
    if (!this.built) return
    this.built.group.removeFromParent()
    for (const g of this.built.geos) g.dispose()
    for (const m of this.built.mats) m.dispose()
    this.built = null
  }

  private build() {
    this.disposeBuilt()
    const D = this.dims
    const seq = clamp(Math.round(D.seq) || 64, 4, 256)
    const heads = clamp(Math.round(D.heads) || 4, 1, 16)
    const blocks = clamp(Math.round(D.blocks) || 2, 1, 8)
    const hIn = clamp(D.input * 0.0058, 0.7, 3.0)
    const hM = clamp(D.model * 0.0058, 0.25, 1.4)
    const rowsIn = clamp(Math.round(D.input / 8), 6, 64)
    const rowsM = clamp(Math.round(D.model / 8), 3, 24)
    const nodesA = clamp(Math.round(D.model), 4, 256)
    const rand = mulberry32(seq * 131 + D.input * 7 + D.model)

    // X layout (before centring).
    const xTraj = 0, xFeat = 1.55, xProj = 3.05, xPos = 4.5
    const xBlocks: number[] = []
    for (let b = 0; b < blocks; b++) xBlocks.push(6.0 + b * 1.5)
    const xLastBlock = xBlocks[xBlocks.length - 1]
    const xPool0 = xLastBlock + 0.55, xApex = xLastBlock + 1.6
    const xA = xApex + 0.5, xB = xApex + 1.1, xOut = xApex + 1.7
    const xOrb = xApex + 2.6
    const shift = (xTraj - 0.4 + xOrb + 0.5) / 2
    const X = (x: number) => x - shift
    const Z = (i: number) => (i / (seq - 1) - 0.5) * SPAN
    const yRow = (r: number, rows: number, h: number) => (rows <= 1 ? 0 : (r / (rows - 1) - 0.5) * h)

    const group = new THREE.Group()
    group.name = 'AuraModelStages'
    const geos: THREE.BufferGeometry[] = []
    const mats: THREE.Material[] = []
    const glass: Built['glass'] = []
    const pts = new PointBuf()
    const lines = new LineBuf()
    const S = STAGE

    // ---- motion -> resampled scene-space skeleton frames
    const motion = this.motion ?? (this.fallback ??= proceduralG1Motion())
    const parents = motion.parents?.length === 34 ? motion.parents : G1_PARENTS
    const T = motion.positions.length
    const J = 35 // 34 joints + a head point
    const skel = new Float32Array(seq * J * 3)
    const rel = new Float32Array(seq * 34 * 3) // root-relative metres (features)
    const yBase = -0.66
    const sc = 0.95, fwd = 0.38
    let minY = Infinity
    for (let i = 0; i < seq; i++) {
      const f = motion.positions[Math.round((i * (T - 1)) / (seq - 1))]
      for (let j = 0; j < 34; j++) minY = Math.min(minY, f[j]?.[1] ?? 0)
    }
    if (!Number.isFinite(minY)) minY = 0
    for (let i = 0; i < seq; i++) {
      const f = motion.positions[Math.round((i * (T - 1)) / (seq - 1))]
      const rx = f[0]?.[0] ?? 0, rz = f[0]?.[2] ?? 0
      for (let j = 0; j < 34; j++) {
        const p = f[j] ?? [0, 0, 0]
        const lx = (p[0] ?? 0) - rx, ly = (p[1] ?? 0) - minY, lz = (p[2] ?? 0) - rz
        rel[(i * 34 + j) * 3] = lx; rel[(i * 34 + j) * 3 + 1] = ly; rel[(i * 34 + j) * 3 + 2] = lz
        const o = (i * J + j) * 3
        skel[o] = X(xTraj) + lx * sc
        skel[o + 1] = yBase + ly * sc
        skel[o + 2] = Z(i) + lz * sc * fwd
      }
      // head: above the shoulder midpoint
      const o = (i * J + 34) * 3, a = (i * J + 18) * 3, b = (i * J + 26) * 3, t = (i * J + 17) * 3
      skel[o] = (skel[a] + skel[b]) / 2 + ((skel[a] + skel[b]) / 2 - skel[t]) * 0.6
      skel[o + 1] = (skel[a + 1] + skel[b + 1]) / 2 + 0.15
      skel[o + 2] = (skel[a + 2] + skel[b + 2]) / 2
    }
    const bones: [number, number][] = []
    for (let j = 1; j < 34; j++) if (parents[j] >= 0 && parents[j] !== j) bones.push([parents[j], j])
    bones.push([17, 34])

    // ---- 1 trajectory: joint cloud, key skeletons, joint trails
    const eggDim = PAL.egg.clone().lerp(PAL.sage, 0.25)
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < J; j++) {
        if (j === 15 || j === 16) continue // pelvis duplicates
        const o = (i * J + j) * 3
        pts.add(skel[o], skel[o + 1], skel[o + 2], j === 34 ? PAL.sunset : eggDim, j === 34 ? 0.05 : 0.024, j === 34 ? 0.5 : 0.32, FLARE_KIND.dust, S.trajectory, i / (seq - 1), 1)
      }
    }
    const keyStep = Math.max(1, Math.round(seq / 8))
    for (let i = 0; i < seq; i += keyStep) {
      for (const [pa, ch] of bones) {
        const a = (i * J + pa) * 3, b = (i * J + ch) * 3
        lines.add(skel[a], skel[a + 1], skel[a + 2], skel[b], skel[b + 1], skel[b + 2], PAL.egg, 0.008, 0.5, S.trajectory, LINE_KIND.plain)
      }
    }
    const trailJoints: [number, THREE.Color][] = [[34, PAL.sunset], [25, PAL.terra], [33, PAL.terra], [7, PAL.sage], [14, PAL.sage], [0, PAL.egg]]
    for (const [j, c] of trailJoints) {
      for (let i = 0; i < seq - 1; i++) {
        const a = (i * J + j) * 3, b = ((i + 1) * J + j) * 3
        lines.add(skel[a], skel[a + 1], skel[a + 2], skel[b], skel[b + 1], skel[b + 2], c, 0.009, 0.55, S.trajectory, LINE_KIND.trail, 0, i / (seq - 1), (i + 1) / (seq - 1))
      }
    }

    // ---- 2 features: per-frame feature column from the motion (pos, vel, root vel / height, rotation proxies)
    const feat = new Float32Array(seq * rowsIn)
    const chan = (i: number, c: number) => {
      const r = (k: number) => rel[i * 102 + k]
      if (c < 102) return r(c)
      if (c < 204) { const k = c - 102, n = Math.min(seq - 1, i + 1), p = Math.max(0, i - 1); return (rel[n * 102 + k] - rel[p * 102 + k]) * 4 }
      if (c < 208) return c === 207 ? r(1) : r((c - 204) % 3) * 2
      const k = (c * 7) % 102
      return Math.sin(r(k) * 6 + c * 0.37)
    }
    for (let r = 0; r < rowsIn; r++) {
      const c = Math.floor(((r + 0.5) / rowsIn) * D.input)
      let mean = 0, sq = 0
      for (let i = 0; i < seq; i++) { const v = chan(i, c); feat[i * rowsIn + r] = v; mean += v }
      mean /= seq
      for (let i = 0; i < seq; i++) sq += (feat[i * rowsIn + r] - mean) ** 2
      const sd = Math.sqrt(sq / seq) || 1
      for (let i = 0; i < seq; i++) feat[i * rowsIn + r] = clamp((feat[i * rowsIn + r] - mean) / sd, -2.5, 2.5)
    }
    const ramp = (z: number, out: THREE.Color) => {
      const t = clamp(0.5 + z * 0.26, 0, 1)
      if (t < 0.5) return out.copy(PAL.navy).lerp(PAL.sage, 0.35 + t * 1.1)
      return out.copy(PAL.sunset).lerp(t > 0.8 ? PAL.hot : PAL.terra, t > 0.8 ? (t - 0.8) * 4 : (0.8 - t) * 0.9)
    }
    const tmp = new THREE.Color()
    for (let i = 0; i < seq; i++) {
      for (let r = 0; r < rowsIn; r++) {
        const z = feat[i * rowsIn + r]
        pts.add(X(xFeat), yRow(r, rowsIn, hIn), Z(i), ramp(z, tmp), 0.028 + 0.012 * Math.min(2, Math.abs(z)), 0.5 + 0.18 * Math.abs(z), FLARE_KIND.dust, S.features, i / (seq - 1), 1)
      }
    }

    // ---- 3+ tokens: a random projection of the features (tanh), then smoothed per block ("attention mixes time")
    const W = Array.from({ length: rowsM * 6 }, () => [Math.floor(rand() * rowsIn), rand() * 2 - 1])
    let tok = new Float32Array(seq * rowsM)
    for (let i = 0; i < seq; i++) for (let r = 0; r < rowsM; r++) {
      let s = 0
      for (let k = 0; k < 6; k++) { const [row, w] = W[r * 6 + k]; s += feat[i * rowsIn + row] * w }
      tok[i * rowsM + r] = Math.tanh(s * 0.6)
    }
    const tokColor = (v: number, out: THREE.Color, warm: number) => {
      const t = 0.5 + v * 0.5
      out.copy(PAL.sage).lerp(PAL.egg, t * 0.85)
      return out.lerp(PAL.sunset, warm * t)
    }
    const sheet = (x: number, values: Float32Array, stage: number, fx: number, warm: number) => {
      for (let i = 0; i < seq; i++) for (let r = 0; r < rowsM; r++) {
        const v = values[i * rowsM + r]
        pts.add(x, yRow(r, rowsM, hM), Z(i), tokColor(v, tmp, warm), 0.042 + 0.012 * Math.abs(v), 0.62 + 0.25 * Math.abs(v), FLARE_KIND.dust, stage, i / (seq - 1), fx)
      }
    }
    sheet(X(xProj + 0.32), tok, S.projection, 1, 0)
    // compression lines: feature column edge -> token sheet edge
    for (let r = 0; r < rowsM; r++) for (const z of [Z(0), Z(seq - 1)]) {
      const yi = yRow(Math.round((r * (rowsIn - 1)) / Math.max(1, rowsM - 1)), rowsIn, hIn)
      lines.add(X(xFeat + 0.2), yi, z, X(xProj + 0.32), yRow(r, rowsM, hM), z, PAL.sunset, 0.006, 0.3, S.projection, LINE_KIND.flow, rand(), 0, 1)
    }
    // 4 position: same tokens + the learned time wave
    sheet(X(xPos), tok, S.position, 2, 0.15)
    for (let i = 0; i < seq - 1; i++) {
      const y = (k: number) => hM / 2 + 0.17 + 0.06 * Math.sin((k / (seq - 1)) * Math.PI * 6)
      lines.add(X(xPos), y(i), Z(i), X(xPos), y(i + 1), Z(i + 1), PAL.sunset, 0.012, 0.7, S.position, LINE_KIND.flow, 0, i / (seq - 1) * 3, (i + 1) / (seq - 1) * 3)
    }
    // 5 transformer blocks
    const arcSeg = 14
    const hOff = heads > 1 ? Math.min(0.1, 0.36 / (heads - 1)) : 0
    for (let b = 0; b < blocks; b++) {
      const next = new Float32Array(seq * rowsM)
      for (let i = 0; i < seq; i++) for (let r = 0; r < rowsM; r++) {
        let s = 0, n = 0
        for (let k = -3; k <= 3; k++) { const ii = clamp(i + k, 0, seq - 1); s += tok[ii * rowsM + r]; n++ }
        next[i * rowsM + r] = Math.tanh(tok[i * rowsM + r] * 0.6 + (s / n) * 0.9)
      }
      tok = next
      const xb = X(xBlocks[b])
      sheet(xb, tok, S.transformer, 2, 0.25 + 0.2 * b)
      const yTop = hM / 2 + 0.04
      const qStep = Math.max(2, Math.round(seq / 13))
      for (let h = 0; h < heads; h++) {
        const xh = xb + (h - (heads - 1) / 2) * hOff
        const c = HEAD_COLORS[h % HEAD_COLORS.length]
        const type = (h + b) % 4
        const period = Math.max(2, Math.round(seq / 8))
        for (let q = (h * 3 + b) % qStep; q < seq; q += qStep) {
          const keys: number[] = []
          if (type === 0) keys.push(clamp(q + (rand() < 0.5 ? -1 : 1) * (1 + Math.floor(rand() * 4)), 0, seq - 1))
          else if (type === 1) { if (q - period >= 0) keys.push(q - period); if (q + period < seq) keys.push(q + period) }
          else if (type === 2) keys.push(rand() < 0.5 ? 0 : seq - 1)
          else keys.push(Math.floor(rand() * seq))
          for (const k of keys) {
            if (k === q) continue
            const dist = Math.abs(k - q) / seq
            const ah = 0.1 + 0.72 * Math.sqrt(dist) + h * 0.02
            const ph = rand()
            for (let s = 0; s < arcSeg; s++) {
              const t0 = s / arcSeg, t1 = (s + 1) / arcSeg
              const y0 = yTop + ah * Math.sin(Math.PI * t0), y1 = yTop + ah * Math.sin(Math.PI * t1)
              const z0 = Z(k) + (Z(q) - Z(k)) * t0, z1 = Z(k) + (Z(q) - Z(k)) * t1
              lines.add(xh, y0, z0, xh, y1, z1, c, 0.011, 0.95, S.transformer, LINE_KIND.arc, ph, t0, t1)
            }
          }
        }
      }
    }
    // 6 pool: funnel from the token sheet into one point
    const apex = new THREE.Vector3(X(xApex), 0, 0)
    for (let i = 0; i < seq; i += Math.max(1, Math.round(seq / 16))) {
      for (const r of [0, rowsM - 1]) {
        lines.add(X(xPool0), yRow(r, rowsM, hM), Z(i), apex.x, apex.y, apex.z, PAL.egg, 0.006, 0.32, S.pool, LINE_KIND.flow, rand(), 0, 1)
      }
    }
    pts.add(apex.x, apex.y, apex.z, PAL.hot, 0.34, 0.95, FLARE_KIND.head, S.pool)
    // 7 head: d_model -> 64 -> 1 neuron fan
    const yA = (k: number) => yRow(k, nodesA, 1.15), yB = (k: number) => yRow(k, HEAD_HIDDEN, 0.86)
    for (let k = 0; k < nodesA; k++) pts.add(X(xA), yA(k), 0, PAL.egg.clone().lerp(PAL.sage, 0.3), 0.03, 0.75, FLARE_KIND.dust, S.head)
    for (let k = 0; k < HEAD_HIDDEN; k++) pts.add(X(xB), yB(k), 0, PAL.sunset, 0.034, 0.8, FLARE_KIND.dust, S.head)
    pts.add(X(xOut), 0, 0, PAL.hot, 0.16, 1, FLARE_KIND.glint, S.head)
    for (let k = 0; k < nodesA; k += Math.max(1, Math.round(nodesA / 12))) {
      lines.add(apex.x, apex.y, apex.z, X(xA), yA(k), 0, PAL.egg, 0.005, 0.3, S.head, LINE_KIND.flow, rand(), 0, 1)
    }
    for (let j = 0; j < HEAD_HIDDEN; j++) {
      for (let e = 0; e < 4; e++) {
        const k = Math.floor(rand() * nodesA)
        lines.add(X(xA), yA(k), 0, X(xB), yB(j), 0, rand() < 0.5 ? PAL.sage : PAL.egg, 0.0045, 0.22, S.head, LINE_KIND.flow, rand(), 0, 1)
      }
      lines.add(X(xB), yB(j), 0, X(xOut), 0, 0, PAL.sunset, 0.005, 0.3, S.head, LINE_KIND.flow, rand(), 0, 1)
    }
    // 8 reward: line from the output neuron into the orb
    const orbPos = new THREE.Vector3(X(xOrb), 0, 0)
    lines.add(X(xOut), 0, 0, orbPos.x - 0.2, 0, 0, PAL.sunset, 0.012, 0.6, S.reward, LINE_KIND.flow, 0, 0, 1)

    // bloom-like halos behind every stage (lit by the stage glow)
    const haloAt = (x: number, y: number, size: number, c: THREE.Color, stage: number, a = 0.07) => pts.add(x, y, 0, c, size, a, FLARE_KIND.dust, stage)

    // ---- glass slabs
    const slab = (stage: number, w: number, h: number, d: number, x: number, y: number, z: number, seed: number, taper?: { y: number; z: number }) => {
      const r = Math.min(w, h, d) * 0.22
      const geo = new RoundedBoxGeometry(w, h, d, 3, r)
      const pos = geo.getAttribute('position') as THREE.BufferAttribute
      geo.setAttribute('aBox', pos.clone())
      if (taper) {
        for (let k = 0; k < pos.count; k++) {
          const f = clamp(pos.getX(k) / w + 0.5, 0, 1)
          pos.setY(k, pos.getY(k) * (1 + (taper.y - 1) * f))
          pos.setZ(k, pos.getZ(k) * (1 + (taper.z - 1) * f))
        }
        geo.computeVertexNormals()
      }
      const learned = LEARNED[stage]
      const mat = new THREE.ShaderMaterial({
        name: 'AuraModelGlass',
        uniforms: {
          ...palUniforms(),
          uTint: { value: learned ? PAL.navy.clone().lerp(PAL.terra, 0.35) : PAL.navy.clone().lerp(PAL.sage, 0.4) },
          uRim: { value: learned ? PAL.terra.clone().lerp(PAL.sunset, 0.45) : PAL.egg.clone().lerp(PAL.sage, 0.15) },
          uHalf: { value: new THREE.Vector3(w / 2, h / 2, d / 2) },
          cHot: { value: PAL.hot.clone() },
          uR: { value: r }, uGlow: { value: 0.6 }, uTime: this.uTime, uSeed: { value: seed }, uLearned: { value: learned ? 1 : 0 },
        },
        vertexShader: GLASS_VERT,
        fragmentShader: GLASS_FRAG,
        transparent: true, depthTest: true, depthWrite: true, side: THREE.FrontSide,
        blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
        blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.ZeroFactor,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(x, y, z)
      mesh.renderOrder = 2
      group.add(mesh)
      geos.push(geo); mats.push(mat); glass.push({ mat, stage, mesh })
    }
    slab(S.trajectory, 0.7, 0.05, SPAN + 0.34, X(xTraj), yBase - 0.05, 0, 0.1)
    slab(S.features, 0.34, hIn + 0.26, SPAN + 0.34, X(xFeat), 0, 0, 0.3)
    slab(S.projection, 0.78, hIn + 0.26, SPAN + 0.34, X(xProj), 0, 0, 0.5, { y: (hM + 0.22) / (hIn + 0.26), z: 1 })
    slab(S.position, 0.32, hM + 0.5, SPAN + 0.34, X(xPos), 0.12, 0, 0.7)
    const blockTop = hM / 2 + 1.0, blockBot = -hM / 2 - 0.14
    for (let b = 0; b < blocks; b++) slab(S.transformer, 0.5, blockTop - blockBot, SPAN + 0.34, X(xBlocks[b]), (blockTop + blockBot) / 2, 0, 0.9 + b * 0.37)
    slab(S.pool, xApex - xPool0 + 0.1, hM + 0.22, SPAN + 0.34, X((xPool0 + xApex) / 2), 0, 0, 0.2, { y: 0.12, z: 0.04 })
    slab(S.head, xOut - xA + 0.5, 1.45, 0.22, X((xA + xOut) / 2), 0, 0, 0.6)

    haloAt(X(xTraj), 0, 3.2, PAL.sage, S.trajectory, 0.05)
    haloAt(X(xFeat), 0, 3.6, PAL.sage, S.features, 0.06)
    haloAt(X(xProj), 0, 3.0, PAL.terra, S.projection)
    haloAt(X(xPos), 0, 2.6, PAL.sunset, S.position)
    for (const xb of xBlocks) haloAt(X(xb), 0.4, 3.4, PAL.sunset, S.transformer, 0.08)
    haloAt(apex.x, 0, 2.2, PAL.sunset, S.pool, 0.08)
    haloAt(X((xA + xOut) / 2), 0, 2.4, PAL.terra, S.head)
    haloAt(orbPos.x, 0, 3.2, PAL.sunset, S.reward, 0.1)

    // ambient fairy dust around the diorama (twinkles)
    for (let k = 0; k < (this.mode === 'hero' ? 90 : 70); k++) {
      const x = X(-0.8 + rand() * (xOrb + 1.6)), y = -1.6 + rand() * 3.6, z = -2.4 + rand() * 4.6
      pts.add(x, y, z, FLARE_COLORS[1 + Math.floor(rand() * 3)], 0.02 + rand() * 0.03, 0.35, rand() < 0.15 ? FLARE_KIND.glint : FLARE_KIND.dust, -1, rand(), 3)
    }

    // ---- meshes
    const pg = pts.geometry()
    const pMesh = new THREE.Points(pg, this.pointMat)
    pMesh.frustumCulled = false
    pMesh.renderOrder = 5
    pMesh.layers.set(AURA_OVERLAY_LAYER)
    group.add(pMesh)
    const lg = lines.geometry()
    const lMesh = new THREE.Mesh(lg, this.lineMat)
    lMesh.frustumCulled = false
    lMesh.renderOrder = 3
    lMesh.layers.set(AURA_OVERLAY_LAYER)
    group.add(lMesh)
    geos.push(pg, lg)
    this.scene.add(group)

    // ---- stages (hit boxes, label anchors)
    const zh = SPAN / 2 + 0.2
    const mk = (id: AuraModelLayerId, x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, cy = 0): Stage => ({
      id, cx: (X(x0) + X(x1)) / 2, cy,
      box: new THREE.Box3(new THREE.Vector3(X(x0), y0, z0), new THREE.Vector3(X(x1), y1, z1)),
      label: new THREE.Vector3((X(x0) + X(x1)) / 2, y0 - 0.12, 0),
    })
    const stages: Stage[] = [
      mk('trajectory', xTraj - 0.45, xTraj + 0.45, yBase - 0.1, 0.75, -zh, zh),
      mk('features', xFeat - 0.35, xFeat + 0.35, -hIn / 2 - 0.15, hIn / 2 + 0.15, -zh, zh),
      mk('projection', xProj - 0.45, xProj + 0.45, -hIn / 2 - 0.15, hIn / 2 + 0.15, -zh, zh),
      mk('position', xPos - 0.35, xPos + 0.35, -hM / 2 - 0.15, hM / 2 + 0.4, -zh, zh, 0.1),
      mk('transformer', xBlocks[0] - 0.4, xLastBlock + 0.4, blockBot, blockTop, -zh, zh, 0.3),
      mk('pool', xPool0, xApex + 0.15, -hM / 2 - 0.15, hM / 2 + 0.15, -zh, zh),
      mk('head', xA - 0.3, xOut + 0.25, -0.75, 0.75, -0.3, 0.3),
      mk('reward', xOrb - 0.55, xOrb + 0.55, -0.55, 0.55, -0.55, 0.55),
    ]
    stages[STAGE.pool].label.x = apex.x
    const bounds = new THREE.Box3()
    for (const s of stages) bounds.union(s.box)

    const knots: Knot[] = [
      { x: X(xTraj - 0.2), h: 0.55, d: SPAN / 2 },
      { x: X(xFeat), h: hIn / 2, d: SPAN / 2 },
      { x: X(xProj - 0.35), h: hIn / 2, d: SPAN / 2 },
      { x: X(xProj + 0.4), h: hM / 2, d: SPAN / 2 },
      { x: X(xPool0), h: hM / 2, d: SPAN / 2 },
      { x: X(xApex), h: 0.02, d: 0.02 },
      { x: X(xA), h: 0.55, d: 0.04 },
      { x: X(xB), h: 0.42, d: 0.03 },
      { x: X(xOut), h: 0, d: 0 },
      { x: orbPos.x, h: 0, d: 0 },
    ]

    this.buildCount++
    this.built = { group, geos, mats, glass, stages, knots, xStart: knots[0].x, xEnd: orbPos.x, orbPos, bounds, skel, skelJoints: J, bones, trajCx: X(xTraj) }
    this.orb.position.copy(orbPos)
    this.collectLabels()
    this.camSnap = true
  }

  private collectLabels() {
    this.labelEls = []
    if (!this.labels) return
    this.labels.querySelectorAll<HTMLElement>('[data-layer]').forEach(el => {
      const id = el.dataset.layer as AuraModelLayerId
      if (STAGE[id] !== undefined) this.labelEls.push({ el, stage: STAGE[id], x: NaN, y: NaN, on: '' })
    })
  }

  /** Re-scan the overlay's [data-layer] children (call after React re-renders them). */
  refreshLabels() { this.collectLabels(); this.invalidate() }

  // ---------------------------------------------------------------- loop
  private onVisibility = () => { this.pageVisible = document.visibilityState !== 'hidden'; this.updateLoop() }

  private get running() { return !this.disposed && this.inView && this.pageVisible && !this.reduced }

  private updateLoop() {
    if (this.running) {
      if (!this.raf) { this.clock.start(); this.clock.getDelta(); this.raf = requestAnimationFrame(this.tick) }
    } else {
      if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0 }
      this.clock.stop()
      this.invalidate()
    }
  }

  /** Reduced motion / paused: render one static frame (coalesced to the next animation frame). */
  private invalidate() {
    if (this.disposed || this.running) return
    if (this.raf) return
    this.raf = requestAnimationFrame(() => { this.raf = 0; if (!this.disposed) this.frame(0) })
  }

  private tick = () => {
    this.raf = 0
    if (!this.running) return
    const dt = Math.min(this.clock.getDelta(), 0.1)
    this.frame(dt)
    this.raf = requestAnimationFrame(this.tick)
  }

  private resize() {
    const r = this.container.getBoundingClientRect()
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height))
    if (w === this.width && h === this.height) return
    this.width = w; this.height = h
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.mode === 'hero' ? 1.75 : 2))
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.applyViewOffset()
    this.camSnap = true
    for (const l of this.labelEls) l.x = NaN
    if (!this.running) this.invalidate()
    else this.frame(0)
  }

  // ---------------------------------------------------------------- frame
  private frame(dt: number) {
    const b = this.built
    if (!b) return
    const reduced = this.reduced
    this.time += dt
    const t = reduced ? 3.2 : this.time
    this.uTime.value = t
    this.uPlay.value = reduced ? 0.62 : (t * 0.11) % 1

    const playing = this.playing

    // Stage glow targets.
    const act = this.active
    for (let i = 0; i < NSTAGE; i++) {
      let g: number
      if (playing) g = i === act ? 1 : i < act ? 0.72 : 0.42
      else if (this.mode === 'hero') g = this.heroFocus < 0 ? 0.74 : i === this.heroFocus ? 1 : i === this.hovered ? 0.82 : 0.5
      else g = i === act ? 1 : i === this.hovered ? 0.82 : 0.5
      this.glowTarget[i] = g
    }
    const kg = reduced || dt === 0 ? 1 : 1 - Math.exp(-dt * 5)
    for (let i = 0; i < NSTAGE; i++) {
      this.glow[i] += (this.glowTarget[i] - this.glow[i]) * kg
      this.pulse[i] = reduced ? 0 : this.pulse[i] * Math.exp(-dt * 2.8)
    }
    for (const gl of b.glass) {
      const pz = this.pulse[gl.stage]
      gl.mat.uniforms.uGlow.value = this.glow[gl.stage] + pz * 0.7
      gl.mesh.scale.setScalar(this.focusScale[gl.stage] * (1 + 0.06 * pz * Math.sin(Math.min(1, pz) * Math.PI)))
    }
    this.orbMat.uniforms.uGlow.value = this.glow[STAGE.reward]
    const waveGoal = act === STAGE.position ? 1 : this.mode === 'hero' ? 0.55 : 0.35
    const arcGoal = act === STAGE.transformer ? 1.15 : this.mode === 'hero' ? 0.9 : 0.6
    this.uWave.value += (waveGoal - this.uWave.value) * kg
    this.uArcs.value += (arcGoal - this.uArcs.value) * kg
    const arcs = this.uArcs.value
    this.uArcs.value = arcs + this.pulse[STAGE.transformer] * 1.4 // attention fires on arrival (restored after render)

    // Orb breathing.
    const breathe = reduced ? 1 : 1 + 0.03 * Math.sin(t * 1.6)
    this.popT += dt
    const pop = reduced ? 0 : 0.45 * Math.exp(-this.popT * 4.5) * Math.cos(this.popT * 16)
    this.orb.scale.setScalar(this.orbScale * breathe * (1 + pop))
    this.orb.rotation.y = t * 0.2

    this.updateCamera(dt)
    this.updateSkeleton()
    this.updateDynamic(dt, t)
    this.uPx.value = this.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2))
    this.world.render(this.camera, dt)
    this.uArcs.value = arcs
    this.updateLabels()
  }

  private stageX(i: number) { return this.built ? this.built.stages[i].cx : 0 }

  /** Distance at which `box` fits the viewport (NDC margin) for this view direction: numeric, so near / far perspective is exact. */
  private fitDistance(box: THREE.Box3, target: THREE.Vector3, dir: THREE.Vector3, mx: number, my: number) {
    const cam = this.fitCam
    cam.copy(this.camera, false)
    cam.near = 0.1; cam.far = 400
    AuraModelScene.shiftView(cam, this.width, this.height, this.safeShift())
    cam.updateProjectionMatrix()
    const m = this.safeTop > 0 ? 0.04 : 1 - my
    const yHi = (1 - 2 * this.safeTop) - m, yLo = (-1 + 2 * this.safeBottom) + m
    let lo = 1, hi = 120
    for (let it = 0; it < 16; it++) {
      const d = (lo + hi) / 2
      cam.position.copy(dir).multiplyScalar(d).add(target)
      cam.lookAt(target)
      cam.updateMatrixWorld()
      let fits = true
      for (let c = 0; c < 8 && fits; c++) {
        this.v.set(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y, c & 4 ? box.max.z : box.min.z).project(cam)
        if (this.v.z > 1 || Math.abs(this.v.x) > mx || this.v.y > yHi || this.v.y < yLo) fits = false
      }
      if (fits) hi = d; else lo = d
    }
    return hi
  }

  private updateCamera(dt: number) {
    const b = this.built
    if (!b) return
    const cam = this.camera, C = this.cam
    // Forward pass: a close-up on the stage the motion is in. Otherwise the hero shows the overview unless a
    // stage is focused, and the explorer frames the active stage with a little context (its neighbours).
    const passing = this.playing
    const focusI = this.mode === 'hero' ? (this.heroFocus >= 0 ? this.heroFocus : passing ? this.active : -1) : this.active
    const overview = focusI < 0
    // portrait canvases look further down the pipeline so the long diorama recedes and can be framed larger
    const portrait = clamp((1.35 - cam.aspect) / 0.85, 0, 1)
    const y0 = (overview ? -0.5 : -0.58) - 0.55 * portrait, p0 = (overview ? 0.2 : 0.24) + 0.1 * portrait
    const box = this.fitBox
    if (overview) {
      box.copy(b.bounds)
      box.getCenter(this.camGoal)
      this.camGoal.x += Math.sin(this.time * 0.13) * 0.12
      this.camGoal.y += 0.15
    } else {
      const i = focusI
      box.copy(b.stages[i].box)
      if (passing) box.expandByScalar(0.35)
      else {
        if (i > 0) box.union(b.stages[i - 1].box)
        if (i < NSTAGE - 1) box.union(b.stages[i + 1].box)
      }
      box.getCenter(this.camGoal)
      this.camGoal.lerp(b.stages[i].box.getCenter(this.v), passing ? 1 : 0.45)
      if (this.safeTop === 0) this.camGoal.y += 0.12
    }
    this.fitDir.set(Math.sin(y0) * Math.cos(p0), Math.sin(p0), Math.cos(y0) * Math.cos(p0))
    this.fitTarget.copy(this.camGoal)
    this.camDistGoal = this.fitDistance(box, this.fitTarget, this.fitDir, overview ? 0.94 : 0.86, overview ? 0.8 : 0.78)
    const G = C.goal
    G[0] = this.camGoal.x; G[1] = this.camGoal.y; G[2] = this.camGoal.z
    G[3] = this.camDistGoal; G[4] = y0; G[5] = p0; G[6] = this.safeShift()

    // A new selection (stage, pass state, safe band, rebuild) starts a glide from the current pose + velocity.
    const changed = focusI !== C.focus || passing !== C.passing || this.safeTop !== C.safeT || this.safeBottom !== C.safeB || this.buildCount !== C.build
    if (this.camSnap || this.reduced) {
      C.cur.set(G); C.v0.fill(0); C.t = C.dur = 1; C.lift = C.swing = 0
    } else if (changed) {
      this.glideVelocity(C.v0) // current velocity (incl. the arc), before overwriting the tween
      // the pose currently on screen (with its arc) is the new start
      C.from.set(C.cur)
      const dx = G[0] - C.cur[0], dy = G[1] - C.cur[1], dz = G[2] - C.cur[2]
      const travel = Math.hypot(dx, dy, dz) + Math.abs(G[3] - C.cur[3]) * 0.35
      C.dur = passing ? 0.9 : overview ? 1.15 : 1.05
      C.lift = Math.min(0.55, travel * 0.08) // gentle lift at mid-flight
      C.swing = Math.min(0.06, travel * 0.012) * (dx >= 0 ? 1 : -1) // and a slight sideways swing
      C.t = 0
    }
    C.focus = focusI; C.passing = passing; C.safeT = this.safeTop; C.safeB = this.safeBottom; C.build = this.buildCount
    this.camSnap = false

    if (C.t < C.dur) {
      if (dt > 0) C.t = Math.min(C.dur, C.t + dt)
      const u = C.t / C.dur, u2 = u * u, u3 = u2 * u
      const h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2
      for (let k = 0; k < 7; k++) C.cur[k] = h00 * C.from[k] + h10 * C.v0[k] * C.dur + h01 * G[k]
      const arc = Math.sin(Math.PI * u)
      C.cur[1] += C.lift * arc
      C.cur[4] += C.swing * arc
    } else {
      C.cur.set(G) // settled: track the live goal (hero drift, resize) exactly
    }

    const kp = this.reduced ? 0 : 1 - Math.exp(-dt * 3)
    this.parallax.x += (this.pointer.x - this.parallax.x) * kp
    this.parallax.y += (this.pointer.y - this.parallax.y) * kp
    const yaw = C.cur[4] + this.parallax.x * 0.07, pitch = C.cur[5] + this.parallax.y * 0.035
    const cp = Math.cos(pitch)
    this.applyViewOffset()
    this.v.set(C.cur[0], C.cur[1], C.cur[2])
    cam.position.set(Math.sin(yaw) * cp, Math.sin(pitch), Math.cos(yaw) * cp).multiplyScalar(C.cur[3]).add(this.v)
    cam.lookAt(this.v)
    cam.updateMatrixWorld()

    // focus: the chosen stage swells a little, eased in time (instant under reduced motion)
    const kf = this.reduced || dt === 0 ? (this.reduced ? 1 : 0) : 1 - Math.exp(-dt * 5)
    for (let i = 0; i < NSTAGE; i++) {
      const goal = !overview && i === focusI ? 1.045 : 1
      this.focusScale[i] += (goal - this.focusScale[i]) * kf
    }
  }

  /** d(pose)/dt of the running glide (zero when settled), written into `out`. */
  private glideVelocity(out: Float64Array) {
    const C = this.cam
    if (C.t >= C.dur) { out.fill(0); return }
    const u = C.t / C.dur, u2 = u * u
    const d00 = 6 * u2 - 6 * u, d10 = 3 * u2 - 4 * u + 1, d01 = -6 * u2 + 6 * u
    for (let k = 0; k < 7; k++) out[k] = (d00 * C.from[k] + d10 * C.v0[k] * C.dur + d01 * C.goal[k]) / C.dur
    const darc = Math.PI * Math.cos(Math.PI * u) / C.dur
    out[1] += C.lift * darc
    out[4] += C.swing * darc
  }

  /** The playhead skeleton walks along the time axis, interpolating the resampled frames. */
  private updateSkeleton() {
    const b = this.built
    if (!b) return
    const seq = b.skel.length / (b.skelJoints * 3)
    const f = this.uPlay.value * (seq - 1)
    const i0 = Math.floor(f), i1 = Math.min(seq - 1, i0 + 1), w = f - i0
    const J = b.skelJoints, s = b.skel
    const geo = this.boneGeo
    const A = geo.getAttribute('aA') as THREE.InstancedBufferAttribute, B = geo.getAttribute('aB') as THREE.InstancedBufferAttribute
    const C = geo.getAttribute('aCol') as THREE.InstancedBufferAttribute, M = geo.getAttribute('aM') as THREE.InstancedBufferAttribute
    const K = geo.getAttribute('aK') as THREE.InstancedBufferAttribute
    const n = Math.min(b.bones.length, A.count)
    const lift = 0 // skeleton stays on its glass strip
    for (let k = 0; k < n; k++) {
      const [pa, ch] = b.bones[k]
      const a0 = (i0 * J + pa) * 3, a1 = (i1 * J + pa) * 3, c0 = (i0 * J + ch) * 3, c1 = (i1 * J + ch) * 3
      A.setXYZ(k, s[a0] + (s[a1] - s[a0]) * w, s[a0 + 1] + (s[a1 + 1] - s[a0 + 1]) * w + lift, s[a0 + 2] + (s[a1 + 2] - s[a0 + 2]) * w)
      B.setXYZ(k, s[c0] + (s[c1] - s[c0]) * w, s[c0 + 1] + (s[c1 + 1] - s[c0 + 1]) * w + lift, s[c0 + 2] + (s[c1 + 2] - s[c0 + 2]) * w)
      const c = ch === 34 || ch >= 18 ? PAL.sunset : PAL.egg
      C.setXYZ(k, c.r, c.g, c.b)
      M.setXYZ(k, 0.016, 1, STAGE.trajectory)
      K.setXYZW(k, LINE_KIND.bone, 0, 0, 1)
    }
    geo.instanceCount = n
    for (const at of [A, B, C, M, K]) { at.clearUpdateRanges(); at.addUpdateRange(0, n * at.itemSize); at.needsUpdate = true }
  }

  private envelope(x: number, out: THREE.Vector3) {
    const kn = this.built!.knots
    if (x <= kn[0].x) return out.set(kn[0].h, kn[0].d, 0)
    for (let i = 1; i < kn.length; i++) {
      if (x <= kn[i].x) {
        const a = kn[i - 1], c = kn[i], f = (x - a.x) / Math.max(1e-4, c.x - a.x)
        return out.set(a.h + (c.h - a.h) * f, a.d + (c.d - a.d) * f, 0)
      }
    }
    const l = kn[kn.length - 1]
    return out.set(l.h, l.d, 0)
  }

  private sOfX(x: number) {
    const b = this.built!
    return (x - b.xStart) / (b.xEnd - b.xStart)
  }

  private startPacket() {
    // a third of the flow joins the forward-pass packet, starting just before the input
    this.packetS = -0.04
    this.passStage = 0
    for (let i = 0; i < this.flowN; i++) {
      if (i % 3 === 0) { this.flag[i] = this.rand(); this.fs[i] = -0.04 - this.flag[i] * 0.06 }
    }
  }

  private endPacket() {
    for (let i = 0; i < this.flowN; i++) this.flag[i] = -1
    this.passStage = -1
  }

  private arrive(stage: number, prev: number) {
    if (stage <= prev && !(stage === 0 && prev === NSTAGE - 1)) return
    this.passStage = stage
    this.pulse[stage] = 1
    const b = this.built
    if (!b) return
    const box = b.stages[stage].box
    const n = stage === STAGE.reward ? 30 : 9
    for (let k = 0; k < n; k++) {
      if (stage === STAGE.reward) this.spawn(b.orbPos.x, b.orbPos.y, b.orbPos.z, 0.25, 1)
      else {
        const r = this.rand
        this.spawn(box.min.x + r() * (box.max.x - box.min.x), box.min.y + r() * (box.max.y - box.min.y), box.min.z + r() * (box.max.z - box.min.z), 0.05, 1)
      }
    }
  }

  private burst(n: number) {
    const b = this.built
    if (!b || this.reduced) return
    for (let k = 0; k < n; k++) this.spawn(b.orbPos.x, b.orbPos.y, b.orbPos.z, 0.25, 1)
  }

  private spawn(x: number, y: number, z: number, spread: number, kind: 0 | 1 | 2) {
    if (this.sCount >= SPARK_MAX || this.reduced) return
    const i = this.sCount++, o = i * 3, r = this.rand
    const th = r() * Math.PI * 2, ph = Math.acos(2 * r() - 1)
    const dx = Math.sin(ph) * Math.cos(th), dy = Math.cos(ph), dz = Math.sin(ph) * Math.sin(th)
    this.sx[o] = x + dx * spread; this.sx[o + 1] = y + dy * spread; this.sx[o + 2] = z + dz * spread
    const sp = kind === 1 ? 0.25 + r() * 0.6 : kind === 2 ? 0.18 + r() * 0.22 : 0.1 + r() * 0.2
    this.sv[o] = dx * sp; this.sv[o + 1] = dy * sp + (kind === 2 ? 0.25 : 0.1); this.sv[o + 2] = dz * sp
    this.slife[i] = 0
    this.smax[i] = kind === 1 ? 0.5 + r() * 0.7 : kind === 2 ? 0.9 + r() * 0.6 : 0.8 + r() * 0.9
    this.ssize[i] = kind === 1 ? 0.1 + r() * 0.1 : kind === 2 ? 0.09 + r() * 0.08 : 0.025 + r() * 0.02
    this.skind[i] = kind
    this.scol[i] = kind === 2 ? 4 + Math.floor(r() * 2) : Math.floor(r() * 3)
  }

  private removeSpark(i: number) {
    const j = --this.sCount
    if (i === j) return
    const o = i * 3, p = j * 3
    for (let k = 0; k < 3; k++) { this.sx[o + k] = this.sx[p + k]; this.sv[o + k] = this.sv[p + k] }
    this.slife[i] = this.slife[j]; this.smax[i] = this.smax[j]; this.ssize[i] = this.ssize[j]; this.skind[i] = this.skind[j]; this.scol[i] = this.scol[j]
  }

  private updateDynamic(dt: number, t: number) {
    const b = this.built!
    const [pos, colA, pA, qA] = this.dynAttrs
    const reduced = this.reduced
    const playing = this.playing
    let m = 0
    const env = this.v2
    const cHot = FLARE_COLORS[0]

    // Packet front follows the active stage.
    if (playing) {
      const goal = this.sOfX(this.stageX(this.active))
      this.packetS += (goal - this.packetS) * (1 - Math.exp(-dt * 7))
    }
    const activeX = this.stageX(this.active)

    // Flow particles.
    for (let i = 0; i < this.flowN; i++) {
      let s: number
      let alpha = 0.75
      if (reduced) {
        s = (i + 0.5) / this.flowN
      } else if (this.flag[i] >= 0) {
        const goal = this.packetS - this.flag[i] * 0.07
        this.fs[i] += (goal - this.fs[i]) * (1 - Math.exp(-dt * 5.5))
        s = this.fs[i]
        alpha = 1
      } else {
        this.fs[i] += dt * this.fspd[i] * (playing ? 0.6 : 1)
        if (this.fs[i] > 1) { this.fs[i] -= 1; this.fu[i] = this.rand() * 2 - 1; this.fv[i] = this.rand() * 2 - 1 }
        s = this.fs[i]
        if (playing) alpha = 0.35
      }
      if (s < 0) continue
      const x = b.xStart + (b.xEnd - b.xStart) * Math.min(s, 1)
      this.envelope(x, env)
      const wob = reduced ? 0 : Math.sin(t * 1.3 + i) * 0.02
      const y = this.fv[i] * env.x + wob, z = this.fu[i] * env.y
      const fade = THREE.MathUtils.smoothstep(s, 0, 0.03) * (1 - THREE.MathUtils.smoothstep(s, 0.94, 1))
      const near = playing ? 0.75 + 0.6 * Math.exp(-(((x - activeX) / 1.3) ** 2)) : 1
      const glint = this.fkind[i] === FLARE_KIND.glint
      pos.setXYZ(m, x, y, z)
      const c = FLARE_COLORS[glint ? 0 : 1 + this.fcol[i] % 2]
      colA.setXYZ(m, c.r, c.g, c.b)
      pA.setXYZW(m, glint ? 0.11 : 0.06, alpha * fade * near * (reduced ? 0.6 : 1), glint ? FLARE_KIND.glint : FLARE_KIND.trail, -1)
      qA.setXY(m, 0, 0)
      m++
    }

    // Sparkles around the orb (rate ~ reward) and from stage arrivals.
    if (!reduced) {
      this.sparkAcc += dt * (1.5 + 9 * (this.reward ?? 0.25)) * (this.mode === 'hero' ? 0.8 : 1)
      for (; this.sparkAcc >= 1; this.sparkAcc--) this.spawn(b.orbPos.x, b.orbPos.y, b.orbPos.z, this.orbScale * 1.1, this.rand() < 0.45 ? 1 : 0)
      const damp = Math.pow(0.95, dt * 60)
      for (let i = 0; i < this.sCount;) {
        this.slife[i] += dt
        if (this.slife[i] >= this.smax[i]) { this.removeSpark(i); continue }
        const o = i * 3
        this.sv[o] *= damp; this.sv[o + 1] = this.sv[o + 1] * damp - 0.12 * dt; this.sv[o + 2] *= damp
        this.sx[o] += this.sv[o] * dt; this.sx[o + 1] += this.sv[o + 1] * dt; this.sx[o + 2] += this.sv[o + 2] * dt
        const l = this.slife[i] / this.smax[i]
        const envl = THREE.MathUtils.smoothstep(l, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(l, 0.5, 1))
        const tw = 0.5 + 0.5 * Math.sin(i * 1.7 + t * 13)
        const glint = this.skind[i] === 1, poof = this.skind[i] === 2
        pos.setXYZ(m, this.sx[o], this.sx[o + 1], this.sx[o + 2])
        const c = SPARK_COLORS[poof ? this.scol[i] : glint ? this.scol[i] % 2 : 1 + (this.scol[i] % 3)]
        colA.setXYZ(m, c.r, c.g, c.b)
        pA.setXYZW(m, this.ssize[i] * (glint ? 0.6 + 0.4 * envl : poof ? 0.6 + 0.8 * l : 1), envl * (glint ? 0.5 + 0.5 * tw : poof ? 0.55 : 0.9),
          glint ? FLARE_KIND.glint : poof ? FLARE_KIND.trail : FLARE_KIND.dust, -1)
        qA.setXY(m, 0, 0)
        m++
        i++
      }
    }

    // The fairy flare carries the motion: a head with a short trail at the packet front.
    if (playing && !reduced && this.packetS > 0) {
      for (let k = 6; k >= 0; k--) {
        const sk = Math.min(1, this.packetS - k * 0.012)
        if (sk < 0) continue
        const x = b.xStart + (b.xEnd - b.xStart) * sk
        const y = 0.22 + Math.sin(t * 3.1 - k * 0.5) * 0.08, z = Math.sin(t * 2.3 - k * 0.4) * 0.12
        pos.setXYZ(m, x, y, z)
        colA.setXYZ(m, 1, 1, 1)
        pA.setXYZW(m, k === 0 ? 0.42 : 0.16 * (1 - k / 8), k === 0 ? 1 : 0.8 * (1 - k / 7), k === 0 ? FLARE_KIND.head : FLARE_KIND.trail, -1)
        qA.setXY(m, 0, 0)
        m++
      }
    }

    // Orb halo (flare head: soft halo + four-point twinkle), brighter with the reward.
    const lv = this.orbLevel, g = this.glow[STAGE.reward]
    const pulse = reduced ? 1 : 1 + 0.05 * Math.sin(t * 2.3)
    pos.setXYZ(m, b.orbPos.x, b.orbPos.y, b.orbPos.z)
    colA.setXYZ(m, cHot.r, cHot.g, cHot.b)
    pA.setXYZW(m, this.orbScale * (5.5 + 3 * lv) * pulse, (0.45 + 0.4 * lv) * (0.6 + 0.4 * g), FLARE_KIND.head, -1)
    qA.setXY(m, 0, 0)
    m++

    // Playhead skeleton joints.
    const seq = b.skel.length / (b.skelJoints * 3)
    const f = this.uPlay.value * (seq - 1)
    const i0 = Math.floor(f), i1 = Math.min(seq - 1, i0 + 1), w = f - i0, J = b.skelJoints, s = b.skel
    for (const j of [34, 25, 33, 7, 14]) {
      const a = (i0 * J + j) * 3, c = (i1 * J + j) * 3
      pos.setXYZ(m, s[a] + (s[c] - s[a]) * w, s[a + 1] + (s[c + 1] - s[a + 1]) * w, s[a + 2] + (s[c + 2] - s[a + 2]) * w)
      const cc = j === 34 ? FLARE_COLORS[1] : FLARE_COLORS[0]
      colA.setXYZ(m, cc.r, cc.g, cc.b)
      pA.setXYZW(m, j === 34 ? 0.13 : 0.07, 0.9 * (0.5 + 0.5 * this.glow[STAGE.trajectory]), j === 34 ? FLARE_KIND.glint : FLARE_KIND.dust, -1)
      qA.setXY(m, 0, 0)
      m++
    }

    uploadLiveRange(this.dynAttrs, m)
    this.dynGeo.setDrawRange(0, m)
  }

  private updateLabels() {
    if (!this.built) return
    // orb anchor for DOM UI (reward pop, prediction prompt): --orb-x / --orb-y in CSS px
    this.v.copy(this.built.orbPos).project(this.camera)
    const ox = Math.round((this.v.x * 0.5 + 0.5) * this.width), oy = Math.round((-this.v.y * 0.5 + 0.5) * this.height)
    if (ox !== this.orbCss.x || oy !== this.orbCss.y) {
      this.orbCss.x = ox; this.orbCss.y = oy
      this.container.style.setProperty('--orb-x', `${ox}px`)
      this.container.style.setProperty('--orb-y', `${oy}px`)
    }
    if (!this.labelEls.length) return
    const w = this.width, h = this.height
    const playing = this.playing
    for (const l of this.labelEls) {
      const st = this.built.stages[l.stage]
      this.v.copy(st.label).project(this.camera)
      const x = Math.round((this.v.x * 0.5 + 0.5) * w), y = Math.round((-this.v.y * 0.5 + 0.5) * h)
      if (x !== l.x || y !== l.y) {
        l.x = x; l.y = y
        l.el.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, 0)`
      }
      const on = l.stage === this.active && (this.mode === 'explorer' || playing || this.heroFocus >= 0) ? 'active' : l.stage === this.hovered ? 'hover' : ''
      if (on !== l.on) { l.on = on; if (on) l.el.dataset.on = on; else delete l.el.dataset.on }
    }
  }

  // ---------------------------------------------------------------- input
  private pick(e: PointerEvent | MouseEvent) {
    const r = this.renderer.domElement.getBoundingClientRect()
    return this.pickNdc(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
  }

  private pickNdc(x: number, y: number) {
    const b = this.built
    if (!b) return -1
    this.ndc.set(x, y)
    this.ray.setFromCamera(this.ndc, this.camera)
    let best = -1, bestD = Infinity
    for (let i = 0; i < b.stages.length; i++) {
      const hit = this.ray.ray.intersectBox(b.stages[i].box, this.v)
      if (!hit) continue
      const d = hit.distanceTo(this.ray.ray.origin)
      if (d < bestD) { bestD = d; best = i }
    }
    return best
  }

  private onPointerMove = (e: PointerEvent) => {
    const r = this.renderer.domElement.getBoundingClientRect()
    if (this.mode === 'hero') this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1))
    const i = this.pick(e)
    if (i !== this.hovered) {
      this.hovered = i
      this.renderer.domElement.style.cursor = i >= 0 ? 'pointer' : ''
      this.invalidate()
    }
  }

  private onPointerLeave = () => {
    this.pointer.set(0, 0)
    if (this.hovered !== -1) { this.hovered = -1; this.renderer.domElement.style.cursor = ''; this.invalidate() }
  }

  private onClick = (e: MouseEvent) => {
    const i = this.pick(e)
    if (i >= 0) this.onSelect?.(AURA_MODEL_LAYER_IDS[i])
  }
}
