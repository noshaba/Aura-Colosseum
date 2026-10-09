/**
 * "Motion A vs Motion B" hero arena: one WebGLRenderer, one canvas, two G1 robots
 * on pedestals. Pure Three.js; React (HeroArena.tsx) owns the game state and
 * calls into this class. Nothing here knows about, or displays, any score.
 */
import * as THREE from 'three'
import { G1Actor, loadArenaGeometries, type PreparedClip } from './g1Actor'
import { FairyActor, type ArenaActor } from './fairyActor'
import { loadMixamoAsset } from './fairyRig'
import { getCurrentCharacter, type CharacterDef } from './characters'
import { AURA_OVERLAY_LAYER, AURA_PALETTE, auraWorldScheme, copyAuraWorldScheme, createAuraWorld, lerpAuraWorldScheme, type AuraWorld, type AuraWorldScheme, type AuraWorldSplit } from './auraWorld'
import { FLARE_COLORS, FLARE_COLS, FLARE_STOPS, GLOW_BLEND, flarePointFrag, premultiply, uploadLiveRange } from './flareShared'
import { RobotFlare, effectorsByName } from './robotFlare'
import { toonStylize, type ToonStylizeHandle } from './toonStylize'
import { FLARE_SWEEP_SECONDS, FLARE_SWEEP_SLOPE, flareSweepEase, isLoaderFlareRunning, onFlareHandoff, registerFlareTarget } from '../flareHandoff'

export type Side = 'A' | 'B'
export type Choice = Side | 'tie' | 'skip'
/** Environment colour state: idle (light), hovering a side (dark), vote feedback (sunset). */
export type Tone = 'light' | 'dark' | 'feedback'
export type SideTones = Record<'A' | 'B', Tone>
export type HudState = { round: number; xp: number; level: number; levelProgress: number; streak: number; playing: boolean }
export type ArenaCallbacks = {
  onPick: (side: Side) => void
  onPlay: () => void
  onHover?: (side: Side | null) => void
  /** Per-side scheme: left half of the arena is A, right half is B. */
  onTone?: (tones: SideTones) => void
  onContextLost?: () => void
}
export type LaneContent = { clip: PreparedClip; label: string }

/**
 * Every themed colour in the arena. The arena is split down the middle: the left
 * half (lane A) and the right half (lane B) each carry their own scheme. `paper`,
 * `accent` and `ink` also drive the canvas-texture labels (see splitMaterial):
 * labels are painted once as channel weights, so a scheme change never redraws a canvas.
 */
const SCHEME_KEYS = ['gold', 'burst', 'pedestal', 'cap', 'ink', 'paper', 'accent', 'shadow', 'p0', 'p1', 'p2', 'p3'] as const
type SchemeKey = typeof SCHEME_KEYS[number]
type Scheme = Record<SchemeKey, string>
type SchemeColors = Record<SchemeKey, THREE.Color>

type Theme = {
  floor: string; s150: string; s200: string; s400: string; s500: string; s600: string; ink: string
  font: string; mono: string
  schemes: Record<Tone, Scheme>
}

function cssVar(style: CSSStyleDeclaration, name: string, fallback: string) {
  const v = style.getPropertyValue(name).trim()
  return v || fallback
}

/** sRGB mix of two hex colours to a hex string, like CSS color-mix(in srgb, a, b t). (auraWorld's mixHex mixes in linear and returns a Color.) */
function mixHexSrgb(a: string, b: string, t: number) {
  const ca = new THREE.Color(a), cb = new THREE.Color(b)
  const ra = { r: 0, g: 0, b: 0 }, rb = { r: 0, g: 0, b: 0 }
  ca.getRGB(ra, THREE.SRGBColorSpace); cb.getRGB(rb, THREE.SRGBColorSpace)
  const { lerp } = THREE.MathUtils
  return '#' + new THREE.Color().setRGB(lerp(ra.r, rb.r, t), lerp(ra.g, rb.g, t), lerp(ra.b, rb.b, t), THREE.SRGBColorSpace).getHexString()
}

/** Scene colours follow the live design tokens (fallbacks: the five-colour palette in palette.css). */
function readTheme(): Theme {
  const s = getComputedStyle(document.documentElement)
  const egg = cssVar(s, '--fx-eggshell', AURA_PALETTE.eggshell)
  const navy = cssVar(s, '--fx-navy', AURA_PALETTE.navy)
  const sunset = cssVar(s, '--fx-sunset', AURA_PALETTE.sunset)
  const sage = cssVar(s, '--fx-sage', AURA_PALETTE.sage)
  const terra = cssVar(s, '--fx-terra', AURA_PALETTE.terra)
  const t = {
    floor: cssVar(s, '--fx-tone-50', egg),
    s150: cssVar(s, '--fx-tone-150', '#e2dfd1'),
    s200: cssVar(s, '--fx-tone-200', sunset),
    s400: cssVar(s, '--fx-tone-400', terra),
    s500: cssVar(s, '--fx-tone-500', '#5e6073'),
    s600: cssVar(s, '--fx-tone-600', '#4f5268'),
    ink: cssVar(s, '--fx-tone-900', navy),
    font: cssVar(s, '--fx-font', "'DM Sans', Arial, sans-serif"),
    mono: cssVar(s, '--fx-mono', "'IBM Plex Mono', ui-monospace, Menlo, monospace"),
  }
  // Primary: eggshell environment, navy ink (the original look).
  const light: Scheme = {
    gold: sunset, burst: terra, pedestal: t.s150, cap: t.s200,
    ink: t.ink, paper: t.floor, accent: t.s200, shadow: t.ink,
    p0: t.ink, p1: t.s400, p2: t.s200, p3: t.s600,
  }
  // Secondary: navy environment, every ink element inverted to eggshell.
  const dark: Scheme = {
    gold: sunset, burst: terra,
    pedestal: mixHexSrgb(navy, egg, 0.2), cap: sunset,
    ink: egg, paper: cssVar(s, '--fx-tone-800', '#34374e'), accent: sunset, shadow: '#14152a',
    p0: egg, p1: terra, p2: sunset, p3: sage,
  }
  // Vote feedback: sunset environment, navy ink (6.6:1).
  const feedback: Scheme = {
    gold: sunset, burst: terra,
    pedestal: mixHexSrgb(sunset, egg, 0.55), cap: terra,
    ink: navy, paper: egg, accent: terra, shadow: navy,
    p0: navy, p1: terra, p2: egg, p3: sage,
  }
  return { ...t, schemes: { light, dark, feedback } }
}

const toColors = (s: Scheme) => Object.fromEntries(SCHEME_KEYS.map(k => [k, new THREE.Color(s[k])])) as SchemeColors

/*
 * Duotone labels: a canvas is painted in pure red / green / blue, read as weights
 * for the paper / accent / ink colours, e.g. 'rgb(10,0,245)' = 4% paper + 96% ink.
 * Antialiased edges blend the weights, so they blend the colours too. The blend is
 * in linear light, so small paper weights already lighten a lot: 4% gives the old
 * --fx-tone-600 muted text (6.7:1 on the light panel, 9.7:1 inverted on dark).
 */
const W = { paper: '#ff0000', accent: '#00ff00', ink: '#0000ff', muted: 'rgb(10,0,245)', track: 'rgb(224,0,31)' }

/**
 * The split: a wavy centre line down the middle of the screen is the A | B
 * boundary. The golden fairy flare (after the loading screen) sweeps down that
 * line once per matchup and fades away; the line itself stays as the split.
 * The camera never yaws (it stays on the x = 0 plane), so world x < 0 is the left
 * half of the screen, and every themed fragment picks its side by comparing
 * gl_FragCoord.x with the ribbon's centre line at that height. One test keeps the
 * pedestals, labels, VS coin and the centred HUD text split on the same line (the
 * Aura world splits its sky / ground / mesas on the same points, see setSplitSchemes). `pts` are the ribbon's centre points in device px, GL y-up, sorted by y
 * (shared uniform array, updated in place when the path changes / on resize).
 */
const RIBBON_N = 24
type Split = { pts: THREE.Vector2[]; col: Record<Side, SchemeColors> }
const SPLIT_FN = `#define RIBBON_N ${RIBBON_N}
uniform vec2 ribbonPts[ RIBBON_N ];
// Points 0 .. N-2 are evenly spaced in y (bottom edge up to the tip under the nav);
// the last one continues the tip straight up. So the segment is found in O(1).
float auraSideAt( vec2 p ) {
  vec2 lo = ribbonPts[ 0 ], tip = ribbonPts[ RIBBON_N - 2 ];
  float x;
  if ( p.y >= tip.y ) x = tip.x;
  else if ( p.y <= lo.y ) x = lo.x;
  else {
    float k = ( p.y - lo.y ) / ( ( tip.y - lo.y ) / float( RIBBON_N - 2 ) );
    int i = clamp( int( floor( k ) ), 0, RIBBON_N - 3 );
    x = mix( ribbonPts[ i ].x, ribbonPts[ i + 1 ].x, clamp( k - float( i ), 0.0, 1.0 ) );
  }
  return step( x, p.x ); // 0 = A (left half), 1 = B (right half)
}
#define SIDE( a, b ) mix( ( a ), ( b ), auraSide )
`
const SIDE_MAIN = 'void main() {\n\tfloat auraSide = auraSideAt( gl_FragCoord.xy );'
const SPLIT_DUO = `#ifdef USE_MAP
  vec4 duoTex = texture2D( map, vMapUv );
  float duoSum = max( duoTex.r + duoTex.g + duoTex.b, 1e-4 );
  diffuseColor.rgb = ( duoTex.r * SIDE( duo0A, duo0B ) + duoTex.g * SIDE( duo1A, duo1B ) + duoTex.b * SIDE( duo2A, duo2B ) ) / duoSum;
  diffuseColor.a *= duoTex.a;
#endif`
type DuoKeys = [SchemeKey, SchemeKey, SchemeKey]
const LABEL_KEYS: DuoKeys = ['paper', 'accent', 'ink']

function splitMaterial<M extends THREE.Material>(mat: M, split: Split, opts: { color?: SchemeKey; duo?: DuoKeys }): M {
  const A = split.col.A, B = split.col.B
  mat.onBeforeCompile = shader => {
    const u = shader.uniforms
    u.ribbonPts = { value: split.pts }
    let head = SPLIT_FN
    let fs = shader.fragmentShader
    if (opts.color) {
      u.sideA = { value: A[opts.color] }; u.sideB = { value: B[opts.color] }
      head += 'uniform vec3 sideA;\nuniform vec3 sideB;\n'
      fs = fs
        .replace('vec4 diffuseColor = vec4( diffuse, opacity );', 'vec4 diffuseColor = vec4( SIDE( sideA, sideB ), opacity );')
        .replace('gl_FragColor = vec4( color, opacity', 'gl_FragColor = vec4( SIDE( sideA, sideB ), opacity') // ShadowMaterial
    }
    if (opts.duo) {
      opts.duo.forEach((k, i) => {
        u[`duo${i}A`] = { value: A[k] }; u[`duo${i}B`] = { value: B[k] }
        head += `uniform vec3 duo${i}A;\nuniform vec3 duo${i}B;\n`
      })
      fs = fs.replace('#include <map_fragment>', SPLIT_DUO)
    }
    shader.fragmentShader = fs.replace('void main() {', head + SIDE_MAIN)
  }
  mat.customProgramCacheKey = () => `aura-split-${opts.color ? 'c' : ''}${opts.duo ? 'd' : ''}`
  return mat
}

/*
 * The fairy flare (continues the loader's flare, see flareHandoff.ts): one sweep
 * down the A|B ribbon per trigger. The head draws the ribbon in behind it (the
 * split animation), then the ribbon dissolves into sparkles that scatter and
 * twinkle out last. Everything is screen space (CSS px, DPR-aware) and shares the
 * loader's colour stops: hot white -> sunset -> gold -> terra (transparent).
 * Blending is premultiplied and partly additive (GLOW_BLEND), so glows add light
 * on the pale backdrop without hard edges. When idle all flare meshes are hidden;
 * the split itself keeps using the ribbon's centre line (`split.pts`).
 */
type RibbonPoint = { x: number; y: number; w: number }
/** Sparkle pool size (stars + fine dust share one Points draw). */
const MOTE_MAX = 640
const DISSOLVE_S = 1.25
const TO_NDC = 'vec4( px.x / res.x * 2.0 - 1.0, 1.0 - px.y / res.y * 2.0, 0.0, 1.0 )'
type FlareUniforms = {
  res: { value: THREE.Vector2 }; headY: { value: number }; reveal: { value: number }; glow: { value: number }
  dpr: { value: number }; tailY: { value: number }; dissolve: { value: number }
}

/** Ribbon: each centre point is widened by ±w; it swells and brightens behind the head, tapers toward the tail, and dissolves on noise. */
function ribbonMaterial(u: FlareUniforms) {
  return new THREE.ShaderMaterial({
    uniforms: { ...FLARE_COLS(), res: u.res, headY: u.headY, reveal: u.reveal, glow: u.glow, tailY: u.tailY, dissolve: u.dissolve },
    vertexShader: `attribute float w;
uniform vec2 res;
uniform float headY;
uniform float tailY;
varying float vSide;
varying float vY;
void main() {
  float near = exp( -abs( position.y - headY ) / 70.0 );
  float q = clamp( ( position.y - tailY ) / max( 1.0, headY - tailY ), 0.0, 1.0 ); // 0 tail .. 1 head
  vec2 px = vec2( position.x + position.z * w * ( 0.45 + 0.55 * q ) * ( 1.0 + near * 1.6 ), position.y );
  vSide = position.z;
  vY = px.y;
  gl_Position = ${TO_NDC};
}`,
    fragmentShader: `${FLARE_STOPS}
uniform float headY;
uniform float tailY;
uniform float reveal;
uniform float glow;
uniform float dissolve;
varying float vSide;
varying float vY;
float fh( vec2 p ) { return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 ); }
float fn( vec2 p ) {
  vec2 i = floor( p ), f = fract( p ); f = f * f * ( 3.0 - 2.0 * f );
  return mix( mix( fh( i ), fh( i + vec2( 1, 0 ) ), f.x ), mix( fh( i + vec2( 0, 1 ) ), fh( i + vec2( 1, 1 ) ), f.x ), f.y );
}
void main() {
  if ( vY > reveal || vY > headY + 1.0 ) discard; // the head draws the ribbon in behind it
  float r = abs( vSide );
  vec4 c = flareStops( r, 0.28, 0.6 );
  float hot = exp( -max( 0.0, headY - vY ) / 180.0 );
  c.rgb = mix( c.rgb, cHot, hot * 0.5 * ( 1.0 - r ) );
  float q = clamp( ( vY - tailY ) / max( 1.0, headY - tailY ), 0.0, 1.0 );
  c.a *= clamp( 0.8 + 0.2 * hot + 0.35 * glow, 0.0, 1.0 ) * ( 0.75 + 0.25 * q );
  // Dissolve: noise patches burn away with a soft hot rim (sparkles spawn on the CPU side).
  float n = 0.65 * fn( vec2( vY * 0.035, vSide * 1.4 + 3.1 ) ) + 0.35 * fn( vec2( vY * 0.11, vSide * 3.0 ) );
  float keep = smoothstep( dissolve - 0.12, dissolve, n );
  float rim = keep * ( 1.0 - keep ) * 4.0 * step( 0.001, dissolve );
  c.rgb = mix( c.rgb, cHot, rim * 0.7 );
  c.a *= keep;
  if ( c.a < 0.003 ) discard;
  gl_FragColor = c;
  #include <colorspace_fragment>
  ${premultiply(1)}
}`,
    side: THREE.DoubleSide,
    ...GLOW_BLEND,
  })
}

/** Head: soft halo + four-point twinkle star, on a quad centred at `center` (CSS px). */
function flareHeadMaterial(u: FlareUniforms) {
  return new THREE.ShaderMaterial({
    uniforms: { ...FLARE_COLS(), res: u.res, center: { value: new THREE.Vector2() }, radius: { value: 30 }, star: { value: 0.35 }, alpha: { value: 1 } },
    vertexShader: `uniform vec2 res;
uniform vec2 center;
uniform float radius;
varying vec2 vP;
void main() {
  vP = position.xy * 2.0;
  vec2 px = center + vec2( position.x, -position.y ) * 2.0 * radius;
  gl_Position = ${TO_NDC};
}`,
    fragmentShader: `${FLARE_STOPS}
uniform float star;
uniform float alpha;
varying vec2 vP;
void main() {
  float r = length( vP );
  vec4 c = flareStops( min( r, 1.0 ), 0.18, 0.5 ) * vec4( 1.0, 1.0, 1.0, 1.0 - smoothstep( 0.92, 1.0, r ) );
  vec2 q = abs( vP ) / star; // astroid: the loader's quadratic-curve twinkle
  float s = smoothstep( 1.0, 0.82, sqrt( q.x ) + sqrt( q.y ) );
  c = vec4( mix( c.rgb, cHot, s ), max( c.a, s * 0.95 ) );
  c.a *= alpha;
  if ( c.a < 0.003 ) discard;
  gl_FragColor = c;
  #include <colorspace_fragment>
  ${premultiply(0.8)}
}`,
    ...GLOW_BLEND,
  })
}

/** Sparkles: four-point glints with soft cross rays + fine round dust, one Points draw. */
function moteMaterial(u: FlareUniforms) {
  return new THREE.ShaderMaterial({
    uniforms: { ...FLARE_COLS(), res: u.res, dpr: u.dpr },
    vertexShader: `attribute float msize;
attribute float malpha;
attribute float mstar;
attribute vec3 mcol;
uniform vec2 res;
uniform float dpr;
varying vec3 vCol;
varying float vA;
varying float vStar;
void main() {
  vCol = mcol; vA = malpha; vStar = mstar;
  vec2 px = position.xy;
  gl_Position = ${TO_NDC};
  gl_PointSize = msize * dpr;
}`,
    fragmentShader: flarePointFrag(0.95), // shared with the robot flares (flareShared.ts)
    ...GLOW_BLEND,
  })
}

// Robot colours match GeneratedG1RobotPreview (glazed ivory body, taupe joints, ink outline).
const ROBOT = { body: 0xf2e9da, joint: 0xa8957a, outline: 0x17130e }

const easeOutCubic = (x: number) => 1 - Math.pow(1 - x, 3)
const easeInCubic = (x: number) => x * x * x
const easeOutBack = (x: number) => { const c1 = 1.4, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2) }
const TONE_SECONDS = 0.35
/** Question label type sizes (canvas px on a 1400-wide label): question line(s) and the keycap hint line. */
const QUESTION_TYPE = { wide: { size: 56, lines: 1, hint: 38 }, compact: { size: 76, lines: 2, hint: 56 } } as const
/** Orbit limits: yaw well under 90° keeps lane A on the left half; pitch keeps the camera above the ground. */
const ORBIT_YAW_MAX = THREE.MathUtils.degToRad(38)
const ORBIT_PITCH_MIN = -0.1
const ORBIT_PITCH_MAX = 0.3
const FEEDBACK_HOLD_SECONDS = 1.1


type Tween = { t: number; dur: number; ease: (x: number) => number; update: (k: number) => void; resolve: () => void }

class CanvasLabel {
  readonly canvas = document.createElement('canvas')
  readonly ctx: CanvasRenderingContext2D
  readonly texture: THREE.CanvasTexture
  /** Labels hold duotone channel weights (see splitMaterial), so the texture is raw data, not sRGB. */
  constructor(readonly w: number, readonly h: number) {
    this.canvas.width = w; this.canvas.height = h
    this.ctx = this.canvas.getContext('2d')!
    this.texture = new THREE.CanvasTexture(this.canvas)
    this.texture.colorSpace = THREE.NoColorSpace
    this.texture.anisotropy = 4
  }
  draw(fn: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
    this.ctx.clearRect(0, 0, this.w, this.h)
    fn(this.ctx, this.w, this.h)
    this.texture.needsUpdate = true
  }
}

function labelPlane(label: CanvasLabel, width: number, split: Split, opts: { hud?: boolean; keys?: DuoKeys } = {}) {
  const mat = splitMaterial(new THREE.MeshBasicMaterial({
    map: label.texture, transparent: true, depthWrite: false, toneMapped: false,
    depthTest: !opts.hud,
  }), split, { duo: opts.keys ?? LABEL_KEYS })
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat)
  mesh.scale.set(width, width * label.h / label.w, 1)
  mesh.raycast = () => {}
  if (opts.hud) mesh.renderOrder = 1000
  return mesh
}

/*
 * VS coin: the original navy coin with a gold rim and light italic "VS", rebuilt
 * as a pedestal turned to face the viewer. Tapered drum (face 'shadow': navy, near
 * black-navy on the dark tone), thin 'gold' rim ring, extruded italic VS ('gold')
 * over a 'burst' drop extrusion, and an inverted-hull 'ink' outline standing in for
 * the composite's ink line (navy on light, eggshell on dark). Lit like the
 * pedestals and coloured through splitMaterial, so it splits on the ribbon and
 * follows each half's tone. It draws on the overlay layer (after the composite, so
 * it stays above the ribbon and under the HUD); hence transparent materials, so
 * renderOrder places it among the flare's layers (950 .. 950.4).
 * Unit space: face radius VS_R, +z toward the camera.
 */
const VS_R = 0.4
const VS_DEPTH = 0.12

/** Arc points (degrees), x squashed/stretched by `sx`. */
function arcPts(cx: number, cy: number, r: number, a0: number, a1: number, sx = 1, skipFirst = false) {
  const n = 20, out: THREE.Vector2[] = []
  for (let i = skipFirst ? 1 : 0; i <= n; i++) {
    const a = THREE.MathUtils.degToRad(THREE.MathUtils.lerp(a0, a1, i / n))
    out.push(new THREE.Vector2(cx + Math.cos(a) * r * sx, cy + Math.sin(a) * r))
  }
  return out
}

/** Bold italic "VS" outlines (cap height h, stroke t), one closed polygon per letter. */
function vsLetterShapes(h: number, t: number) {
  const hh = h / 2, gap = 0.135, hw = 0.12
  const v = [[-hw, hh], [-hw + t * 1.05, hh], [0, -hh + t * 1.45], [hw - t * 1.05, hh], [hw, hh], [t * 0.6, -hh], [-t * 0.6, -hh]]
    .map(([x, y]) => new THREE.Vector2(x - gap, y))
  // S: two stacked bowls traced as one outline (top bowl CCW, bottom bowl CW), so there is no seam at the joint.
  const rc = (h - t) / 4, ro = rc + t / 2, ri = rc - t / 2, sx = 1.4
  const s = [
    ...arcPts(gap, rc, ro, 28, 270, sx),
    ...arcPts(gap, -rc, ri, 90, -152, sx, true),
    ...arcPts(gap, -rc, ro, -152, 90, sx),
    ...arcPts(gap, rc, ri, 270, 28, sx, true),
  ]
  return [v, s].map(pts => new THREE.Shape(pts.map(p => new THREE.Vector2(p.x + 0.2 * p.y, p.y))))
}

function buildVsCoin(split: Split) {
  const g = new THREE.Group()
  const lit = (key: SchemeKey) =>
    splitMaterial(new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0, transparent: true }), split, { color: key })
  const add = (geo: THREE.BufferGeometry, mat: THREE.Material, order: number, z = 0) => {
    const m = new THREE.Mesh(geo, mat)
    m.position.z = z; m.renderOrder = order; m.raycast = () => {}
    g.add(m)
  }
  const face = VS_DEPTH / 2, taper = 0.64 / 0.6 // the pedestal's taper
  const drum = (grow: number) => new THREE.CylinderGeometry(VS_R + grow, VS_R * taper + grow, VS_DEPTH + grow * 2, 96).rotateX(Math.PI / 2)
  add(drum(0.016), splitMaterial(new THREE.MeshBasicMaterial({ transparent: true, side: THREE.BackSide, toneMapped: false }), split, { color: 'ink' }), 950) // ink outline (inverted hull)
  add(drum(0), lit('shadow'), 950.1)
  add(new THREE.RingGeometry(VS_R * 0.88, VS_R * 0.95, 96), lit('gold'), 950.2, face + 0.002) // gold rim ring
  const shapes = vsLetterShapes(0.29, 0.08)
  const drop = new THREE.ExtrudeGeometry(shapes, { depth: 0.012, curveSegments: 1, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.01, bevelOffset: 0.006, bevelSegments: 1 })
  drop.translate(0.014, -0.016, face + 0.006) // drop extrusion under the letters, offset down-right
  add(drop, lit('burst'), 950.3)
  const letters = new THREE.ExtrudeGeometry(shapes, { depth: 0.03, curveSegments: 1, bevelEnabled: true, bevelThickness: 0.012, bevelSize: 0.007, bevelSegments: 2 })
  letters.translate(0, 0, face + 0.024)
  add(letters, lit('gold'), 950.4)
  return g
}

type Lane = {
  side: Side
  root: THREE.Group
  lift: THREE.Group
  actor: ArenaActor
  /** The lane's current clip (kept so a character swap can hand it to the new body). */
  clip: PreparedClip | null
  hit: THREE.Mesh
  progress: THREE.Mesh
  progressSegs: number
  badge: THREE.Sprite
  caption: CanvasLabel
  captionMesh: THREE.Mesh
  spot: THREE.SpotLight
  label: string
  // animated state
  liftV: number; dimV: number; spotV: number; scaleV: number; hoverV: number
}

/** `wCol` / `wFrom`: the same tween for the Aura world's colours (sky, clouds, mesas, ground, ink) on this half. */
type SideScheme = { tone: Tone; t: number; col: SchemeColors; from: SchemeColors; hold: number; wCol: AuraWorldScheme; wFrom: AuraWorldScheme }

const HUD_PX = { pad: 16, panelH: 64 }
// The hero is the first screen and the fixed nav floats over it (bottom edge at 92px
// desktop, 72px under 768px), so the top HUD panels start below that band.
const navClearPx = (width: number) => (width < 768 ? 84 : 104)

export class HeroArenaScene {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(34, 1, 0.05, 220) // far covers the Aura world's distant mesas
  private theme = readTheme()
  // Colour scheme state, one per half. `col` holds the live colours: every themed
  // material, the backdrop and the label uniforms reference these Color objects,
  // so a scheme tween only lerps ~15 colours per side.
  private schemes: Record<Tone, SchemeColors> = {
    light: toColors(this.theme.schemes.light), dark: toColors(this.theme.schemes.dark), feedback: toColors(this.theme.schemes.feedback),
  }
  private sides: Record<Side, SideScheme> = {
    A: { tone: 'light', t: 1, col: toColors(this.theme.schemes.light), from: toColors(this.theme.schemes.light), hold: 0, wCol: auraWorldScheme('light'), wFrom: auraWorldScheme('light') },
    B: { tone: 'light', t: 1, col: toColors(this.theme.schemes.light), from: toColors(this.theme.schemes.light), hold: 0, wCol: auraWorldScheme('light'), wFrom: auraWorldScheme('light') },
  }
  // The arena lives inside the shared Aura world; its colours are tweened per half like the scheme above.
  private worldSchemes: Record<Tone, AuraWorldScheme> = { light: auraWorldScheme('light'), dark: auraWorldScheme('dark'), feedback: auraWorldScheme('feedback') }
  private world: AuraWorld
  private worldSplit: AuraWorldSplit
  private lastDt = 0
  // Adaptive pixel ratio: steps down (to 1.25 at most) if frames stay slower than 50 fps for 2 s.
  private frameEma = 1 / 60
  private slowFor = 0
  /** One fairy flare per lane riding the robot's fastest limb (robotFlare.ts); none under reduced motion. */
  private robotFlares: RobotFlare[] = []
  /** Painted-character pass on the G1 fallback bodies (toonStylize.ts); the fairy gets it in createFairyMaterial. */
  private stylized: ToonStylizeHandle[] = []
  private holdSeq = 0 // vote token; a side's `hold` is the token of the vote showing feedback there (0 = none)
  private split: Split = { pts: Array.from({ length: RIBBON_N }, () => new THREE.Vector2()), col: { A: this.sides.A.col, B: this.sides.B.col } }
  // Fairy flare state (ribbon divider + head + dust).
  private ribbon: RibbonPoint[] = []
  private ribbonGeo = new THREE.BufferGeometry()
  private flareU: FlareUniforms = { res: { value: new THREE.Vector2(1, 1) }, headY: { value: -1e4 }, reveal: { value: 1e6 }, glow: { value: 0 }, dpr: { value: 1 }, tailY: { value: 0 }, dissolve: { value: 0 } }
  private flareHead!: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private ribbonMesh!: THREE.Mesh
  private motesMesh!: THREE.Points
  private motesGeo = new THREE.BufferGeometry()
  private motesAttrs: THREE.BufferAttribute[] = [] // position, msize, malpha, mstar, mcol (set in buildStage)
  // Sparkle pool (no per-frame allocation): position, velocity, life, size, kind (0 dust / 1 glint), twinkle phase + rate, colour index.
  private mx = new Float32Array(MOTE_MAX); private my = new Float32Array(MOTE_MAX)
  private mvx = new Float32Array(MOTE_MAX); private mvy = new Float32Array(MOTE_MAX)
  private mlife = new Float32Array(MOTE_MAX); private mmax = new Float32Array(MOTE_MAX)
  private msize = new Float32Array(MOTE_MAX); private mkind = new Uint8Array(MOTE_MAX)
  private mphase = new Float32Array(MOTE_MAX); private mrate = new Float32Array(MOTE_MAX)
  private mcol = new Uint8Array(MOTE_MAX)
  private moteCount = 0
  private flyY = 0 // head position (CSS px) along the ribbon
  private flying = false
  private sweepStart = 0 // rAF ms
  private sweepFromLoader = false
  private dissolveStart = -1 // rAF ms, -1 = not dissolving
  private sparkleAcc = 0 // fractional spawn carry so rates are frame-rate independent
  private dustAcc = 0
  // Loader -> hero flare handoff (flareHandoff.ts).
  private awaitHandoff = false
  private handoffSwept = false
  private offTarget: () => void = () => {}
  private offHandoff: () => void = () => {}
  private vsPunch = 0
  private vsBody: THREE.Group // 3D VS coin (see buildVsCoin), scaled from vsPx in layoutHud
  private vsFlip = 0 // remaining coin-flip angle (rad), set on a vote, eases back to 0
  private lanes: Lane[] = []
  private tweens: Tween[] = []
  private raycaster = new THREE.Raycaster()
  private pointer = new THREE.Vector2()
  private hovered: Side | null = null
  private focused: Side | null = null
  private raf = 0
  private last = 0
  private clock = 0
  private matchTime = 0
  private playing = false
  private active = false
  private disposed = false
  private ready = false
  private dirty = true
  private width = 1
  private height = 1
  private laneX = 1.15
  private camDist = 7
  private camTarget = new THREE.Vector3(0, 0.8, 0)
  /**
   * Drag-to-orbit (replaces hover-to-highlight / tap-to-vote). Damped yaw/pitch around the arena
   * centre; yaw is clamped so lane A stays on the left half and B on the right (the camera always
   * looks at the arena centre, so the screen-centre A|B split stays valid). Touch: horizontal drags
   * orbit (yaw only), vertical swipes scroll the page (canvas has touch-action: pan-y). No zoom.
   */
  private orbit = { yaw: 0, pitch: 0, tYaw: 0, tPitch: 0, dragging: false, pid: -1, lx: 0, ly: 0, sx: 0, sy: 0, moved: false, touch: false, idle: 99 }
  /** A brief navy flash on the side a key vote picked (HeroArena calls flashSide before voting). */
  private flash: Side | null = null
  private flashTimer = 0
  private coarsePointer = typeof window !== 'undefined' && !!window.matchMedia?.('(hover: none) and (pointer: coarse)').matches
  private hudLeft = new CanvasLabel(600, 128)
  private hudRight = new CanvasLabel(440, 128)
  // Tall enough for two lines on narrow screens; text is bottom-aligned so the
  // plane's bottom edge stays just above the DOM vote controls.
  private hudQuestion = new CanvasLabel(1400, 300)
  private questionCompact = false
  private hudLeftMesh: THREE.Mesh
  private hudRightMesh: THREE.Mesh
  private hudQuestionMesh: THREE.Mesh
  private hud: HudState = { round: 1, xp: 0, level: 1, levelProgress: 0, streak: 0, playing: false }
  private vs = new THREE.Group() // camera-attached VS coin, centred on the ribbon
  private playSprite: THREE.Sprite
  private xpPop: THREE.Sprite
  private xpLabel = new CanvasLabel(256, 96)
  private xpPopV = 0
  private xpPopY = 0
  private particles: THREE.Points
  private particleVel: Float32Array
  private particleAge = 99
  private observer: ResizeObserver
  private marble: THREE.Texture | null = null
  private geometries: THREE.BufferGeometry[] = []
  /** Character the lanes show, or are loading (setCharacter). */
  private characterId: string | null = null
  /** Bumped by every setCharacter call: a load that finishes after a newer call is dropped. */
  private bodyToken = 0
  private host: HTMLElement

  constructor(host: HTMLElement, private opts: { reducedMotion: boolean; modelBase: string; callbacks: ArenaCallbacks }) {
    this.host = host
    // Throws if WebGL is unavailable; the caller renders the static fallback.
    // No MSAA on the canvas: the world composite renders the scene into its own
    // target and only a full-screen quad + the overlay land here (a 4x canvas
    // resolve at 2880x1800 was a large part of the DPR-2 cost).
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' })
    if (!this.renderer.getContext()) throw new Error('WebGL unavailable')
    const r = this.renderer
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    r.outputColorSpace = THREE.SRGBColorSpace
    r.shadowMap.enabled = true
    r.shadowMap.type = THREE.PCFSoftShadowMap
    r.toneMapping = THREE.ACESFilmicToneMapping
    r.toneMappingExposure = 1.06
    r.domElement.style.display = 'block'
    r.domElement.style.width = '100%'
    r.domElement.style.height = '100%'
    r.domElement.style.touchAction = 'pan-y'
    host.appendChild(r.domElement)
    r.domElement.addEventListener('webglcontextlost', this.onContextLost)

    // Environment: the shared Aura world (sky dome, posterized clouds, ground, mesas,
    // ink-outline composite), same as the motion viewers. Robots and pedestals go
    // through its ink/grade pass; it takes each half's scheme via setSplitSchemes,
    // split on the ribbon's centre line (the same points the hero's materials use).
    this.world = createAuraWorld(this.scene, r, { scale: 1, groundY: 0, ring: false, split: true, reducedMotion: opts.reducedMotion, msaa: 'auto',
      // the camera looks down -z: keep near mesas out of the robots' backdrop (portrait FOV is ~+-10 deg)
      sightline: { yaw: -Math.PI / 2, halfAngle: THREE.MathUtils.degToRad(24) },
      // quiet interactive props outside the arena disc; taller ones stay out of the camera-front sector,
      // and the pointer is ignored while it is over a robot or the play button (those clicks vote / play)
      props: { density: 1, keepOut: 3.45, clearFront: true, pointerBlocked: (x, y) => this.pointerOnUi(x, y) } })
    this.worldSplit = { a: this.sides.A.wCol, b: this.sides.B.wCol, pts: this.split.pts }
    this.scene.add(this.camera)
    // UI (HUD, badges, captions, VS, flare, XP pop, confetti) lives on the overlay layer:
    // drawn after the composite, never inked or tone-mapped. The world only runs its
    // overlay pass when a visible scene child is on that layer, so keep one there.
    const overlayRoot = new THREE.Object3D()
    overlayRoot.layers.set(AURA_OVERLAY_LAYER)
    this.scene.add(overlayRoot)
    this.raycaster.layers.enable(AURA_OVERLAY_LAYER) // the play sprite is an overlay

    this.buildStage()
    this.playSprite = this.buildPlaySprite()
    this.xpPop = new THREE.Sprite(splitMaterial(new THREE.SpriteMaterial({ map: this.xpLabel.texture, transparent: true, depthWrite: false, toneMapped: false, opacity: 0 }), this.split, { duo: LABEL_KEYS }))
    this.xpPop.scale.set(0.5, 0.1875, 1); this.xpPop.visible = false; this.xpPop.renderOrder = 10
    this.scene.add(this.xpPop)
    const pt = this.buildParticles(); this.particles = pt.points; this.particleVel = pt.vel

    this.hudLeftMesh = labelPlane(this.hudLeft, 1, this.split, { hud: true })
    this.hudRightMesh = labelPlane(this.hudRight, 1, this.split, { hud: true })
    this.hudQuestionMesh = labelPlane(this.hudQuestion, 1, this.split, { hud: true })
    this.camera.add(this.hudLeftMesh, this.hudRightMesh, this.hudQuestionMesh)
    this.vsBody = buildVsCoin(this.split)
    this.vsBody.rotation.x = 0.2 // lean back a little, so the coin reads like the pedestals seen from above
    this.vs.add(this.vsBody)
    this.camera.add(this.vs)
    const overlay: THREE.Object3D[] = [this.hudLeftMesh, this.hudRightMesh, this.hudQuestionMesh, ...this.vsBody.children, this.xpPop, this.particles, this.playSprite,
      this.ribbonMesh, this.motesMesh, this.flareHead, ...this.lanes.flatMap(l => [l.badge, l.captionMesh])]
    overlay.forEach(o => o.layers.set(AURA_OVERLAY_LAYER))
    this.setPath(false)
    // If the loading screen's flare is running, our first sweep is its continuation.
    this.awaitHandoff = !opts.reducedMotion && isLoaderFlareRunning()
    this.offTarget = registerFlareTarget(() => {
      if (!this.awaitHandoff || this.disposed || this.ribbon.length !== RIBBON_N) return null
      const rect = this.renderer.domElement.getBoundingClientRect()
      const top = this.sweepTop(), vy = FLARE_SWEEP_SLOPE * (this.height + 60 - top) / FLARE_SWEEP_SECONDS
      const x = this.ribbonX(top)
      return { x: rect.left + x, y: rect.top + top, vx: ((this.ribbonX(top + 4) - x) / 4) * vy, vy, radius: this.headRadius() }
    })
    this.offHandoff = onFlareHandoff(t => {
      if (!this.awaitHandoff || this.disposed) return
      this.awaitHandoff = false
      const hasMatchup = this.lanes.some(l => l.label)
      if (t === null) { if (hasMatchup) this.setPath(true) ; return } // loader ended without us: own sweep
      this.handoffSwept = !hasMatchup // the first matchup must not restart it
      this.startSweep(t, true)
    })

    this.observer = new ResizeObserver(() => this.resize())
    this.observer.observe(host)
    this.resize()
    this.drawHud()
    document.fonts?.ready.then(() => { if (!this.disposed) { this.redrawText(); this.dirty = true } }).catch(() => {})

    const el = r.domElement
    el.style.cursor = 'grab'
    el.addEventListener('pointerdown', this.onPointerDown)
    el.addEventListener('pointermove', this.onPointerMove)
    el.addEventListener('pointerleave', this.onPointerLeave)
    el.addEventListener('pointerup', this.onPointerUp)
    el.addEventListener('pointercancel', this.onPointerUp)
    el.addEventListener('click', this.onClick)
  }

  // ------------------------------------------------------------------ build
  /** Unlit material whose colour follows the side it is drawn on. */
  private envMat = (key: SchemeKey, params: THREE.MeshBasicMaterialParameters) =>
    splitMaterial(new THREE.MeshBasicMaterial(params), this.split, { color: key })

  private buildStage() {
    const t = this.theme
    const hemi = new THREE.HemisphereLight(0xfff4e4, new THREE.Color(t.s400), 1.55)
    const key = new THREE.DirectionalLight(0xfff0dc, 3.1)
    key.position.set(-2.6, 6.2, 4.6)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    const sc = key.shadow.camera
    sc.left = -3.6; sc.right = 3.6; sc.top = 3.6; sc.bottom = -3.6; sc.near = 1; sc.far = 16
    key.shadow.bias = -0.0005; key.shadow.normalBias = 0.02
    const fill = new THREE.DirectionalLight(0xd8c7ad, 1.35); fill.position.set(4, 2.5, 3)
    const rim = new THREE.DirectionalLight(0xcdb898, 1.7); rim.position.set(0, 3.2, -4.5)
    this.scene.add(hemi, key, fill, rim)
    // The same rig also lights the overlay pass: the 3D VS medallion lives there and is shaded like the pedestals.
    for (const l of [hemi, key, fill, rim]) l.layers.enable(AURA_OVERLAY_LAYER)

    // No hero floor: the world's ground is the floor (it takes the key light's cast shadow).

    // Fairy flare down the split line: ribbon trail, dust motes, head (transient, hidden when idle).
    const n = RIBBON_N
    this.ribbonGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 2 * 3), 3))
    this.ribbonGeo.setAttribute('w', new THREE.BufferAttribute(new Float32Array(n * 2), 1))
    const idx: number[] = []
    for (let i = 0; i < n - 1; i++) { const a = i * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3) }
    this.ribbonGeo.setIndex(idx)
    const ribbon = new THREE.Mesh(this.ribbonGeo, ribbonMaterial(this.flareU))
    ribbon.renderOrder = 900
    this.motesGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MOTE_MAX * 3), 3))
    this.motesGeo.setAttribute('msize', new THREE.BufferAttribute(new Float32Array(MOTE_MAX), 1))
    this.motesGeo.setAttribute('malpha', new THREE.BufferAttribute(new Float32Array(MOTE_MAX), 1))
    this.motesGeo.setAttribute('mstar', new THREE.BufferAttribute(new Float32Array(MOTE_MAX), 1))
    this.motesGeo.setAttribute('mcol', new THREE.BufferAttribute(new Float32Array(MOTE_MAX * 3), 3))
    this.motesGeo.setDrawRange(0, 0)
    this.motesAttrs = ['position', 'msize', 'malpha', 'mstar', 'mcol'].map(n => this.motesGeo.getAttribute(n) as THREE.BufferAttribute)
    const motes = new THREE.Points(this.motesGeo, moteMaterial(this.flareU))
    motes.renderOrder = 952
    this.flareHead = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), flareHeadMaterial(this.flareU))
    this.flareHead.renderOrder = 955
    for (const m of [ribbon, motes, this.flareHead]) { m.frustumCulled = false; m.raycast = () => {}; m.visible = false; this.scene.add(m) }
    this.ribbonMesh = ribbon; this.motesMesh = motes

    // Lanes.
    for (const side of ['A', 'B'] as Side[]) this.lanes.push(this.buildLane(side))
  }

  private buildLane(side: Side): Lane {
    const t = this.theme, env = this.envMat
    const root = new THREE.Group()
    const lift = new THREE.Group()
    root.add(lift)
    this.scene.add(root)

    const pedestalH = 0.2
    const pedestalMat = splitMaterial(new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0 }), this.split, { color: 'pedestal' })
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.64, pedestalH, 96), pedestalMat)
    pedestal.position.y = pedestalH / 2; pedestal.castShadow = true; pedestal.receiveShadow = true; pedestal.raycast = () => {}
    const cap = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.6, 96), env('cap', { transparent: true, opacity: 0.8, depthWrite: false }))
    cap.rotation.x = -Math.PI / 2; cap.position.y = pedestalH + 0.002; cap.raycast = () => {}
    lift.add(pedestal, cap)

    // Progress: faint track + filled arc on the pedestal top edge.
    const segs = 128
    const track = new THREE.Mesh(new THREE.RingGeometry(0.535, 0.565, segs, 1), env('ink', { transparent: true, opacity: 0.12, depthWrite: false }))
    const progress = new THREE.Mesh(new THREE.RingGeometry(0.535, 0.565, segs, 1), env('ink', { transparent: true, opacity: 0.85, depthWrite: false, toneMapped: false }))
    for (const m of [track, progress]) { m.rotation.set(-Math.PI / 2, 0, -Math.PI / 2); m.position.y = pedestalH + 0.004; m.raycast = () => {} }
    progress.geometry.setDrawRange(0, 0)
    lift.add(track, progress)

    // Caption on the pedestal front (the clip's short label).
    const caption = new CanvasLabel(768, 96)
    const captionMesh = labelPlane(caption, 0.86, this.split)
    captionMesh.position.set(0, pedestalH / 2, 0.645)
    lift.add(captionMesh)

    // Robot.
    const actorHolder = new THREE.Group(); actorHolder.position.y = pedestalH
    lift.add(actorHolder)
    const actor = this.placeholderActor()
    actorHolder.add(actor.group)

    // Lane badge above the robot.
    const badgeLabel = new CanvasLabel(192, 192)
    badgeLabel.draw((ctx, w, h) => {
      ctx.fillStyle = W.ink
      ctx.beginPath(); ctx.arc(w / 2, h / 2, w / 2 - 4, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = W.paper; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.font = `700 104px ${t.font}`
      ctx.fillText(side, w / 2, h / 2 + 6)
    })
    const badge = new THREE.Sprite(splitMaterial(new THREE.SpriteMaterial({ map: badgeLabel.texture, transparent: true, depthWrite: false, toneMapped: false }), this.split, { duo: LABEL_KEYS }))
    badge.scale.set(0.24, 0.24, 1); badge.position.y = pedestalH + 1.72; badge.raycast = () => {}
    lift.add(badge)

    // Invisible hit box: clicking anywhere on the lane votes.
    const hit = new THREE.Mesh(new THREE.BoxGeometry(1.35, 2.15, 1.35), new THREE.MeshBasicMaterial({ visible: false }))
    hit.position.y = 1.05; hit.userData.side = side
    root.add(hit)

    const spot = new THREE.SpotLight(0xfff3df, 0, 0, 0.36, 0.55, 0)
    spot.position.set(0, 5, 1.4)
    spot.target = lift
    root.add(spot)

    return { side, root, lift, actor, clip: null, hit, progress, progressSegs: segs, badge, caption, captionMesh, spot, label: '', liftV: 0, dimV: 0, spotV: 0, scaleV: 0, hoverV: 0 }
  }

  private placeholderActor() {
    // Real actors are swapped in by init(); this empty one keeps types simple.
    return new G1Actor({ transforms: new Map(), geometries: new Map() }, ROBOT, null, 0)
  }

  private buildPlaySprite() {
    const label = new CanvasLabel(256, 256)
    label.draw((ctx, w, h) => {
      ctx.fillStyle = W.ink
      ctx.beginPath(); ctx.arc(w / 2, h / 2, w / 2 - 4, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = W.paper
      ctx.beginPath(); ctx.moveTo(w * 0.41, h * 0.31); ctx.lineTo(w * 0.71, h * 0.5); ctx.lineTo(w * 0.41, h * 0.69); ctx.closePath(); ctx.fill()
    })
    const sprite = new THREE.Sprite(splitMaterial(new THREE.SpriteMaterial({ map: label.texture, transparent: true, depthWrite: false, depthTest: false, toneMapped: false }), this.split, { duo: LABEL_KEYS }))
    sprite.scale.set(0.42, 0.42, 1); sprite.position.set(0, 1.0, 0.7); sprite.renderOrder = 960
    sprite.visible = false
    sprite.userData.play = true
    this.scene.add(sprite)
    return sprite
  }

  private buildParticles() {
    const n = 90
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true, opacity: 0, depthWrite: false, toneMapped: false }))
    points.visible = false; points.frustumCulled = false; points.raycast = () => {}
    this.scene.add(points)
    return { points, vel: new Float32Array(n * 3) }
  }

  /** Builds both robots for the selected character (three/characters.ts). */
  init(def: CharacterDef = getCurrentCharacter()) {
    return this.setCharacter(def)
  }

  /**
   * Shows `def` on both lanes, live: loads its body, scales the current robots out,
   * swaps them (old actors, toonStylize handles, robot flares and, when leaving the G1,
   * its STL geometries and marble are disposed), hands each lane its clip back (time
   * comes from matchTime, look from the tick) and scales them in. Instant under reduced
   * motion or while the arena is off screen. Overlapping calls: the latest wins.
   */
  async setCharacter(def: CharacterDef) {
    if (this.disposed || def.id === this.characterId) return
    this.characterId = def.id
    const token = ++this.bodyToken
    const root = this.opts.modelBase.replace(/models\/g1-native\/$/, '')
    let built: { make: () => ArenaActor; geometries: THREE.BufferGeometry[]; marble: THREE.Texture | null }
    try {
      built = await this.loadBody(def, root)
    } catch (err) {
      if (token === this.bodyToken) this.characterId = null // let a later call retry this character
      throw err
    }

    const stale = () => this.disposed || token !== this.bodyToken
    const release = () => { built.geometries.forEach(g => g.dispose()); built.marble?.dispose() }
    if (stale()) { release(); return }

    const animate = this.ready && this.active && !this.opts.reducedMotion
    const shown = this.lanes.map(l => (l.clip ? 1 : 0))
    if (animate) {
      const from = this.lanes.map(l => l.scaleV)
      await this.tween(0.22, k => this.lanes.forEach((l, i) => { l.scaleV = from[i] * (1 - k) }), easeInCubic)
      if (stale()) { release(); return }
    }

    // Swap: handles and flares first (they hold the old bodies), then the actors, then shared buffers.
    this.robotFlares.forEach(f => f.dispose()); this.robotFlares = []
    this.stylized.forEach(h => h.dispose()); this.stylized = []
    for (const lane of this.lanes) {
      const holder = lane.actor.group.parent!
      lane.actor.dispose()
      lane.actor = built.make()
      lane.actor.group.visible = false
      lane.actor.group.scale.setScalar(0.0001)
      holder.add(lane.actor.group)
      if (lane.clip) lane.actor.setClip(lane.clip) // visible again; pose follows matchTime in the tick
      // G1: paint into the world's palette (toonStylize.ts); mixamo bodies get it in createFairyMaterial.
      if (def.kind === 'g1-rigid') this.stylized.push(toonStylize(lane.actor.group))
    }
    this.geometries.forEach(g => g.dispose()); this.marble?.dispose()
    this.geometries = built.geometries; this.marble = built.marble
    if (!this.opts.reducedMotion) {
      this.robotFlares = this.lanes.map(lane => {
        // hands, feet and head; subtle, so the motion itself stays what's judged
        const flare = new RobotFlare(effectorsByName(lane.actor.group, def.effectors), { scale: 1, strength: 0.7, layer: AURA_OVERLAY_LAYER })
        this.scene.add(flare.points)
        return flare
      })
    }
    const first = !this.ready
    this.ready = true
    this.dirty = true
    if (first) return // setMatchup animates the first robots in
    if (animate) await this.tween(0.42, k => this.lanes.forEach((l, i) => { l.scaleV = shown[i] * k }), easeOutBack)
    else this.lanes.forEach((l, i) => { l.scaleV = Math.max(l.scaleV, shown[i]) })
    this.dirty = true
  }

  /** Downloads / clones what `def` needs; `make` builds one lane actor from it. */
  private async loadBody(def: CharacterDef, root: string): Promise<{ make: () => ArenaActor; geometries: THREE.BufferGeometry[]; marble: THREE.Texture | null }> {
    const dim = new THREE.Color(this.theme.s500).getHex()
    if (def.kind === 'g1-rigid') {
      // STL rig and marble texture are independent downloads: fetch them together.
      const [assets, marble] = await Promise.all([
        loadArenaGeometries(root + def.modelBase),
        new THREE.TextureLoader().loadAsync(root + 'textures/marble-gold.png').catch(() => null),
      ])
      if (marble) {
        marble.colorSpace = THREE.SRGBColorSpace
        marble.wrapS = marble.wrapT = THREE.RepeatWrapping
        marble.repeat.set(0.62, 0.62); marble.center.set(0.5, 0.5); marble.rotation = -0.08
      }
      return { make: () => new G1Actor(assets, ROBOT, marble, dim), geometries: [...assets.geometries.values()], marble }
    }
    // One shared GLB (cached), one clone per lane, fitted to the G1's stage height.
    const asset = await loadMixamoAsset(def, root)
    return { make: () => new FairyActor(asset, ROBOT, dim, { displayHeight: 1.4 }, def), geometries: [], marble: null }
  }

  // ------------------------------------------------------------------ public API
  setActive(active: boolean) {
    if (this.disposed || active === this.active) return
    this.active = active
    if (active) { this.last = performance.now(); this.dirty = true; this.raf = requestAnimationFrame(this.tick) }
    else cancelAnimationFrame(this.raf)
  }

  setPlaying(playing: boolean) {
    this.playing = playing
    this.playSprite.visible = !playing && this.ready
    this.vs.visible = playing || !this.ready
    this.dirty = true
  }

  /** Keyboard focus (or mouse hover) of the DOM A/B buttons: rim highlight + dark scheme. */
  setFocusSide(side: Side | null) { this.focused = side; this.dirty = true; this.syncTone() }

  /** Brief navy flash on one half (a key vote is about to register). */
  flashSide(side: Side, ms = 220) {
    if (this.disposed) return
    this.flash = side; this.syncTone()
    window.clearTimeout(this.flashTimer)
    this.flashTimer = window.setTimeout(() => { this.flash = null; this.syncTone() }, ms)
  }

  getTones(): SideTones { return { A: this.sides.A.tone, B: this.sides.B.tone } }

  setHud(hud: HudState) { this.hud = hud; this.drawHud(); this.dirty = true }

  /** Shows a new pair. The first call animates the robots in; later calls swap them. */
  async setMatchup(a: LaneContent, b: LaneContent) {
    const reduced = this.opts.reducedMotion
    const first = this.lanes.every(l => !l.label)
    if (!first) {
      const from = this.lanes.map(l => ({ lift: l.liftV, dim: l.dimV, spot: l.spotV }))
      await this.tween(reduced ? 0 : 0.38, k => this.lanes.forEach((l, i) => {
        l.scaleV = 1 - k; l.liftV = from[i].lift * (1 - k); l.dimV = from[i].dim * (1 - k); l.spotV = from[i].spot * (1 - k)
      }), easeInCubic)
    }
    const contents = [a, b]
    this.lanes.forEach((l, i) => {
      l.clip = contents[i].clip
      l.actor.setClip(contents[i].clip)
      this.robotFlares[i]?.reset()
      l.label = contents[i].label
      l.liftV = 0; l.dimV = 0; l.spotV = 0
    })
    this.drawCaptions()
    // The flare sweeps down and draws the new divider (the first one may be the loader's flare arriving).
    const fly = !this.awaitHandoff && !this.handoffSwept
    this.handoffSwept = false
    this.setPath(fly, 0.1)
    this.matchTime = 0
    this.xpPop.visible = false
    await this.tween(reduced ? 0 : 0.55, k => this.lanes.forEach(l => { l.scaleV = k }), easeOutBack)
  }

  /** Visual result of a vote: winner lifts into a spotlight, loser dims. No score is shown. */
  async resolve(choice: Choice, xpGain: number) {
    const reduced = this.opts.reducedMotion
    if (choice === 'skip') return
    const [A, B] = this.lanes
    const winners = choice === 'tie' ? [A, B] : [choice === 'A' ? A : B]
    const losers = choice === 'tie' ? [] : [choice === 'A' ? B : A]
    const liftTo = choice === 'tie' ? 0.07 : 0.14
    const spotTo = choice === 'tie' ? 0.6 : 1
    // Environment takes the sunset feedback colour for the celebration, then falls
    // back to dark (pointer / focus still on a side) or to the primary scheme.
    this.dustBurst(1) // the flare bursts into dust around the VS, which punches with the vote
    // Only the voted half lights up (both for a tie); the other half is left as it is.
    const token = ++this.holdSeq
    const fbSides = winners.map(l => l.side)
    fbSides.forEach(sd => { this.sides[sd].hold = token })
    this.syncTone()
    void this.wait(FEEDBACK_HOLD_SECONDS).then(() => {
      if (this.disposed) return
      fbSides.forEach(sd => { if (this.sides[sd].hold === token) this.sides[sd].hold = 0 })
      this.syncTone()
    })
    // XP pop over the chosen lane (or the VS medallion for a tie).
    this.xpLabel.draw((ctx, w, h) => {
      ctx.fillStyle = W.ink
      ctx.beginPath(); ctx.roundRect(4, 8, w - 8, h - 16, (h - 16) / 2); ctx.fill()
      ctx.fillStyle = W.paper; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.font = `700 44px ${this.theme.font}`
      ctx.fillText(`+${xpGain} XP`, w / 2, h / 2 + 2)
    })
    const anchor = new THREE.Vector3()
    if (choice === 'tie') anchor.set(0, 1.45, 0.3)
    else winners[0].root.localToWorld(anchor.set(0, 2.2, 0))
    this.xpPop.position.copy(anchor)
    this.xpPopY = anchor.y
    this.xpPop.visible = true
    this.xpPopV = 0
    if (!reduced && choice !== 'tie') this.burst(winners[0])
    await Promise.all([
      this.tween(reduced ? 0 : 0.45, k => {
        winners.forEach(l => { l.liftV = liftTo * k; l.spotV = spotTo * k })
        losers.forEach(l => { l.dimV = k })
      }, easeOutCubic),
      this.tween(reduced ? 0 : 1.0, k => { this.xpPopV = k }, x => x),
    ])
    await this.wait(reduced ? 0.7 : 0.75)
    this.xpPop.visible = false
  }

  dispose() {
    this.disposed = true
    this.robotFlares.forEach(f => f.dispose()); this.robotFlares = []
    this.stylized.forEach(h => h.dispose()); this.stylized = []
    this.offTarget(); this.offHandoff()
    cancelAnimationFrame(this.raf)
    this.observer.disconnect()
    const el = this.renderer.domElement
    el.removeEventListener('pointerdown', this.onPointerDown)
    el.removeEventListener('pointermove', this.onPointerMove)
    el.removeEventListener('pointerleave', this.onPointerLeave)
    el.removeEventListener('pointerup', this.onPointerUp)
    el.removeEventListener('pointercancel', this.onPointerUp)
    el.removeEventListener('click', this.onClick)
    window.clearTimeout(this.flashTimer)
    el.removeEventListener('webglcontextlost', this.onContextLost)
    this.tweens.forEach(t => t.resolve()); this.tweens = []
    this.world.dispose() // removes and disposes the world group + composite targets
    this.lanes.forEach(l => l.actor.dispose()) // detach actors first: the fairy's geometry is shared (cached GLB)
    const textures = new Set<THREE.Texture>()
    this.scene.traverse(obj => {
      const o = obj as THREE.Mesh
      if (o.geometry) o.geometry.dispose()
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : []
      for (const m of mats) {
        const map = (m as THREE.MeshBasicMaterial).map
        if (map) textures.add(map)
        m.dispose()
      }
    })
    textures.forEach(t => t.dispose())
    this.geometries.forEach(g => g.dispose())
    this.marble?.dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
    el.remove()
  }

  // ------------------------------------------------------------------ internals
  private tween(dur: number, update: (k: number) => void, ease: (x: number) => number = easeOutCubic) {
    if (dur <= 0 || this.disposed) { update(1); this.dirty = true; return Promise.resolve() }
    return new Promise<void>(resolve => this.tweens.push({ t: 0, dur, ease, update, resolve }))
  }

  private wait(seconds: number) { return this.tween(seconds, () => {}, x => x) }

  private burst(lane: Lane) {
    const pos = this.particles.geometry.getAttribute('position') as THREE.BufferAttribute
    const origin = lane.root.localToWorld(new THREE.Vector3(0, 1.25, 0))
    for (let i = 0; i < pos.count; i++) {
      const a = Math.random() * Math.PI * 2, up = 1.6 + Math.random() * 2.2, out = 0.6 + Math.random() * 1.4
      pos.setXYZ(i, origin.x + (Math.random() - 0.5) * 0.3, origin.y + Math.random() * 0.4, origin.z + (Math.random() - 0.5) * 0.3)
      this.particleVel.set([Math.cos(a) * out, up, Math.sin(a) * out * 0.6], i * 3)
    }
    pos.needsUpdate = true
    // Confetti takes the palette of the scheme its half is celebrating in (feedback).
    const col = this.particles.geometry.getAttribute('color') as THREE.BufferAttribute
    const sch = this.schemes[this.sides[lane.side].tone], pal = [sch.p0, sch.p1, sch.p2, sch.p3]
    for (let i = 0; i < col.count; i++) { const c = pal[i % pal.length]; col.setXYZ(i, c.r, c.g, c.b) }
    col.needsUpdate = true
    this.particleAge = 0
    this.particles.visible = true
  }

  private adaptPixelRatio(dt: number) {
    if (dt <= 0) return
    this.frameEma += (dt - this.frameEma) * 0.05
    const pr = this.renderer.getPixelRatio()
    if (this.frameEma > 1 / 50 && pr > 1.25) {
      this.slowFor += dt
      if (this.slowFor > 2) {
        this.slowFor = 0; this.frameEma = 1 / 60
        this.renderer.setPixelRatio(Math.max(1.25, pr - 0.25))
        this.resize()
      }
    } else this.slowFor = 0
  }

  private resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight)
    this.width = w; this.height = h
    this.renderer.setSize(w, h, false)
    const aspect = w / h
    this.camera.aspect = aspect
    // Portrait: lanes closer together and a slightly wider lens.
    this.laneX = aspect < 0.9 ? 0.66 : aspect < 1.3 ? 0.98 : 1.15
    this.camera.fov = aspect < 0.9 ? 40 : 34
    this.camera.updateProjectionMatrix()
    const inward = aspect < 0.9 ? 0.14 : 0.22
    this.lanes.forEach((l, i) => {
      const s = i === 0 ? -1 : 1
      l.root.position.set(s * this.laneX, 0, 0)
      l.root.rotation.y = -s * inward
    })
    // Fit both pedestals horizontally and the robots (pedestal .. badge) vertically into
    // the band left between the top reserve (nav, plus the HUD row if the badges would
    // hit the panels) and the bottom reserve (DOM vote bar + the question block).
    const vHalf = THREE.MathUtils.degToRad(this.camera.fov / 2)
    const hHalf = Math.atan(Math.tan(vHalf) * aspect)
    const halfW = this.laneX + (aspect < 0.9 ? 0.64 : 0.72)
    const q = this.questionMetrics()
    const bottomPx = q.bottomPx + q.blockPx + 24 // 24px between the pedestals and the question
    const fit = (topPx: number) => {
      const usable = Math.max(0.45, 1 - (topPx + bottomPx) / h)
      this.camDist = Math.max(halfW / Math.tan(hHalf), (1.12 / usable) / Math.tan(vHalf), 4.2)
      const wppT = (2 * Math.tan(vHalf) * this.camDist) / h // world units per CSS px at the target
      // Sit the robots on the bottom of the free band (just above the question); any
      // spare height (width-limited portrait fits) stays as sky above them.
      const robotsHalfPx = 1.12 / wppT
      const centreFromTop = Math.max(topPx + robotsHalfPx, h - bottomPx - robotsHalfPx)
      this.camTarget.set(0, 0.92 + (centreFromTop - h / 2) * wppT, 0)
    }
    const nav = navClearPx(this.width)
    fit(nav + 16)
    if (this.badgesHitPanels()) fit(nav + HUD_PX.panelH + 2 * HUD_PX.pad)
    this.layoutHud()
    this.setPath(false) // re-fit the ribbon to the new size
    this.dirty = true
    if (this.active) this.render()
  }

  /** Question label geometry in CSS px: width, text block height, and the text block's bottom (from the screen bottom). */
  private questionMetrics() {
    const compact = this.width < 560
    const widthPx = Math.min(700, this.width - 2 * HUD_PX.pad)
    const k = widthPx / this.hudQuestion.w
    const q = QUESTION_TYPE[compact ? 'compact' : 'wide']
    // question line(s) + the "Press A or B" hint line with keycaps (see drawHud)
    const blockPx = (q.lines * q.size * 1.18 + q.hint * 1.6) * k
    const controlsPx = compact ? 12 + 44 : 16 + 44 // .hero-arena__controls: bottom offset + 44px bar (measured)
    return { compact, widthPx, blockPx, bottomPx: controlsPx + 16 }
  }

  /** With the current camera fit: would an A/B badge come within 16px of a top HUD panel's column? */
  private badgesHitPanels() {
    const compact = this.width < 560
    const leftPx = compact ? Math.min(178, (this.width - 3 * HUD_PX.pad) / 2) : 250
    const rightPx = compact ? Math.min(140, (this.width - 3 * HUD_PX.pad) / 2) : 190
    const el = 0.13, d = this.camDist
    this.camera.position.set(this.camTarget.x, this.camTarget.y + Math.sin(el) * d, this.camTarget.z + Math.cos(el) * d)
    this.camera.lookAt(this.camTarget)
    this.camera.updateMatrixWorld()
    const v = this.tmpBadge
    const badgeHalfPx = (0.12 / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * d)) * this.height
    for (const sx of [-1, 1]) {
      v.set(sx * this.laneX, 0.2 + 1.72, 0).project(this.camera)
      const x = (v.x + 1) / 2 * this.width
      const y = (1 - v.y) / 2 * this.height
      const top = navClearPx(this.width), panelBottom = top + HUD_PX.panelH
      const inRow = y - badgeHalfPx < panelBottom + 16
      const hitsLeft = x - badgeHalfPx < HUD_PX.pad + leftPx + 16
      const hitsRight = x + badgeHalfPx > this.width - HUD_PX.pad - rightPx - 16
      if (inRow && (hitsLeft || hitsRight)) return true
    }
    return false
  }
  private tmpBadge = new THREE.Vector3()

  private layoutHud() {
    const d = 1
    const halfH = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * d
    const halfW = halfH * this.camera.aspect
    const wpp = (2 * halfH) / this.height // world units per CSS px at depth d
    const pad = HUD_PX.pad * wpp
    const compact = this.width < 560
    const leftPx = compact ? Math.min(178, (this.width - 3 * HUD_PX.pad) / 2) : 250
    const rightPx = compact ? Math.min(140, (this.width - 3 * HUD_PX.pad) / 2) : 190
    const ph = HUD_PX.panelH * wpp
    const set = (m: THREE.Mesh, label: CanvasLabel, pxW: number, x: number, y: number) => {
      const wW = pxW * wpp, hW = wW * label.h / label.w
      m.scale.set(wW, hW, 1)
      m.position.set(x, y, -d)
      return { wW, hW }
    }
    const top = halfH - pad - ph / 2
    const topInset = navClearPx(this.width) * wpp
    const l = set(this.hudLeftMesh, this.hudLeft, leftPx, 0, top)
    this.hudLeftMesh.position.x = -halfW + pad + l.wW / 2
    this.hudLeftMesh.position.y = halfH - topInset - l.hW / 2
    const r = set(this.hudRightMesh, this.hudRight, rightPx, 0, top)
    this.hudRightMesh.position.x = halfW - pad - r.wW / 2
    this.hudRightMesh.position.y = halfH - topInset - r.hW / 2
    const qm = this.questionMetrics()
    if (qm.compact !== this.questionCompact) { this.questionCompact = qm.compact; this.drawHud() }
    const q = set(this.hudQuestionMesh, this.hudQuestion, qm.widthPx, 0, 0)
    // Sits under the arena with 16px clearance above the DOM vote controls. drawHud()
    // bottom-aligns the text 16 canvas px above the label's bottom edge.
    const planeBottomPx = qm.bottomPx - 16 * (qm.widthPx / this.hudQuestion.w)
    this.hudQuestionMesh.position.y = -halfH + planeBottomPx * wpp + q.hW / 2
    // VS coin on the ribbon's centre pin (uniform scale: it has depth). 0.6 keeps it ~70% of the old burst;
    // the floor keeps the lettering legible on phones.
    const vsPx = THREE.MathUtils.clamp(Math.min(this.width * 0.3, this.height * 0.34), 108, 270)
    this.vsBody.scale.setScalar(Math.max(vsPx * 0.6, 92) * wpp)
    this.vs.position.set(0, (this.height / 2 - this.vsY()) * wpp, -d)
  }

  private redrawText() {
    this.drawHud()
    this.drawCaptions()
  }

  /**
   * Gentle wave for the ribbon, pinned to the screen centre at the VS. With `fly`,
   * the flare head starts a pass from under the nav and draws the ribbon in behind
   * it (reduced motion: the full ribbon at once, the head parked on the VS).
   */
  private setPath(fly: boolean, punch = 0) {
    const reduced = this.opts.reducedMotion
    const W = this.width, H = this.height, cx = W / 2
    const top = navClearPx(W) - 6, vsY = this.vsY()
    const amp = THREE.MathUtils.clamp(W * 0.012, 6, 16), hw = THREE.MathUtils.clamp(W * 0.006, 5, 9)
    if (fly || !this.ribbon.length) this.wave = { freq: 1 + Math.random() * 0.35, dir: Math.random() < 0.5 ? -1 : 1 }
    const xAt = (y: number) => cx + this.wave.dir * amp * Math.sin(((y - vsY) / H) * Math.PI * 2 * this.wave.freq)
    const pts: RibbonPoint[] = [{ x: xAt(top), y: -60, w: 0 }, { x: xAt(top), y: top, w: 0 }]
    const steps = RIBBON_N - 2
    for (let i = 1; i <= steps; i++) {
      const y = THREE.MathUtils.lerp(top, H + 60, i / steps)
      pts.push({ x: xAt(y), y, w: hw * Math.min(1, 0.35 + i * 0.22) })
    }
    this.ribbon = pts
    this.layoutRibbon()
    if (fly && !reduced) this.startSweep(performance.now(), false, punch)
    else if (reduced || !this.flying) this.flareU.reveal.value = 1e6
    this.dirty = true
  }

  /** Top of the ribbon sweep (CSS px), just under the nav. */
  private sweepTop() { return navClearPx(this.width) - 6 }
  private headRadius() { return THREE.MathUtils.clamp(this.width * 0.024, 22, 34) }

  /**
   * One split sweep (shared by the loader handoff and every new matchup): the head
   * runs down the ribbon on a Hermite ease from `t0` (rAF ms), drawing the split in
   * behind it, then the ribbon dissolves into sparkles. `fromLoader`: the head is
   * already fully visible (it arrives from the loading screen at full speed).
   */
  private startSweep(t0: number, fromLoader: boolean, punch = 0) {
    if (this.opts.reducedMotion) return
    const top = this.sweepTop()
    this.sweepStart = t0; this.sweepFromLoader = fromLoader
    this.flying = true; this.flyY = top; this.dissolveStart = -1
    this.flareU.reveal.value = top; this.flareU.headY.value = top; this.flareU.tailY.value = top
    this.flareU.dissolve.value = 0
    this.flareU.glow.value = 1
    this.vsPunch = Math.max(this.vsPunch, punch)
    this.ribbonMesh.visible = true; this.flareHead.visible = true
    this.dirty = true
  }
  private wave = { freq: 1.15, dir: 1 }

  /** Ribbon centre x at a CSS y. */
  private ribbonX(y: number) {
    const p = this.ribbon
    for (let i = 0; i < p.length - 1; i++) if (y >= p[i].y && y <= p[i + 1].y) return THREE.MathUtils.lerp(p[i].x, p[i + 1].x, (y - p[i].y) / Math.max(1e-6, p[i + 1].y - p[i].y))
    return this.width / 2
  }

  /** Adds one sparkle to the pool (dropped when full). kind 0 = fine dust, 1 = four-point glint. */
  private spawnMote(x: number, y: number, vx: number, vy: number, max: number, size: number, kind: 0 | 1) {
    if (this.moteCount >= MOTE_MAX) return
    const i = this.moteCount++
    this.mx[i] = x; this.my[i] = y; this.mvx[i] = vx; this.mvy[i] = vy
    this.mlife[i] = 0; this.mmax[i] = max; this.msize[i] = size; this.mkind[i] = kind
    this.mphase[i] = Math.random() * Math.PI * 2; this.mrate[i] = 9 + Math.random() * 9
    this.mcol[i] = kind ? 1 + Math.floor(Math.random() * 1.8) : 2 + Math.floor(Math.random() * 1.7) // glints: sunset/gold with a hot core (shader)
  }

  /** Dust shed by the flare: `n` sparkles around (x, y), like the loader's falling sparkle dust. */
  private shedDust(x: number, y: number, n: number, speed: [number, number]) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, s = speed[0] + Math.random() * (speed[1] - speed[0])
      const glint = Math.random() < 0.22
      this.spawnMote(x + (Math.random() - 0.5) * 6, y + (Math.random() - 0.5) * 6, Math.cos(a) * s, Math.sin(a) * s + 10,
        glint ? 0.5 + Math.random() * 0.7 : 1.1 + Math.random() * 1.2, glint ? 6 + Math.random() * 8 : 1.4 + Math.random() * 2.2, glint ? 1 : 0)
    }
  }

  /** Vote: the flare bursts into dust around the VS and the ribbon flashes. */
  private dustBurst(power: number) {
    if (this.opts.reducedMotion) return
    this.shedDust(this.ribbonX(this.vsY()), this.vsY(), Math.round(90 * power), [40, 150])
    this.flareU.glow.value = Math.max(this.flareU.glow.value, power)
    this.vsPunch = Math.max(this.vsPunch, 0.24 * power)
    this.vsFlip += Math.PI * 2 // the coin flips with the vote
  }

  /** Screen y (CSS px) of the VS coin / ribbon pin: a little above centre. */
  private vsY() { return Math.round(this.height * 0.47) }

  /** Pushes the ribbon into the geometry (CSS px) and the split uniform (device px, GL y-up). */
  private layoutRibbon() {
    if (this.ribbon.length !== RIBBON_N) return
    const pos = this.ribbonGeo.getAttribute('position') as THREE.BufferAttribute
    const wAt = this.ribbonGeo.getAttribute('w') as THREE.BufferAttribute
    const dpr = this.renderer.getPixelRatio(), H = this.height
    this.ribbon.forEach((p, i) => {
      pos.setXYZ(i * 2, p.x, p.y, -1); pos.setXYZ(i * 2 + 1, p.x, p.y, 1)
      wAt.setX(i * 2, p.w); wAt.setX(i * 2 + 1, p.w)
      this.split.pts[RIBBON_N - 1 - i].set(p.x * dpr, (H - p.y) * dpr)
    })
    pos.needsUpdate = true; wAt.needsUpdate = true
    this.flareU.res.value.set(this.width, this.height)
    this.flareU.dpr.value = dpr
  }

  /** Per frame: sweep the head down the ribbon, dissolve it, spawn/age the sparkles, upload them. */
  private stepFlare(dt: number) {
    const H = this.height, top = this.sweepTop()
    const head = this.flareHead.material.uniforms
    if (this.opts.reducedMotion) {
      // No sweep, no sparkles, no persistent divider: the split still uses the ribbon's line.
      this.flying = false; this.dissolveStart = -1; this.moteCount = 0
      this.flareU.headY.value = -1e4
      this.motesGeo.setDrawRange(0, 0)
      this.ribbonMesh.visible = false; this.flareHead.visible = false; this.motesMesh.visible = false
      return
    }
    const now = this.last // rAF timestamp of this frame (same clock as the loader's handoff time)
    const radius = this.headRadius()
    head.radius.value = radius
    const rand = Math.random

    if (this.flying) {
      const u = Math.max(0, (now - this.sweepStart) / 1000 / FLARE_SWEEP_SECONDS)
      this.flyY = top + (H + 60 - top) * flareSweepEase(u)
      this.flareU.reveal.value = this.flyY
      this.flareU.headY.value = this.flyY
      this.flareU.tailY.value = top
      if (u >= 1) { this.flying = false; this.dissolveStart = now; this.flareU.reveal.value = 1e6 }
    }
    const hx = this.ribbonX(this.flyY), hy = this.flyY
    const fadeIn = this.sweepFromLoader ? 1 : THREE.MathUtils.smoothstep(this.flyY, top, top + 60)
    head.center.value.set(hx, hy)
    head.alpha.value = this.flying ? fadeIn : 0
    head.star.value = 0.32 + 0.05 * Math.sin(this.clock * 11)
    this.flareHead.visible = this.flying && hy < H + radius
    this.flareU.glow.value = Math.max(0, this.flareU.glow.value - dt * 2.4)

    // Sparkles while flying: fine dust trailing just behind the head, glints scattered around it.
    if (this.flying && hy < H + 20) {
      this.dustAcc += dt * 190
      for (; this.dustAcc >= 1; this.dustAcc--) {
        const back = rand() * 26
        const a = rand() * Math.PI * 2, sp = 6 + rand() * 22
        this.spawnMote(this.ribbonX(hy - back) + (rand() - 0.5) * 8, hy - back, Math.cos(a) * sp, Math.sin(a) * sp + 8, 0.8 + rand() * 1.1, 1.2 + rand() * 1.6, 0)
      }
      this.sparkleAcc += dt * 40
      for (; this.sparkleAcc >= 1; this.sparkleAcc--) {
        const a = rand() * Math.PI * 2, d = radius * (0.35 + rand() * 0.9)
        this.spawnMote(hx + Math.cos(a) * d, hy - rand() * 30 + Math.sin(a) * d * 0.6, Math.cos(a) * 10, Math.sin(a) * 10 + 4, 0.45 + rand() * 0.6, 6 + rand() * 9, 1)
      }
    }

    // Dissolve: the ribbon burns away on noise while glints and dust scatter off it and twinkle out last.
    if (this.dissolveStart >= 0) {
      const k = Math.min(1, (now - this.dissolveStart) / 1000 / DISSOLVE_S)
      this.flareU.dissolve.value = (0.5 - 0.5 * Math.cos(Math.PI * k)) * 1.15
      this.sparkleAcc += dt * 110 * (1 - k * 0.7)
      for (; this.sparkleAcc >= 1; this.sparkleAcc--) {
        const y = top + rand() * (H - top), glint = rand() < 0.45
        const a = rand() * Math.PI * 2, sp = 8 + rand() * 26
        this.spawnMote(this.ribbonX(y) + (rand() - 0.5) * 10, y, Math.cos(a) * sp, Math.sin(a) * sp - 4,
          glint ? 0.6 + rand() * 0.9 : 0.9 + rand() * 1.1, glint ? 5 + rand() * 9 : 1.2 + rand() * 1.8, glint ? 1 : 0)
      }
      if (k >= 1) { this.dissolveStart = -1; this.flareU.headY.value = -1e4 }
    }
    this.ribbonMesh.visible = this.flying || this.dissolveStart >= 0

    // Age + upload (swap-remove, no allocation).
    const [pos, size, alpha, kind, col] = this.motesAttrs
    const damp = Math.pow(0.97, dt * 60)
    for (let i = 0; i < this.moteCount;) {
      this.mlife[i] += dt
      if (this.mlife[i] >= this.mmax[i]) {
        const j = --this.moteCount
        this.mx[i] = this.mx[j]; this.my[i] = this.my[j]; this.mvx[i] = this.mvx[j]; this.mvy[i] = this.mvy[j]
        this.mlife[i] = this.mlife[j]; this.mmax[i] = this.mmax[j]; this.msize[i] = this.msize[j]; this.mkind[i] = this.mkind[j]
        this.mphase[i] = this.mphase[j]; this.mrate[i] = this.mrate[j]; this.mcol[i] = this.mcol[j]
        continue
      }
      this.mvx[i] *= damp; this.mvy[i] = this.mvy[i] * damp + 14 * dt // drift down like falling dust
      this.mx[i] += this.mvx[i] * dt; this.my[i] += this.mvy[i] * dt
      const l = this.mlife[i] / this.mmax[i]
      const env = THREE.MathUtils.smoothstep(l, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(l, 0.5, 1))
      const tw = 0.5 + 0.5 * Math.sin(this.mphase[i] + this.clock * this.mrate[i])
      const glint = this.mkind[i] === 1
      pos.setXYZ(i, this.mx[i], this.my[i], 0)
      size.setX(i, glint ? this.msize[i] * (0.55 + 0.45 * env) * (0.8 + 0.2 * tw) : this.msize[i])
      alpha.setX(i, glint ? env * (0.3 + 0.7 * tw) : env * (0.75 + 0.25 * tw))
      kind.setX(i, this.mkind[i])
      const c = FLARE_COLORS[this.mcol[i]]
      col.setXYZ(i, c.r, c.g, c.b)
      i++
    }
    uploadLiveRange(this.motesAttrs, this.moteCount)
    this.motesGeo.setDrawRange(0, this.moteCount)
    this.motesMesh.visible = this.moteCount > 0
  }

  private drawCaptions() {
    const t = this.theme
    for (const lane of this.lanes) {
      lane.caption.draw((ctx, w, h) => {
        ctx.fillStyle = W.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.font = `600 40px ${t.font}`
        ctx.fillText(lane.label, w / 2, h / 2 + 2, w - 24)
      })
    }
  }

  private drawHud() {
    const t = this.theme, h = this.hud
    const panel = (ctx: CanvasRenderingContext2D, w: number, hh: number) => {
      ctx.fillStyle = W.paper
      ctx.globalAlpha = 0.9
      ctx.beginPath(); ctx.roundRect(2, 2, w - 4, hh - 4, 28); ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = W.accent; ctx.lineWidth = 2
      ctx.beginPath(); ctx.roundRect(2, 2, w - 4, hh - 4, 28); ctx.stroke()
    }
    this.hudLeft.draw((ctx, w, hh) => {
      panel(ctx, w, hh)
      ctx.fillStyle = W.muted; ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'
      ctx.font = `500 30px ${t.mono}`
      ctx.fillText(`ROUND ${String(h.round).padStart(2, '0')}`, 30, 54)
      ctx.textAlign = 'right'
      ctx.fillText(`LV ${h.level}`, w - 30, 54)
      // level progress bar
      const bx = 30, by = 76, bw = w - 60, bh = 20
      ctx.fillStyle = W.track
      ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, bh / 2); ctx.fill()
      ctx.fillStyle = W.ink
      const fw = Math.max(bh, bw * Math.min(1, Math.max(0, h.levelProgress)))
      if (h.levelProgress > 0) { ctx.beginPath(); ctx.roundRect(bx, by, fw, bh, bh / 2); ctx.fill() }
    })
    this.hudRight.draw((ctx, w, hh) => {
      panel(ctx, w, hh)
      ctx.textBaseline = 'alphabetic'
      ctx.fillStyle = W.muted; ctx.font = `500 28px ${t.mono}`; ctx.textAlign = 'left'
      ctx.fillText('XP', 30, 48)
      ctx.textAlign = 'right'; ctx.fillText('STREAK', w - 30, 48)
      ctx.fillStyle = W.ink; ctx.font = `700 46px ${t.font}`
      ctx.textAlign = 'left'; ctx.fillText(String(h.xp), 30, 102)
      ctx.textAlign = 'right'; ctx.fillText(`×${h.streak}`, w - 30, 102)
    })
    this.hudQuestion.draw((ctx, w, hh) => {
      ctx.textBaseline = 'middle'; ctx.lineJoin = 'round'
      // Bottom-aligned block: the question (one line wide, two lines compact), then a hint line
      // with keycap chips: "Press [A] or [B] to choose · drag to look around" (touch: "Tap ... swipe").
      const q = QUESTION_TYPE[this.questionCompact ? 'compact' : 'wide']
      const lines = h.playing
        ? (this.questionCompact ? ['Which one moves', 'more naturally?'] : ['Which one moves more naturally?'])
        : (this.questionCompact ? ['Press play, then pick', 'the motion you prefer.'] : ['Press play, then pick the motion you prefer.'])
      const lineH = q.size * 1.18, hintH = q.hint * 1.6
      const halo = (size: number) => { ctx.strokeStyle = W.paper; ctx.lineWidth = Math.round(size * 0.3) }
      ctx.textAlign = 'center'
      ctx.font = `600 ${q.size}px ${t.font}`
      lines.forEach((text, i) => {
        const y = hh - 16 - hintH - lineH / 2 - (lines.length - 1 - i) * lineH
        halo(q.size); ctx.strokeText(text, w / 2, y, w - 40)
        ctx.fillStyle = W.ink; ctx.fillText(text, w / 2, y, w - 40)
      })
      // hint line with keycaps, centred and shrunk to fit
      const verb = this.coarsePointer ? 'Tap' : 'Press', look = this.coarsePointer ? 'swipe to look around' : 'drag to look around'
      const parts: (string | { key: string })[] = [`${verb} `, { key: 'A' }, ' or ', { key: 'B' }, ` to choose · ${look}`]
      let size = q.hint
      const capW = (sz: number) => sz * 1.25, gap = (sz: number) => sz * 0.18
      const measure = (sz: number) => {
        ctx.font = `500 ${sz}px ${t.font}`
        return parts.reduce((acc, p) => acc + (typeof p === 'string' ? ctx.measureText(p).width : capW(sz) + 2 * gap(sz)), 0)
      }
      let total = measure(size)
      if (total > w - 40) { size *= (w - 40) / total; total = measure(size) }
      let x = (w - total) / 2
      const y = hh - 16 - hintH / 2
      ctx.textAlign = 'left'
      for (const p of parts) {
        if (typeof p === 'string') {
          ctx.font = `500 ${size}px ${t.font}`
          halo(size); ctx.strokeText(p, x, y)
          ctx.fillStyle = W.ink; ctx.fillText(p, x, y)
          x += ctx.measureText(p).width
        } else {
          // keycap chip: paper face, ink border, a thin ink "depth" edge underneath, ink letter
          const cw = capW(size), ch = size * 1.22, cx = x + gap(size), top = y - ch / 2, r = size * 0.22
          ctx.fillStyle = W.ink
          ctx.beginPath(); ctx.roundRect(cx, top + size * 0.1, cw, ch, r); ctx.fill()
          ctx.fillStyle = W.paper; ctx.strokeStyle = W.ink; ctx.lineWidth = Math.max(2, size * 0.08)
          ctx.beginPath(); ctx.roundRect(cx, top, cw, ch, r); ctx.fill(); ctx.stroke()
          ctx.fillStyle = W.ink; ctx.textAlign = 'center'; ctx.font = `700 ${size * 0.78}px ${t.mono}`
          ctx.fillText(p.key, cx + cw / 2, y + size * 0.02)
          ctx.textAlign = 'left'
          x += cw + 2 * gap(size)
        }
      }
    })
  }

  // ------------------------------------------------------------------ colour scheme
  private syncTone() {
    if (this.disposed) return
    let changed = false
    for (const side of ['A', 'B'] as Side[]) {
      const st = this.sides[side]
      const next: Tone = st.hold ? 'feedback' : (this.flash === side || this.hovered === side || this.focused === side) ? 'dark' : 'light'
      if (next === st.tone) continue
      st.tone = next
      for (const k of SCHEME_KEYS) st.from[k].copy(st.col[k])
      copyAuraWorldScheme(st.wFrom, st.wCol)
      st.t = 0
      if (this.opts.reducedMotion) this.stepTone(st, 1) // instant switch, same colours
      changed = true
    }
    if (!changed) return
    this.dirty = true
    this.opts.callbacks.onTone?.(this.getTones())
  }

  /** Advances one side's scheme tween (ease-out over TONE_SECONDS). */
  private stepTone(st: SideScheme, dt: number) {
    if (st.t >= 1) return
    st.t = Math.min(1, st.t + dt / TONE_SECONDS)
    const k = easeOutCubic(st.t), to = this.schemes[st.tone]
    for (const key of SCHEME_KEYS) st.col[key].lerpColors(st.from[key], to[key], k)
    lerpAuraWorldScheme(st.wCol, st.wFrom, this.worldSchemes[st.tone], k)
  }

  /** For the world props: is the pointer (NDC) over a robot's hit proxy or the play sprite? */
  private pointerOnUi(x: number, y: number) {
    if (this.orbit.dragging) return true // the pointer is orbiting the camera
    this.propsNdc.set(x, y)
    this.raycaster.setFromCamera(this.propsNdc, this.camera)
    for (const l of this.lanes) if (this.raycaster.intersectObject(l.hit, false).length) return true
    return this.playSprite.visible && this.raycaster.intersectObject(this.playSprite, false).length > 0
  }
  private propsNdc = new THREE.Vector2()

  private hitTest(): { side: Side | null; play: boolean } {
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const targets: THREE.Object3D[] = this.lanes.map(l => l.hit)
    if (this.playSprite.visible) targets.unshift(this.playSprite)
    const hit = this.raycaster.intersectObjects(targets, false)[0]
    if (!hit) return { side: null, play: false }
    if (hit.object.userData.play) return { side: null, play: true }
    return { side: hit.object.userData.side as Side, play: false }
  }

  private setPointer(e: PointerEvent | MouseEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect()
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
  }

  private onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0 || this.orbit.dragging) return
    const o = this.orbit
    o.dragging = true; o.pid = e.pointerId; o.moved = false; o.touch = e.pointerType !== 'mouse'
    o.lx = o.sx = e.clientX; o.ly = o.sy = e.clientY
    if (!o.touch) { this.renderer.domElement.setPointerCapture?.(e.pointerId); this.renderer.domElement.style.cursor = 'grabbing' }
  }

  private onPointerMove = (e: PointerEvent) => {
    const o = this.orbit
    if (!o.dragging) { this.hoverAt(e); return }
    if (e.pointerId !== o.pid) return
    const dx = e.clientX - o.lx, dy = e.clientY - o.ly
    o.lx = e.clientX; o.ly = e.clientY
    if (!o.moved && Math.hypot(e.clientX - o.sx, e.clientY - o.sy) > 5) o.moved = true
    if (!o.moved) return
    // touch: yaw only (vertical swipes belong to page scroll; the browser takes them via pan-y)
    o.tYaw = THREE.MathUtils.clamp(o.tYaw - dx * 0.0055, -ORBIT_YAW_MAX, ORBIT_YAW_MAX)
    if (!o.touch) o.tPitch = THREE.MathUtils.clamp(o.tPitch + dy * 0.004, ORBIT_PITCH_MIN, ORBIT_PITCH_MAX)
    o.idle = 0
    this.dirty = true
  }

  private onPointerUp = (e: PointerEvent) => {
    const o = this.orbit
    if (!o.dragging || e.pointerId !== o.pid) return
    o.dragging = false; o.pid = -1; o.idle = 0
    this.renderer.domElement.style.cursor = 'grab'
  }

  /**
   * Hover (mouse / pen, no button pressed): the half under the pointer takes the navy hover
   * scheme, decided against the split line itself (ribbon x at the pointer's height), so it
   * always matches the visible A | B halves. Frozen while dragging; touch has no hover. Never votes.
   */
  private hoverAt(e: PointerEvent) {
    if (e.pointerType === 'touch' || e.buttons !== 0) return
    const rect = this.renderer.domElement.getBoundingClientRect()
    const x = e.clientX - rect.left, y = e.clientY - rect.top
    this.setHovered(x < this.ribbonX(y) ? 'A' : 'B')
  }
  private setHovered(side: Side | null) {
    if (side === this.hovered) return
    this.hovered = side
    this.opts.callbacks.onHover?.(side)
    this.syncTone()
    this.dirty = true
  }
  private onPointerLeave = () => { if (!this.orbit.dragging) this.setHovered(null) }

  /** Clicks only start playback via the play sprite; voting is A / B (keys or the buttons). A drag never counts. */
  private onClick = (e: MouseEvent) => {
    if (!this.ready || this.orbit.moved) return
    this.setPointer(e)
    const { play } = this.hitTest()
    if (play) this.opts.callbacks.onPlay()
  }

  /** Damped orbit; after a few idle seconds it eases back to the default framing (snaps under reduced motion). */
  private stepOrbit(dt: number) {
    const o = this.orbit, reduced = this.opts.reducedMotion
    if (!o.dragging) o.idle += dt
    if (!o.dragging && o.idle > 3.5) {
      if (reduced) { o.tYaw = 0; o.tPitch = 0 }
      else { const k = Math.exp(-dt * 2.2); o.tYaw *= k; o.tPitch *= k }
    }
    if (reduced) { o.yaw = o.tYaw; o.pitch = o.tPitch }
    else { const k = 1 - Math.exp(-dt * 9); o.yaw += (o.tYaw - o.yaw) * k; o.pitch += (o.tPitch - o.pitch) * k }
    return Math.abs(o.yaw - o.tYaw) > 1e-4 || Math.abs(o.pitch - o.tPitch) > 1e-4 || Math.abs(o.yaw) > 1e-4
  }

  private onContextLost = (e: Event) => {
    e.preventDefault()
    this.opts.callbacks.onContextLost?.()
  }

  private tick = (now: number) => {
    if (this.disposed || !this.active) return
    this.raf = requestAnimationFrame(this.tick)
    const dt = Math.min((now - this.last) / 1000, 0.08); this.last = now
    const reduced = this.opts.reducedMotion
    this.adaptPixelRatio(dt)
    this.clock += dt
    let animating = false

    if (this.tweens.length) {
      animating = true
      const done: Tween[] = []
      for (const tw of this.tweens) {
        tw.t += dt
        const k = Math.min(1, tw.t / tw.dur)
        tw.update(tw.ease(k))
        if (k >= 1) done.push(tw)
      }
      if (done.length) { this.tweens = this.tweens.filter(t => !done.includes(t)); done.forEach(t => t.resolve()) }
    }

    if (this.playing && this.ready) { this.matchTime += dt; animating = true }

    for (const st of [this.sides.A, this.sides.B]) if (st.t < 1) { this.stepTone(st, reduced ? 1 : dt); animating = true }

    if (this.stepOrbit(dt)) animating = true

    // Fairy flare: head flight, dust, ribbon flash; VS punch decay.
    this.stepFlare(dt)
    if (!reduced) { this.vsPunch *= Math.exp(-dt * 6); this.vsFlip *= Math.exp(-dt * 4.5) }

    // Hover / keyboard-focus rim, smoothed.
    for (const lane of this.lanes) {
      const target = (this.flash === lane.side || this.hovered === lane.side || this.focused === lane.side) ? 1 : 0
      const next = reduced ? target : THREE.MathUtils.lerp(lane.hoverV, target, 1 - Math.exp(-dt * 12))
      if (Math.abs(next - lane.hoverV) > 1e-3) animating = true
      lane.hoverV = Math.abs(next - target) < 1e-3 ? target : next
    }

    if (this.particles.visible) {
      animating = true
      this.particleAge += dt
      const pos = this.particles.geometry.getAttribute('position') as THREE.BufferAttribute
      for (let i = 0; i < pos.count; i++) {
        this.particleVel[i * 3 + 1] -= 4.2 * dt
        pos.setXYZ(i, pos.getX(i) + this.particleVel[i * 3] * dt, Math.max(0.01, pos.getY(i) + this.particleVel[i * 3 + 1] * dt), pos.getZ(i) + this.particleVel[i * 3 + 2] * dt)
      }
      pos.needsUpdate = true
      const m = this.particles.material as THREE.PointsMaterial
      m.opacity = Math.max(0, 1 - this.particleAge / 1.4)
      if (this.particleAge > 1.4) this.particles.visible = false
    }

    if (!reduced) animating = true // idle drift + VS bob
    if (!animating && !this.dirty) return
    this.dirty = false
    this.lastDt = dt
    this.render()
  }

  private updateScene() {
    const reduced = this.opts.reducedMotion
    for (const lane of this.lanes) {
      lane.lift.position.y = lane.liftV
      lane.actor.group.scale.setScalar(Math.max(0.0001, lane.scaleV))
      lane.actor.setTime(this.matchTime)
      lane.actor.setLook(lane.hoverV, lane.dimV)
      lane.spot.intensity = lane.spotV * 1.6
      const dur = lane.actor.duration
      const p = dur > 0 ? (this.matchTime % dur) / dur : 0
      lane.progress.geometry.setDrawRange(0, 6 * Math.round(p * lane.progressSegs))
      ;(lane.captionMesh.material as THREE.MeshBasicMaterial).opacity = 1 - lane.dimV * 0.55
      ;(lane.badge.material as THREE.SpriteMaterial).opacity = 1 - lane.dimV * 0.55
      lane.badge.scale.setScalar(0.24 * (1 + lane.hoverV * 0.18) * Math.max(0.0001, lane.scaleV))
    }
    if (this.xpPop.visible) {
      const k = this.xpPopV
      ;(this.xpPop.material as THREE.SpriteMaterial).opacity = reduced ? 1 : Math.min(1, k * 5) * (1 - Math.max(0, k - 0.7) / 0.3)
      this.xpPop.position.y = this.xpPopY + (reduced ? 0 : k * 0.14)
    }
    // VS medallion: slow sway; camera: gentle vertical drift + slight vertical pointer parallax.
    const c = this.clock
    if (!reduced) {
      this.vs.rotation.z = Math.sin(c * 1.3) * 0.035
      this.vs.scale.setScalar(1 + Math.sin(c * 2.1) * 0.015 + this.vsPunch)
      this.vsBody.rotation.y = Math.sin(c * 0.9) * 0.08 + this.vsFlip // idle turn shows the coin's edge; a vote flips it
    }
    // Orbit around the arena centre. The camera always looks at the centre and yaw is clamped
    // (±ORBIT_YAW_MAX), so A stays left of the screen centre and B right: the A | B split line
    // (screen centre) stays valid without re-projecting.
    const az = this.orbit.yaw
    const el = 0.13 + this.orbit.pitch + (reduced ? 0 : Math.sin(c * 0.09) * 0.02)
    const d = this.camDist
    this.camera.position.set(
      this.camTarget.x + Math.sin(az) * Math.cos(el) * d,
      this.camTarget.y + Math.sin(el) * d,
      this.camTarget.z + Math.cos(az) * Math.cos(el) * d,
    )
    this.camera.lookAt(this.camTarget)
  }

  /** Robot flares follow each lane's fastest limb; dimmed with a losing robot, hidden while it scales in/out. */
  private updateRobotFlares() {
    if (!this.robotFlares.length) return
    this.camera.updateMatrixWorld()
    const dpr = this.renderer.getPixelRatio()
    this.lanes.forEach((lane, i) => {
      const flare = this.robotFlares[i]
      if (!flare) return
      const shown = lane.actor.group.visible && lane.scaleV > 0.98
      if (!shown) { flare.reset(); flare.points.visible = false; return }
      flare.update(this.lastDt, dpr, this.camera, this.height, 1 - lane.dimV * 0.7)
    })
  }

  private render() {
    this.updateScene()
    this.updateRobotFlares()
    this.world.setSplitSchemes(this.worldSplit) // copies the live per-half colours + split line (no allocation)
    this.world.render(this.camera, this.lastDt)
    this.lastDt = 0
  }
}
