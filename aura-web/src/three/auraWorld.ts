/**
 * Aura world: a stylised, palette-locked environment that sits behind the
 * motion viewers (sky dome, ground, distant mesas) plus an ink-outline
 * composite pass.
 *
 * Technique (own implementation, inspired by hand-inked anime backgrounds):
 *  - World surfaces are flat palette fills, not lit with an N·L ramp. Cast
 *    shadows on the ground swap to a hue-shifted, darker version of the base
 *    colour with a hard cut, instead of multiplying toward grey.
 *  - Sky gradient and clouds are posterized: noise is thresholded into two
 *    tones with anti-aliased steps, so it reads as painted shapes.
 *  - One full-screen pass draws ink lines from depth discontinuities. Creases
 *    come from the second difference of 1/depth (exactly zero on planes),
 *    silhouettes from depth jumps. Lines on world surfaces are broken up by
 *    world-space noise; lines on the robot stay solid so it reads clearly.
 *  - World materials write alpha 0 into the HDR target, everything else alpha
 *    1, so the composite tone-maps only the robot and keeps the world on the
 *    exact palette.
 *
 * One call per viewer: `createAuraWorld(scene, renderer, opts)`, then
 * `world.render(camera, dt)` instead of `renderer.render(scene, camera)`.
 */
import * as THREE from 'three'
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { createWorldProps, mulberry32, type WorldProps, type WorldPropsOptions } from './worldProps'

/** Brand palette (aura-web/src/palette.css). */
export const AURA_PALETTE = {
  eggshell: '#f4f1de',
  navy: '#3d405b',
  sunset: '#f2cc8f',
  sage: '#81b29a',
  terra: '#e07a5f',
} as const

export type AuraWorldOptions = {
  /** World units per metre-ish. 1 for real-scale G1 (1.3 m), ~2 for the 2.7-unit stage robot. */
  scale?: number
  /** Ground height in world units. */
  groundY?: number
  /** Disables cloud drift and animated grain. Defaults to the media query. */
  reducedMotion?: boolean
  /** Draw a soft hue-shifted blob under the focus point (for viewers without shadow maps). */
  contactShadow?: boolean
  /** Draw the dashed arena ring around the focus point. */
  ring?: boolean
  /**
   * Compile the per-half scheme variant up front (see `setSplitSchemes`), so the
   * first call does not recompile. Off by default: the shaders are then exactly
   * the unsplit ones.
   */
  split?: boolean
  /**
   * MSAA samples of the composite target. Default 4. 'auto' uses 4 below a
   * pixel ratio of 1.5 and none above it: on high-DPR screens the 4x resolve of
   * a full-screen half-float target is the dominant cost (hero at 2880x1800:
   * 40 -> 117 fps), and the pixel density plus the ink pass already hide aliasing.
   */
  msaa?: number | 'auto'
  /**
   * Keep the near mesa ring clear in this direction (yaw of the ground direction,
   * radians, atan2(z, x)) +- halfAngle, so subjects in front of the camera stand
   * against sky / far hills. Off by default (all viewers keep the full ring).
   */
  sightline?: { yaw: number; halfAngle: number }
  /**
   * Interactive fairy-world props (flower tufts, butterflies, toadstools, firefly wisps, tap sparkles; see
   * worldProps.ts). Off by default. They listen to pointer events on the canvas
   * passively and stay outside `keepOut` (and the sightline sector).
   */
  props?: WorldPropsOptions
}

/**
 * One colour scheme for the world (linear working-space colours). Build one with
 * `auraWorldScheme(tone)`, blend with `lerpAuraWorldScheme` (allocation-free).
 */
export type AuraWorldScheme = {
  /** Horizon haze: sky at the horizon, ground and mesa fog. */
  haze: THREE.Color
  skyLow: THREE.Color
  skyHigh: THREE.Color
  cloud: THREE.Color
  cloudShade: THREE.Color
  sunCol: THREE.Color
  glow: THREE.Color
  ground: THREE.Color
  groundLight: THREE.Color
  /** Ground shadow / ring colour. */
  groundInk: THREE.Color
  accent: THREE.Color
  /** Mesa recolour: rgb = tint, `mesaAmount` = how far the vertex colours move toward it (0 = palette). */
  mesaTint: THREE.Color
  mesaAmount: number
  /** Composite ink-line colour. */
  ink: THREE.Color
  /** Composite vignette colour. */
  vignette: THREE.Color
}
export type AuraWorldTone = 'light' | 'dark' | 'feedback'
export type AuraWorldSplit = {
  /** Scheme of the left half (fragments left of the split line). */
  a: AuraWorldScheme
  /** Scheme of the right half. */
  b: AuraWorldScheme
  /**
   * Split polyline in drawing-buffer px, GL y-up, sorted by ascending y (at most
   * AURA_SPLIT_MAX_POINTS). Below the first / above the last point the line
   * continues straight. Same convention as the hero's ribbon centre line.
   */
  pts?: ArrayLike<THREE.Vector2>
  /** Straight vertical split at this x (drawing-buffer px). Default: the buffer's centre. */
  splitX?: number
}
const AURA_SPLIT_MAX_POINTS = 32

/**
 * Objects on this layer (and not on layer 0) are drawn after the ink/grade
 * composite, straight to the screen: use it for thin helper lines such as
 * skeleton previews so they keep their colour and are never inked over.
 */
export const AURA_OVERLAY_LAYER = 5

/** The world's fixed sun (world space, unit). Shared by the backdrop, props and stylized characters (toonStylize.ts). */
export const AURA_SUN_DIR = new THREE.Vector3(-0.62, 0.17, -0.77).normalize()

export type AuraWorld = {
  /** Render one frame (replaces renderer.render). dt in seconds. */
  render(camera: THREE.PerspectiveCamera, dt?: number): void
  /** Centre of the arena ring / contact shadow, in world XZ. */
  setFocus(x: number, z: number, propsKeepOut?: number): void
  /** Position of the fake contact shadow (contactShadow option), in world XZ. */
  setShadow(x: number, z: number): void
  /** Pause-when-offscreen helper: observe the viewer element. */
  observe(el: Element): void
  /** False while the observed element is off-screen or the tab is hidden. */
  readonly visible: boolean
  /**
   * Optional per-half colour schemes: sky, clouds, ground, mesas and the ink
   * composite each take the scheme of the half they fall in (screen-space test
   * against `pts` / `splitX`). Colours are copied, so call it every frame while
   * tweening; no allocations, and no recompiles if created with `split: true`.
   * `null` restores the default single-scheme look.
   */
  setSplitSchemes(split: AuraWorldSplit | null): void
  dispose(): void
}

import { prefersReducedMotion } from '../motionPrefs'
export { prefersReducedMotion }

// ---------------------------------------------------------------------------
// Shared GLSL helpers (hash/value-noise/fbm, hsv shift, anti-aliased step)
// ---------------------------------------------------------------------------
const GLSL_COMMON = /* glsl */ `
float aw_hash(vec3 p) {
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}
float aw_noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = aw_hash(i), n100 = aw_hash(i + vec3(1, 0, 0));
  float n010 = aw_hash(i + vec3(0, 1, 0)), n110 = aw_hash(i + vec3(1, 1, 0));
  float n001 = aw_hash(i + vec3(0, 0, 1)), n101 = aw_hash(i + vec3(1, 0, 1));
  float n011 = aw_hash(i + vec3(0, 1, 1)), n111 = aw_hash(i + vec3(1, 1, 1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
float aw_fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 4; i++) { s += a * aw_noise(p); p = p * 2.03 + vec3(17.1, 3.7, 9.2); a *= 0.5; }
  return s / 0.9375;
}
// anti-aliased threshold: crisp painted edge without shimmering
float aw_step(float edge, float x) {
  float w = max(fwidth(x), 1e-4) * 0.75;
  return smoothstep(edge - w, edge + w, x);
}
vec3 aw_rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}
vec3 aw_hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * clamp(p - 1.0, 0.0, 1.0);
}
// shadow tone: rotate hue toward blue, drop value, keep saturation
vec3 aw_shade(vec3 c, float amount) {
  vec3 h = aw_rgb2hsv(c);
  h.x = fract(h.x + 0.035 * amount);
  h.y = min(1.0, h.y * (1.0 + 0.08 * amount));
  h.z *= 1.0 - 0.38 * amount;
  return aw_hsv2rgb(h);
}
`

// Haze colour shared by sky (at the horizon) and fog so ground, mesas and sky
// melt into one band. Sunset with a touch of terra.
const lin = (hex: string) => new THREE.Color(hex) // THREE.Color converts sRGB hex to linear working space
function mixHex(a: string, b: string, t: number) { return lin(a).lerp(lin(b), t) }

function palette() {
  return {
    eggshell: lin(AURA_PALETTE.eggshell),
    navy: lin(AURA_PALETTE.navy),
    sunset: lin(AURA_PALETTE.sunset),
    sage: lin(AURA_PALETTE.sage),
    terra: lin(AURA_PALETTE.terra),
    haze: mixHex(AURA_PALETTE.sunset, AURA_PALETTE.eggshell, 0.28).lerp(lin(AURA_PALETTE.terra), 0.08),
    skyLow: mixHex(AURA_PALETTE.sage, AURA_PALETTE.eggshell, 0.42),
    skyHigh: mixHex(AURA_PALETTE.sage, AURA_PALETTE.navy, 0.1),
    ground: mixHex(AURA_PALETTE.sage, AURA_PALETTE.navy, 0.4),
    groundLight: mixHex(AURA_PALETTE.sage, AURA_PALETTE.navy, 0.2),
  }
}

// ---------------------------------------------------------------------------
// Optional per-half schemes (only compiled in with the AW_SPLIT define)
// ---------------------------------------------------------------------------
const GLSL_SPLIT = /* glsl */ `
#ifdef AW_SPLIT
#define AW_SPLIT_N ${AURA_SPLIT_MAX_POINTS}
uniform vec2 uSplitPts[ AW_SPLIT_N ];
uniform int uSplitCount;
// 0 = left half (scheme a), 1 = right half (scheme b)
float aw_side( vec2 p ) {
  float x = uSplitPts[ 0 ].x;
  if ( p.y > uSplitPts[ 0 ].y ) {
    for ( int i = 1; i < AW_SPLIT_N; i++ ) {
      if ( i >= uSplitCount ) break;
      vec2 a = uSplitPts[ i - 1 ], b = uSplitPts[ i ];
      x = b.x;
      if ( p.y <= b.y ) { x = mix( a.x, b.x, clamp( ( p.y - a.y ) / max( b.y - a.y, 1e-3 ), 0.0, 1.0 ) ); break; }
    }
  }
  return step( x, p.x );
}
#define AW_SIDE( a, b ) mix( ( a ), ( b ), awSide )
#endif
`

/** Default (light) scheme: exactly the colours the unsplit world uses. */
function lightScheme(p: ReturnType<typeof palette>): AuraWorldScheme {
  return {
    haze: p.haze.clone(), skyLow: p.skyLow.clone(), skyHigh: p.skyHigh.clone(),
    cloud: p.eggshell.clone(), cloudShade: mixHex(AURA_PALETTE.eggshell, AURA_PALETTE.sage, 0.38),
    sunCol: p.terra.clone(), glow: p.sunset.clone(),
    ground: p.ground.clone(), groundLight: p.groundLight.clone(), groundInk: p.navy.clone(), accent: p.sunset.clone(),
    mesaTint: p.navy.clone(), mesaAmount: 0,
    ink: p.navy.clone(), vignette: p.navy.clone(),
  }
}

/**
 * Palette-derived world schemes. `light` is the default look; `dark` is a navy
 * dusk (navy sky / ground / mesas, eggshell ink); `feedback` is sunset-based
 * with navy ink. Every scheme keeps distinct two-tone pairs (sky bands, cloud
 * lit / shade, ground base / patch) so the posterized look survives; shadows
 * still use the hue-shifted shade in the shaders.
 */
export function auraWorldScheme(tone: AuraWorldTone = 'light'): AuraWorldScheme {
  const p = palette()
  if (tone === 'light') return lightScheme(p)
  const { eggshell: E, navy: N, sunset: S, sage: G, terra: T } = AURA_PALETTE
  const deep = '#1d1f33' // navy pushed toward night, for the shade side of the dark scheme
  if (tone === 'dark') {
    return {
      haze: mixHex(N, G, 0.24).lerp(lin(S), 0.06),
      skyLow: mixHex(N, G, 0.12),
      skyHigh: mixHex(N, deep, 0.45),
      cloud: mixHex(N, E, 0.22).lerp(lin(S), 0.03),
      cloudShade: mixHex(N, E, 0.08),
      sunCol: mixHex(S, E, 0.4),
      glow: mixHex(N, E, 0.1),
      ground: mixHex(N, deep, 0.2),
      groundLight: mixHex(N, G, 0.16),
      groundInk: lin(deep),
      accent: lin(S),
      mesaTint: mixHex(N, G, 0.2),
      mesaAmount: 0.97,
      ink: lin(E),
      vignette: lin(deep),
    }
  }
  return {
    haze: mixHex(S, E, 0.4),
    skyLow: mixHex(S, E, 0.18),
    skyHigh: mixHex(S, T, 0.32),
    cloud: lin(E),
    cloudShade: mixHex(E, T, 0.3),
    sunCol: lin(T),
    glow: mixHex(E, S, 0.4),
    ground: mixHex(S, T, 0.3),
    groundLight: mixHex(S, E, 0.22),
    groundInk: mixHex(T, N, 0.45),
    accent: lin(E),
    mesaTint: mixHex(T, S, 0.35),
    mesaAmount: 0.45,
    ink: lin(N),
    vignette: mixHex(T, N, 0.5),
  }
}

const SCHEME_COLOR_KEYS = ['haze', 'skyLow', 'skyHigh', 'cloud', 'cloudShade', 'sunCol', 'glow', 'ground', 'groundLight', 'groundInk', 'accent', 'mesaTint', 'ink', 'vignette'] as const

export function copyAuraWorldScheme(out: AuraWorldScheme, src: AuraWorldScheme) {
  for (const k of SCHEME_COLOR_KEYS) out[k].copy(src[k])
  out.mesaAmount = src.mesaAmount
  return out
}

/** out = mix(a, b, t), allocation-free (out may alias a or b). */
export function lerpAuraWorldScheme(out: AuraWorldScheme, a: AuraWorldScheme, b: AuraWorldScheme, t: number) {
  for (const k of SCHEME_COLOR_KEYS) out[k].lerpColors(a[k], b[k], t)
  out.mesaAmount = a.mesaAmount + (b.mesaAmount - a.mesaAmount) * t
  return out
}

// ---------------------------------------------------------------------------
// Sky dome
// ---------------------------------------------------------------------------
function skyMaterial(p: ReturnType<typeof palette>, sunDir: THREE.Vector3) {
  return new THREE.ShaderMaterial({
    name: 'AuraWorldSky',
    uniforms: {
      uTime: { value: 0 },
      uSun: { value: sunDir },
      uHaze: { value: p.haze },
      uLow: { value: p.skyLow },
      uHigh: { value: p.skyHigh },
      uCloud: { value: p.eggshell },
      uCloudShade: { value: mixHex(AURA_PALETTE.eggshell, AURA_PALETTE.sage, 0.38) },
      uSunCol: { value: p.terra },
      uGlow: { value: p.sunset },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vDir = wp.xyz - cameraPosition;
        gl_Position = projectionMatrix * viewMatrix * wp;
        gl_Position.z = gl_Position.w; // pin to the far plane
      }`,
    fragmentShader: /* glsl */ `
      ${GLSL_COMMON}
      uniform float uTime;
      uniform vec3 uSun, uHaze, uLow, uHigh, uCloud, uCloudShade, uSunCol, uGlow;
      ${GLSL_SPLIT}
      #ifdef AW_SPLIT
      uniform vec3 uHazeB, uLowB, uHighB, uCloudB, uCloudShadeB, uSunColB, uGlowB;
      #endif
      varying vec3 vDir;
      void main() {
        #ifdef AW_SPLIT
        float awSide = aw_side(gl_FragCoord.xy);
        vec3 sHaze = AW_SIDE(uHaze, uHazeB), sLow = AW_SIDE(uLow, uLowB), sHigh = AW_SIDE(uHigh, uHighB);
        vec3 sCloud = AW_SIDE(uCloud, uCloudB), sCloudShade = AW_SIDE(uCloudShade, uCloudShadeB);
        vec3 sSunCol = AW_SIDE(uSunCol, uSunColB), sGlow = AW_SIDE(uGlow, uGlowB);
        #else
        vec3 sHaze = uHaze, sLow = uLow, sHigh = uHigh, sCloud = uCloud, sCloudShade = uCloudShade, sSunCol = uSunCol, sGlow = uGlow;
        #endif
        vec3 d = normalize(vDir);
        float h = d.y;
        // stepped gradient: haze -> low sky -> high sky, boundaries wobbled by noise
        float wob = aw_fbm(vec3(d.xz * 3.0, 1.7)) - 0.5;
        float b1 = aw_step(0.045 + wob * 0.03, h);
        float b2 = aw_step(0.2 + wob * 0.06, h);
        vec3 col = mix(sHaze, sLow, b1);
        col = mix(col, mix(sLow, sHigh, 0.55), b2 * (1.0 - aw_step(0.48 + wob * 0.08, h)));
        col = mix(col, sHigh, aw_step(0.48 + wob * 0.08, h));

        // sun: crisp disc + one stepped halo ring, warm glow band on the horizon below it
        float sd = dot(d, normalize(uSun));
        float halo = aw_step(0.985, sd);
        col = mix(col, mix(col, sGlow, 0.65), halo);
        col = mix(col, sSunCol, aw_step(0.9975, sd));
        float glow = smoothstep(0.55, 1.0, sd) * (1.0 - smoothstep(0.0, 0.16, h));
        col = mix(col, sGlow, aw_step(0.5, glow + wob * 0.2) * 0.6);

        // posterized clouds on a virtual plane, two tones (lit / shaded underside)
        if (h > 0.02) {
          vec2 uv = d.xz / (h + 0.18) * 0.55;
          vec3 q = vec3(uv + vec2(uTime * 0.012, uTime * 0.004), uTime * 0.01);
          float n = aw_fbm(q * 1.6);
          float n2 = aw_fbm(q * 1.6 + vec3(0.06, -0.09, 0.0));
          float band = smoothstep(0.03, 0.2, h) * (1.0 - smoothstep(0.55, 0.85, h));
          float body = aw_step(0.6, n * band + 0.12 * band);
          float shade = body * (1.0 - aw_step(0.62, n2 * band + 0.16 * band));
          col = mix(col, sCloud, body);
          col = mix(col, sCloudShade, shade);
        }
        gl_FragColor = vec4(col, 0.0);
        #ifdef AW_DIRECT
        gl_FragColor.a = 1.0; // no composite: opaque, so a transparent canvas still shows the world
        #endif
        #include <colorspace_fragment>
      }`,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: true,
    toneMapped: false,
    fog: false,
    blending: THREE.NoBlending,
  })
}

// ---------------------------------------------------------------------------
// Ground: flat fills + noise patches + hue-shifted cast shadow + fog
// ---------------------------------------------------------------------------
function groundMaterial(p: ReturnType<typeof palette>, scale: number, radius: number, opts: { contact: boolean; ring: boolean }) {
  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.lights,
    {
      uBase: { value: p.ground },
      uPatch: { value: p.groundLight },
      uInk: { value: p.navy },
      uAccent: { value: p.sunset },
      uHaze: { value: p.haze },
      uFocus: { value: new THREE.Vector2() },
      uShadowPos: { value: new THREE.Vector2() },
      uScale: { value: scale },
      uFogDensity: { value: 1.9 / radius },
      uContact: { value: opts.contact ? 1 : 0 },
      uRing: { value: opts.ring ? 1 : 0 },
    },
  ])
  return new THREE.ShaderMaterial({
    name: 'AuraWorldGround',
    uniforms,
    lights: true,
    vertexShader: /* glsl */ `
      #include <common>
      #include <shadowmap_pars_vertex>
      varying vec3 vWorld;
      void main() {
        #include <beginnormal_vertex>
        #include <defaultnormal_vertex>
        #include <begin_vertex>
        #include <project_vertex>
        #include <worldpos_vertex>
        #include <shadowmap_vertex>
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
      }`,
    fragmentShader: /* glsl */ `
      #include <common>
      #include <packing>
      #include <lights_pars_begin>
      #include <shadowmap_pars_fragment>
      #include <shadowmask_pars_fragment>
      ${GLSL_COMMON}
      uniform vec3 uBase, uPatch, uInk, uAccent, uHaze;
      uniform vec2 uFocus, uShadowPos;
      uniform float uScale, uFogDensity, uContact, uRing;
      ${GLSL_SPLIT}
      #ifdef AW_SPLIT
      uniform vec3 uBaseB, uPatchB, uInkB, uAccentB, uHazeB;
      #endif
      varying vec3 vWorld;
      void main() {
        #ifdef AW_SPLIT
        float awSide = aw_side(gl_FragCoord.xy);
        vec3 sBase = AW_SIDE(uBase, uBaseB), sPatch = AW_SIDE(uPatch, uPatchB), sInk = AW_SIDE(uInk, uInkB);
        vec3 sAccent = AW_SIDE(uAccent, uAccentB), sHaze = AW_SIDE(uHaze, uHazeB);
        #else
        vec3 sBase = uBase, sPatch = uPatch, sInk = uInk, sAccent = uAccent, sHaze = uHaze;
        #endif
        vec2 w = vWorld.xz / uScale;
        vec2 rel = (vWorld.xz - uFocus) / uScale;
        float r = length(rel);

        // two-tone painted patches + sparse light flecks
        float n = aw_fbm(vec3(w * 0.16, 0.0));
        vec3 col = mix(sBase, sPatch, aw_step(0.56, n));
        float fleck = aw_noise(vec3(w * 7.0, 3.0));
        col = mix(col, mix(sPatch, sAccent, 0.2), aw_step(0.93, fleck) * 0.35 * (1.0 - smoothstep(3.0, 8.0, r)));

        // arena ring: dashed ink circle + faint inner disc, centred on the subject
        if (uRing > 0.5) {
          float ang = atan(rel.y, rel.x);
          float dash = aw_step(0.42, fract(ang * 9.549)); // 60 dashes
          float ringMask = (1.0 - aw_step(0.022, abs(r - 1.9))) * dash;
          col = mix(col, mix(col, sInk, 0.55), ringMask);
          col = mix(col, mix(col, sPatch, 0.55), 1.0 - aw_step(1.6, r + (n - 0.5) * 0.15));
        }

        // cast shadow from the scene's shadow-casting lights, hard cut, hue-shifted
        float lit = getShadowMask();
        // fake contact shadow for viewers without shadow maps
        vec2 srel = (vWorld.xz - uShadowPos) / uScale;
        lit = min(lit, mix(1.0, aw_step(0.5, length(srel / vec2(0.5, 0.4)) + (n - 0.5) * 0.08), uContact));
        vec3 shadowCol = mix(aw_shade(col, 0.7), sInk, 0.3);
        col = mix(shadowCol, col, smoothstep(0.35, 0.65, lit));

        // exp2 distance fog into the horizon haze
        float dist = length(vWorld - cameraPosition);
        float f = 1.0 - exp(-pow(dist * uFogDensity, 2.0));
        col = mix(col, sHaze, clamp(f, 0.0, 1.0));
        gl_FragColor = vec4(col, 0.0);
        #ifdef AW_DIRECT
        gl_FragColor.a = 1.0; // no composite: opaque, so a transparent canvas still shows the world
        #endif
        #include <colorspace_fragment>
      }`,
    toneMapped: false,
    fog: false,
    blending: THREE.NoBlending,
  })
}

// ---------------------------------------------------------------------------
// Distant backdrop (fairy hills, groves, castle): one merged faceted geometry, two-tone sun facing, heavy fog
// ---------------------------------------------------------------------------
function mesaGeometry(scale: number, p: ReturnType<typeof palette>, sightline?: { yaw: number; halfAngle: number }) {
  // Fairy-world backdrop (one merged, faceted mesh): soft rolling hills far out, rounded
  // tree groves in the mid ring, and a distant domed castle. No pointed shapes.
  const rand = mulberry32(7)
  const parts: THREE.BufferGeometry[] = []
  const hillCols = [p.navy.clone().lerp(p.sage, 0.45), p.sage.clone().lerp(p.navy, 0.15), p.navy.clone().lerp(p.sage, 0.65)]
  const leafCols = [p.sage, p.sage.clone().lerp(p.eggshell, 0.3), p.sage.clone().lerp(p.navy, 0.25), p.sunset.clone().lerp(p.sage, 0.35), p.terra.clone().lerp(p.sunset, 0.4)]
  const trunkCol = p.navy.clone().lerp(p.terra, 0.35)
  const add = (geo: THREE.BufferGeometry, color: THREE.Color) => {
    const flat = geo.index ? geo.toNonIndexed() : geo
    if (flat !== geo) geo.dispose()
    flat.deleteAttribute('uv')
    flat.deleteAttribute('normal')
    const count = flat.getAttribute('position').count
    const colors = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) colors.set([color.r, color.g, color.b], i * 3)
    flat.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    parts.push(flat)
  }
  const blob = (r: number, sx: number, sy: number, sz: number, x: number, y: number, z: number, rot: number, color: THREE.Color) => {
    const g = new THREE.IcosahedronGeometry(r, 1)
    g.scale(sx, sy, sz); g.rotateY(rot); g.translate(x, y, z)
    add(g, color)
  }
  const inSight = (a: number, r: number, radius: number) => {
    if (!sightline) return false
    const d = Math.abs(Math.atan2(Math.sin(a - sightline.yaw), Math.cos(a - sightline.yaw)))
    return d < sightline.halfAngle + Math.atan(radius / r)
  }
  // far ring: soft rolling hills (flattened, overlapping domes sunk into the ground)
  for (let i = 0; i < 20; i++) {
    const a = (i / 20) * Math.PI * 2 + rand() * 0.25
    const r = (46 + rand() * 12) * scale
    const R = (9 + rand() * 8) * scale
    const sy = 0.22 + rand() * 0.16
    blob(R, 1.3, sy, 1, Math.cos(a) * r, -R * sy * 0.35, Math.sin(a) * r, rand() * 6.28, hillCols[i % hillCols.length])
  }
  // mid ring: groves of round-canopy trees (kept out of the sightline sector)
  for (let gI = 0; gI < 14; gI++) {
    const a = (gI / 14) * Math.PI * 2 + rand() * 0.3
    const r = (27 + rand() * 9) * scale
    const trees = 2 + Math.floor(rand() * 3)
    const skip = inSight(a, r, 3.2 * scale)
    for (let t = 0; t < trees; t++) {
      const off = (rand() - 0.5) * 4.5 * scale, dep = (rand() - 0.5) * 3 * scale
      const h = (1.6 + rand() * 1.8) * scale
      const cr = (0.9 + rand() * 0.8) * scale
      const leaf = leafCols[Math.floor(rand() * leafCols.length)]
      const rot = rand() * 6.28
      if (skip) continue
      const cx = Math.cos(a) * r - Math.sin(a) * off + Math.cos(a) * dep
      const cz = Math.sin(a) * r + Math.cos(a) * off + Math.sin(a) * dep
      const trunk = new THREE.CylinderGeometry(0.12 * scale, 0.18 * scale, h, 6, 1)
      trunk.translate(cx, h / 2, cz)
      add(trunk, trunkCol)
      blob(cr, 1, 0.9, 1, cx, h + cr * 0.45, cz, rot, leaf)
      blob(cr * 0.7, 1, 0.85, 1, cx + cr * 0.6, h + cr * 0.1, cz + cr * 0.2, rot, leaf)
      blob(cr * 0.62, 1, 0.85, 1, cx - cr * 0.55, h + cr * 0.15, cz - cr * 0.25, rot, leaf)
    }
  }
  // a distant fairy castle: round towers with domed caps (left-back, never behind the subject)
  {
    const a = -Math.PI / 2 - 0.85, r = 52 * scale
    if (!inSight(a, r, 6 * scale)) {
      const bx = Math.cos(a) * r, bz = Math.sin(a) * r
      const wall = p.eggshell.clone().lerp(p.sunset, 0.35), roof = p.terra.clone().lerp(p.sunset, 0.2)
      const towers: [number, number, number][] = [[0, 7, 1.4], [-3.2, 4.8, 1.0], [3, 5.4, 1.1], [-1.4, 3.4, 2.6], [1.6, 3.0, 2.2]]
      for (const [dx, h, rad] of towers) {
        const x = bx + dx * scale * Math.sin(a) * -1, z = bz + dx * scale * Math.cos(a)
        const tw = new THREE.CylinderGeometry(rad * 0.85 * scale, rad * scale, h * scale, 8, 1)
        tw.translate(x, h * scale / 2, z); add(tw, wall)
        const cap = new THREE.SphereGeometry(rad * 1.05 * scale, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2)
        cap.scale(1, 1.25, 1); cap.translate(x, h * scale, z); add(cap, roof)
      }
    }
  }
  const merged = mergeGeometries(parts, false)
  parts.forEach(g => g.dispose())
  return merged
}

function mesaMaterial(p: ReturnType<typeof palette>, sunDir: THREE.Vector3, scale: number) {
  return new THREE.ShaderMaterial({
    name: 'AuraWorldMesas',
    uniforms: {
      uSun: { value: sunDir },
      uHaze: { value: p.haze },
      uScale: { value: scale },
    },
    vertexColors: true,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      ${GLSL_COMMON}
      uniform vec3 uSun, uHaze;
      uniform float uScale;
      ${GLSL_SPLIT}
      #ifdef AW_SPLIT
      uniform vec3 uHazeB;
      uniform vec4 uMesaTint, uMesaTintB; // rgb tint, a = amount
      #endif
      varying vec3 vWorld;
      varying vec3 vColor;
      void main() {
        #ifdef AW_SPLIT
        float awSide = aw_side(gl_FragCoord.xy);
        vec3 sHaze = AW_SIDE(uHaze, uHazeB);
        vec4 mt = AW_SIDE(uMesaTint, uMesaTintB);
        // recolour toward the tint, keeping each mesa's relative lightness
        float vLum = dot(vColor, vec3(0.2126, 0.7152, 0.0722));
        vec3 sColor = mix(vColor, mt.rgb * (0.55 + 1.1 * vLum), mt.a);
        #else
        vec3 sHaze = uHaze, sColor = vColor;
        #endif
        vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld))); // faceted normal
        float wob = aw_noise(vWorld / uScale * 0.6) - 0.5;
        float lit = aw_step(0.1, dot(n, normalize(uSun)) + wob * 0.25);
        vec3 col = mix(aw_shade(sColor, 1.0), sColor, lit);
        // painted strata on the mesa walls
        // painted banding, only on steep faces (trunks, tower walls); domes and canopies stay clean
        float strata = aw_step(0.72, aw_noise(vec3(vWorld.xz / uScale * 0.4, vWorld.y / uScale * 2.2)));
        col = mix(col, aw_shade(col, 0.45), strata * 0.35 * (1.0 - smoothstep(0.3, 0.6, abs(n.y))));
        // atmospheric perspective: distance fog plus a low ground-hugging haze layer
        float dist = length(vWorld.xz - cameraPosition.xz) / uScale;
        float f = smoothstep(18.0, 58.0, dist) * 0.7 + 0.22;
        f = max(f, (1.0 - smoothstep(0.0, 1.6, vWorld.y / uScale)) * 0.6);
        col = mix(col, sHaze, clamp(f, 0.0, 0.94));
        gl_FragColor = vec4(col, 0.0);
        #ifdef AW_DIRECT
        gl_FragColor.a = 1.0; // no composite: opaque, so a transparent canvas still shows the world
        #endif
        #include <colorspace_fragment>
      }`,
    toneMapped: false,
    fog: false,
    blending: THREE.NoBlending,
  })
}

// ---------------------------------------------------------------------------
// Ink outline + grade composite
// ---------------------------------------------------------------------------
function compositeMaterial(p: ReturnType<typeof palette>) {
  return new THREE.ShaderMaterial({
    name: 'AuraWorldComposite',
    uniforms: {
      tColor: { value: null },
      tDepth: { value: null },
      uTexel: { value: new THREE.Vector2(1, 1) },
      uThickness: { value: 1 },
      uNear: { value: 0.1 },
      uFar: { value: 100 },
      uProjInv: { value: new THREE.Matrix4() },
      uCamWorld: { value: new THREE.Matrix4() },
      uScale: { value: 1 },
      uInk: { value: p.navy },
      uToneMap: { value: 1 },
      uExposure: { value: 1 },
      uTime: { value: 0 },
      uGrain: { value: 1 },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: /* glsl */ `
      ${GLSL_COMMON}
      uniform sampler2D tColor;
      uniform sampler2D tDepth;
      uniform vec2 uTexel;
      uniform float uThickness, uNear, uFar, uScale, uToneMap, uExposure, uTime, uGrain;
      uniform mat4 uProjInv, uCamWorld;
      uniform vec3 uInk;
      ${GLSL_SPLIT}
      #ifdef AW_SPLIT
      uniform vec3 uInkB, uVignette, uVignetteB;
      #endif
      varying vec2 vUv;

      float viewDepth(vec2 uv) {
        float d = texture2D(tDepth, uv).x;
        return (uNear * uFar) / (uFar - d * (uFar - uNear)); // positive eye distance
      }
      vec3 acesFit(vec3 c) {
        // RRT+ODT fit (Narkowicz-style), matches the robot's ACES look closely enough
        c *= uExposure * 0.6;
        return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
      }
      void main() {
        #ifdef AW_SPLIT
        float awSide = aw_side(gl_FragCoord.xy);
        vec3 sInk = AW_SIDE(uInk, uInkB), sVignette = AW_SIDE(uVignette, uVignetteB);
        #else
        vec3 sInk = uInk, sVignette = uInk;
        #endif
        vec4 src = texture2D(tColor, vUv);
        float isObject = clamp(src.a, 0.0, 1.0);
        // alpha 0.5 = painted character (toonStylize.ts): exact palette (no tone map), solid ink, calmer creases
        float isChar = 1.0 - smoothstep(0.06, 0.14, abs(src.a - 0.5));
        vec3 col = mix(src.rgb, uToneMap > 0.5 ? acesFit(src.rgb) : src.rgb, isObject * (1.0 - isChar));

        float raw = texture2D(tDepth, vUv).x;
        vec2 o = uTexel * uThickness;
        float zc = viewDepth(vUv);
        float zl = viewDepth(vUv - vec2(o.x, 0.0)), zr = viewDepth(vUv + vec2(o.x, 0.0));
        float zb = viewDepth(vUv - vec2(0.0, o.y)), zt = viewDepth(vUv + vec2(0.0, o.y));

        // creases: second difference of 1/z is zero on any plane
        float ic = 1.0 / zc;
        float crease = (abs(1.0 / zl + 1.0 / zr - 2.0 * ic) + abs(1.0 / zb + 1.0 / zt - 2.0 * ic)) / ic;
        // silhouettes: only on the near side of a depth jump, so lines hug the front object
        float jump = max(max(zl, zr), max(zb, zt)) - zc;
        float sil = smoothstep(0.06, 0.16, jump / zc);
        // features thinner than the kernel (1px skeleton lines, wires) keep their own colour
        float tj = 0.06 * zc;
        float thin = max(step(tj, zl - zc) * step(tj, zr - zc), step(tj, zb - zc) * step(tj, zt - zc));
        sil *= 1.0 - thin;
        float edge = max(smoothstep(mix(0.012, 0.05, isChar), mix(0.04, 0.12, isChar), crease), sil);
        edge *= step(raw, 0.99999); // never on the sky

        // world position for stable, world-anchored line breakup
        vec4 vp = uProjInv * vec4(vUv * 2.0 - 1.0, raw * 2.0 - 1.0, 1.0);
        vec3 wp = (uCamWorld * vec4(vp.xyz / vp.w, 1.0)).xyz;
        float broken = step(0.32, aw_noise(wp / uScale * 2.6));
        float fade = 1.0 - smoothstep(8.0, 60.0, zc / uScale) * 0.7;
        float strength = mix(0.62 * broken * fade, 0.95, max(isObject, isChar));
        col = mix(col, sInk, edge * strength);

        // grade: soft navy vignette + fine grain (static when reduced motion)
        vec2 q = vUv - 0.5;
        col = mix(col, sVignette, smoothstep(0.42, 0.95, length(q * vec2(1.1, 1.0))) * 0.16);
        float g = aw_hash(vec3(gl_FragCoord.xy, floor(uTime * 24.0))) - 0.5;
        col += g * 0.018 * uGrain;
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------
const overlayProbe = new THREE.Layers()
overlayProbe.set(AURA_OVERLAY_LAYER)
function overlayCount(scene: THREE.Scene) {
  let n = 0
  for (const child of scene.children) if (child.visible && child.layers.test(overlayProbe)) n++
  return n
}

/**
 * Per-half scheme controller (additive, inert by default). Adds the side-B and
 * split uniforms to the world materials up front (unused until the AW_SPLIT
 * define is set, so the default shaders and output are unchanged). Enabling
 * swaps the side-A uniform values to its own scheme colours; disabling puts the
 * original values back. Per-frame calls only copy numbers.
 */
function createSplitController(
  p: ReturnType<typeof palette>,
  mats: { sky: THREE.ShaderMaterial; ground: THREE.ShaderMaterial; mesa: THREE.ShaderMaterial; comp: THREE.ShaderMaterial | null },
  initiallyOn: boolean,
) {
  const A = lightScheme(p), B = lightScheme(p)
  const pts = Array.from({ length: AURA_SPLIT_MAX_POINTS }, () => new THREE.Vector2())
  const count = { value: 2 }
  const ptsU = { value: pts }
  const mesaTintA = new THREE.Vector4(A.mesaTint.r, A.mesaTint.g, A.mesaTint.b, 0)
  const mesaTintB = new THREE.Vector4(B.mesaTint.r, B.mesaTint.g, B.mesaTint.b, 0)
  let autoX = true
  let on = false
  const all = [mats.sky, mats.ground, mats.mesa, ...(mats.comp ? [mats.comp] : [])]
  for (const m of all) { m.uniforms.uSplitPts = ptsU; m.uniforms.uSplitCount = count }

  type Key = Exclude<keyof AuraWorldScheme, 'mesaAmount'>
  // [material uniforms, side-A uniform name, scheme key]; side B is `${name}B`
  const bind: [Record<string, THREE.IUniform>, string, Key][] = [
    [mats.sky.uniforms, 'uHaze', 'haze'], [mats.sky.uniforms, 'uLow', 'skyLow'], [mats.sky.uniforms, 'uHigh', 'skyHigh'],
    [mats.sky.uniforms, 'uCloud', 'cloud'], [mats.sky.uniforms, 'uCloudShade', 'cloudShade'],
    [mats.sky.uniforms, 'uSunCol', 'sunCol'], [mats.sky.uniforms, 'uGlow', 'glow'],
    [mats.ground.uniforms, 'uBase', 'ground'], [mats.ground.uniforms, 'uPatch', 'groundLight'], [mats.ground.uniforms, 'uInk', 'groundInk'],
    [mats.ground.uniforms, 'uAccent', 'accent'], [mats.ground.uniforms, 'uHaze', 'haze'],
    [mats.mesa.uniforms, 'uHaze', 'haze'],
  ]
  if (mats.comp) {
    mats.comp.uniforms.uVignette = { value: mats.comp.uniforms.uInk.value }
    bind.push([mats.comp.uniforms, 'uInk', 'ink'], [mats.comp.uniforms, 'uVignette', 'vignette'])
  }
  const saved = bind.map(([u, name]) => u[name].value as unknown)
  for (const [u, name, key] of bind) u[`${name}B`] = { value: B[key] }
  mats.mesa.uniforms.uMesaTint = { value: mesaTintA }
  mats.mesa.uniforms.uMesaTintB = { value: mesaTintB }

  const setDefine = (enable: boolean) => {
    for (const m of all) {
      if (enable) m.defines.AW_SPLIT = ''
      else delete m.defines.AW_SPLIT
      m.needsUpdate = true
    }
  }
  const setOn = (enable: boolean) => {
    if (enable === on) return
    on = enable
    bind.forEach(([u, name, key], i) => { u[name].value = enable ? A[key] : saved[i] })
    setDefine(enable)
  }
  const vertical = (x: number) => {
    pts[0].set(x, 0); pts[1].set(x, 1)
    count.value = 2
  }
  if (initiallyOn) setOn(true)
  vertical(0)

  return {
    set(split: AuraWorldSplit | null) {
      if (!split) { setOn(false); return }
      copyAuraWorldScheme(A, split.a)
      copyAuraWorldScheme(B, split.b)
      mesaTintA.set(A.mesaTint.r, A.mesaTint.g, A.mesaTint.b, A.mesaAmount)
      mesaTintB.set(B.mesaTint.r, B.mesaTint.g, B.mesaTint.b, B.mesaAmount)
      const src = split.pts
      if (src && src.length > 0) {
        const n = Math.min(src.length, AURA_SPLIT_MAX_POINTS)
        for (let i = 0; i < n; i++) pts[i].copy(src[i])
        if (n === 1) { pts[1].set(pts[0].x, pts[0].y + 1) }
        count.value = Math.max(2, n)
        autoX = false
      } else if (split.splitX !== undefined) {
        vertical(split.splitX)
        autoX = false
      } else {
        autoX = true
      }
      setOn(true)
    },
    /** Extra world materials (props) that share the ground's split uniforms: toggle AW_SPLIT with the rest. */
    add(ms: THREE.ShaderMaterial[]) {
      for (const m of ms) {
        all.push(m)
        if (on) { m.defines.AW_SPLIT = ''; m.needsUpdate = true }
      }
    },
    /** Per frame: keep the default centre split on the drawing buffer's centre. */
    frame(renderer: THREE.WebGLRenderer, size: THREE.Vector2) {
      if (!on || !autoX) return
      renderer.getDrawingBufferSize(size)
      vertical(size.x / 2)
    },
  }
}

/** The composite needs WebGL2 with renderable float colour buffers; otherwise the world renders directly. */
function auraCompositeSupported(renderer: THREE.WebGLRenderer) {
  const gl = renderer.getContext()
  return typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext
    && (renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float'))
}

/**
 * Renderer for an Aura world viewer. The composite draws the scene into its own MSAA
 * target, so the canvas only receives a full-screen quad and the overlay layer: no canvas
 * MSAA then (as in the hero). Without float colour buffers the world renders straight to
 * the canvas, so the renderer is rebuilt with antialias on.
 */
export function createAuraRenderer(params: THREE.WebGLRendererParameters = {}) {
  const renderer = new THREE.WebGLRenderer({ ...params, antialias: false })
  if (auraCompositeSupported(renderer)) return renderer
  renderer.dispose()
  renderer.forceContextLoss()
  return new THREE.WebGLRenderer({ ...params, antialias: true })
}

export function createAuraWorld(scene: THREE.Scene, renderer: THREE.WebGLRenderer, opts: AuraWorldOptions = {}): AuraWorld {
  const scale = opts.scale ?? 1
  const reduced = opts.reducedMotion ?? prefersReducedMotion()
  const p = palette()
  const sunDir = AURA_SUN_DIR.clone()

  const group = new THREE.Group()
  group.name = 'AuraWorld'
  scene.add(group)
  scene.background = null
  scene.fog = null

  const skyGeo = new THREE.SphereGeometry(1, 32, 16)
  const skyMat = skyMaterial(p, sunDir)
  const sky = new THREE.Mesh(skyGeo, skyMat)
  sky.name = 'AuraWorldSky'
  sky.frustumCulled = false
  sky.renderOrder = -1000
  group.add(sky)

  const groundRadius = 70 * scale
  const groundGeo = new THREE.CircleGeometry(groundRadius, 64)
  groundGeo.rotateX(-Math.PI / 2)
  const groundMat = groundMaterial(p, scale, groundRadius * 0.55, { contact: !!opts.contactShadow, ring: opts.ring ?? true })
  const ground = new THREE.Mesh(groundGeo, groundMat)
  ground.name = 'AuraWorldGround'
  ground.position.y = opts.groundY ?? 0
  ground.receiveShadow = true
  ground.renderOrder = -999
  group.add(ground)

  const mesaGeo = mesaGeometry(scale, p, opts.sightline)
  const mesaMat = mesaMaterial(p, sunDir, scale)
  const mesas = new THREE.Mesh(mesaGeo, mesaMat)
  mesas.name = 'AuraWorldMesas'
  mesas.position.y = opts.groundY ?? 0
  mesas.renderOrder = -998
  group.add(mesas)

  // Composite: HDR (half float) target with MSAA + depth texture. Any failure
  // to set it up (no float colour buffers, older drivers) falls back to direct
  // rendering; the world materials still look right on their own.
  let postEnabled = auraCompositeSupported(renderer)
  let target: THREE.WebGLRenderTarget | null = null
  let quad: FullScreenQuad | null = null
  let compMat: THREE.ShaderMaterial | null = null
  const samplesFor = () => opts.msaa === 'auto' ? (renderer.getPixelRatio() >= 1.5 ? 0 : 4) : (opts.msaa ?? 4)
  const makeTarget = (samples: number) => {
    const depthTexture = new THREE.DepthTexture(1, 1)
    depthTexture.type = THREE.UnsignedIntType
    return new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      samples,
      depthBuffer: true,
      depthTexture,
    })
  }
  if (postEnabled) {
    try {
      target = makeTarget(samplesFor())
      compMat = compositeMaterial(p)
      compMat.uniforms.uGrain.value = 1
      compMat.uniforms.uScale.value = scale
      quad = new FullScreenQuad(compMat)
    } catch {
      postEnabled = false
      target?.dispose(); target = null
      compMat?.dispose(); compMat = null
    }
  }

  // Direct-render fallback: world surfaces write opaque alpha (the alpha-0 tag
  // only matters to the composite). Not set when the composite runs.
  if (!postEnabled) for (const m of [skyMat, groundMat, mesaMat]) m.defines.AW_DIRECT = ''

  const splitCtl = createSplitController(p, { sky: skyMat, ground: groundMat, mesa: mesaMat, comp: compMat }, !!opts.split)

  // Props share the ground's uniforms by reference (incl. the *B scheme twins and split points).
  let props: WorldProps | null = null
  if (opts.props) {
    props = createWorldProps({
      renderer, scene, group, overlayLayer: AURA_OVERLAY_LAYER, scale, groundY: opts.groundY ?? 0, reduced, direct: !postEnabled,
      glslCommon: GLSL_COMMON, glslSplit: GLSL_SPLIT, ground: groundMat.uniforms, sunDir,
      palette: { terra: p.terra, sunset: p.sunset, eggshell: p.eggshell, navy: p.navy },
      sightline: opts.sightline,
    }, opts.props)
    splitCtl.add(props.materials)
  }

  let time = 0
  let visible = true
  let inView = true
  let io: IntersectionObserver | null = null
  const onVisibility = () => { visible = inView && document.visibilityState !== 'hidden' }
  document.addEventListener('visibilitychange', onVisibility)

  const size = new THREE.Vector2()
  const prevClear = new THREE.Color()

  const render = (camera: THREE.PerspectiveCamera, dt = 0) => {
    if (!reduced) time += Math.min(dt, 0.1)
    skyMat.uniforms.uTime.value = time
    // sky follows the camera and sits just inside the far plane
    sky.position.copy(camera.position)
    sky.scale.setScalar(camera.far * 0.92)
    sky.updateMatrixWorld()
    splitCtl.frame(renderer, size)
    props?.update(camera, dt)

    if (!postEnabled || !target || !quad || !compMat) {
      camera.layers.enable(AURA_OVERLAY_LAYER)
      renderer.render(scene, camera)
      camera.layers.disable(AURA_OVERLAY_LAYER)
      return
    }
    renderer.getDrawingBufferSize(size)
    const w = Math.max(1, Math.floor(size.x)), h = Math.max(1, Math.floor(size.y))
    if (opts.msaa === 'auto' && target.samples !== samplesFor()) {
      // pixel ratio crossed the threshold (screen change / adaptive DPR): rebuild with the right sample count
      target.depthTexture?.dispose(); target.dispose()
      target = makeTarget(samplesFor())
    }
    if (target.width !== w || target.height !== h) target.setSize(w, h)

    const prevTarget = renderer.getRenderTarget()
    renderer.getClearColor(prevClear)
    const prevAlpha = renderer.getClearAlpha()
    renderer.setClearColor(p.haze, 0)
    renderer.setRenderTarget(target)
    renderer.clear()
    renderer.render(scene, camera)
    renderer.setRenderTarget(prevTarget)
    renderer.setClearColor(prevClear, prevAlpha)

    const u = compMat.uniforms
    u.tColor.value = target.texture
    u.tDepth.value = target.depthTexture
    u.uTexel.value.set(1 / w, 1 / h)
    u.uThickness.value = Math.max(1, h / 640)
    u.uNear.value = camera.near
    u.uFar.value = camera.far
    u.uProjInv.value.copy(camera.projectionMatrixInverse)
    u.uCamWorld.value.copy(camera.matrixWorld)
    u.uToneMap.value = renderer.toneMapping === THREE.NoToneMapping ? 0 : 1
    u.uExposure.value = renderer.toneMappingExposure
    u.uTime.value = time
    u.uGrain.value = reduced ? 0.6 : 1
    quad.render(renderer)

    if (overlayCount(scene) > 0) {
      const prevAuto = renderer.autoClear
      const prevMask = camera.layers.mask
      const prevBg = scene.background
      renderer.autoClear = false
      camera.layers.set(AURA_OVERLAY_LAYER)
      renderer.clearDepth()
      // the main pass already refreshed the shadow maps this frame
      const prevShadowAuto = renderer.shadowMap.autoUpdate
      renderer.shadowMap.autoUpdate = false
      renderer.render(scene, camera)
      renderer.shadowMap.autoUpdate = prevShadowAuto
      camera.layers.mask = prevMask
      renderer.autoClear = prevAuto
      scene.background = prevBg
    }
  }

  return {
    render,
    setFocus(x: number, z: number, propsKeepOut?: number) {
      props?.layout(x, z, propsKeepOut)
      groundMat.uniforms.uFocus.value.set(x, z)
      groundMat.uniforms.uShadowPos.value.set(x, z)
    },
    setShadow(x: number, z: number) { groundMat.uniforms.uShadowPos.value.set(x, z) },
    observe(el: Element) {
      io?.disconnect()
      if (typeof IntersectionObserver === 'undefined') return
      io = new IntersectionObserver(entries => {
        for (const e of entries) inView = e.isIntersecting
        onVisibility()
      }, { rootMargin: '120px' })
      io.observe(el)
    },
    get visible() { return visible },
    setSplitSchemes(split: AuraWorldSplit | null) { splitCtl.set(split) },
    dispose() {
      document.removeEventListener('visibilitychange', onVisibility)
      io?.disconnect()
      scene.remove(group)
      skyGeo.dispose(); skyMat.dispose()
      groundGeo.dispose(); groundMat.dispose()
      mesaGeo.dispose(); mesaMat.dispose()
      target?.depthTexture?.dispose()
      target?.dispose()
      compMat?.dispose()
      quad?.dispose()
      props?.dispose()
    },
  }
}
