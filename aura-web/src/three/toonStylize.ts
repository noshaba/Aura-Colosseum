/**
 * toonStylize: paint a character (the Blossom Fairy, the G1 robot, any lit mesh)
 * into the Aura world's flat-palette language.
 *
 * The fairy's photographic look is mostly her texture (a photo bake with
 * lighting and shading baked in) plus a fine normal map. So the main lever is a
 * one-off texture bake, cached and shared by every instance:
 *   1. flatten the baked lighting: compress luminance toward a mid value, keep hue
 *      (per pixel only: the texture is a UV atlas of islands on black gutters, so
 *      blurs would bleed islands into each other);
 *   2. pull each pixel toward the nearest colour of an extended palette built
 *      only from eggshell / navy / sunset / sage / terra mixes (so skin, hair,
 *      pink flowers, white outfit and wings stay distinguishable);
 *   3. posterize value into a few bands.
 * The normal map is dropped (no fine shading, and it keeps the ink pass calm).
 *
 * On top, a light shader layer (chained onto any existing onBeforeCompile, so
 * skinning, morphs and wing animation keep working): hard two-tone light from a
 * key on the sun's azimuth mirrored to the camera side (the world's sun,
 * AURA_SUN_DIR, sits low behind the subjects and would leave them in shade),
 * hue-kept shadows (hue nudged, value down, like the
 * world); a warm rim where edges face the real sun; a soft rim; and a stronger
 * rim from the existing emissive (hover highlight). It writes alpha 0.5, which tells the
 * world's composite "painted character": no tone mapping (exact palette), solid
 * ink silhouettes, and calmer crease lines (see auraWorld composite).
 */
import * as THREE from 'three'
import { AURA_PALETTE, AURA_SUN_DIR } from './auraWorld'

export type ToonStylizeHandle = { dispose(): void }

/** Soft rim amount 0..1. */
const RIM = 0.16
/** How strongly texture colours snap to the palette, 0..1. */
const PULL = 0.3
/** Value bands of the bake (applied at partial strength; skin is not banded). */
const BANDS = 4

// ---------------------------------------------------------------------------- bake
const hex = (h: string) => { const c = new THREE.Color(h); return [c.r, c.g, c.b] } // linear
const mixc = (a: number[], b: number[], t: number) => a.map((v, i) => v + (b[i] - v) * t)
const P = {
  egg: hex(AURA_PALETTE.eggshell), navy: hex(AURA_PALETTE.navy), sun: hex(AURA_PALETTE.sunset),
  sage: hex(AURA_PALETTE.sage), terra: hex(AURA_PALETTE.terra),
}
/** Extended palette: the five brand colours plus a few mixes for skin, hair, pinks and leaves. */
const EXT = [
  P.egg, P.navy, P.sun, P.sage, P.terra,
  mixc(P.terra, P.navy, 0.48), // dark auburn (hair shade)
  mixc(P.terra, mixc(P.navy, P.sun, 0.25), 0.38), // auburn (hair)
  mixc(P.terra, P.navy, 0.3), // warm brown (skin, hair light)
  mixc(P.terra, P.navy, 0.42), // deep brown (skin shade)
  mixc(P.sun, P.terra, 0.45), // tan (skin)
  mixc(P.terra, P.egg, 0.55), // rose pink (flowers, ribbons)
  mixc(P.sage, P.navy, 0.45), // deep leaf green
  mixc(P.egg, P.sage, 0.3), // pale green-white (wings)
  mixc(P.egg, P.sun, 0.35), // cream (outfit shade)
  mixc(P.egg, P.navy, 0.3), // warm light grey (sneakers, straps)
]
const toSrgb = (v: number) => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055)
const EXT_SRGB = EXT.map(c => c.map(toSrgb))

/**
 * Baked textures by source texture uuid, ref-counted per stylized material. A bake whose
 * source belongs to a session-cached asset (the fairy GLB: `keepBake`) is kept after its
 * last user is gone: viewers mount and unmount as the user navigates (Studio ->
 * Methodology -> Studio), and re-baking her atlas is a full per-pixel pass on the main
 * thread each time. Tradeoff: that one <= 1024 px canvas and its texture (about 4 MB of
 * CPU memory, plus the GPU copy while a renderer holds it) stay resident for the session.
 * Per-viewer sources (the G1 marble, XBot maps: a fresh texture per mount) are still
 * disposed with their last user, so they never pile up.
 */
type BakeEntry = { texture: THREE.Texture; refs: number; keep: boolean }
const bakeCache = new Map<string, BakeEntry>()

function bakeTexture(src: THREE.Texture): THREE.Texture | null {
  const pull = PULL, bands = BANDS
  const img = src.image as (CanvasImageSource & { width: number; height: number }) | undefined
  if (!img || !img.width || !img.height || typeof document === 'undefined') return null
  const scale = Math.min(1, 1024 / Math.max(img.width, img.height))
  const W = Math.max(1, Math.round(img.width * scale)), H = Math.max(1, Math.round(img.height * scale))
  const make = (w: number, h: number) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c }
  const full = make(W, H)
  const fctx = full.getContext('2d', { willReadFrequently: true })
  if (!fctx) return null
  fctx.drawImage(img, 0, 0, W, H)
  // Per-pixel only: the fairy's texture is a UV atlas of small islands on black
  // gutters, so any blur would bleed neighbouring islands / gutters into each other.
  let data: ImageData
  try { data = fctx.getImageData(0, 0, W, H) } catch { return null } // tainted canvas: skip the bake
  const px = data.data
  const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b
  const smooth = (e0: number, e1: number, x: number) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t) }
  for (let i = 0; i < px.length; i += 4) {
    let r = px[i] / 255, g = px[i + 1] / 255, b = px[i + 2] / 255
    const L0 = lum(r, g, b)
    px[i + 3] = 255 // opaque: low-alpha island-edge texels round-trip through the canvas as rainbow garbage
    if (L0 < 0.02) continue // gutters
    // Skin-like warm browns (r > g > b, moderate warmth, mid value; also catches auburn hair):
    // keep them close to her real tone, even out the baked photo shading, and don't posterize
    // them (value bands split skin gradients into darker islands).
    const warm = r - b
    const skinK = (r > g && g > b ? 1 : 0)
      * smooth(0.05, 0.1, warm) * (1 - smooth(0.42, 0.55, warm))
      * smooth(0.06, 0.12, L0) * (1 - smooth(0.72, 0.85, L0))
    // 1) flatten the baked photo lighting: compress luminance toward a mid value, keep hue.
    // Non-skin: lift deep shadows a little and tame highlights. Skin: compress harder toward its
    // own mid tone, so baked knee / elbow shading doesn't read as dark patches.
    const LfOther = L0 < 0.45 ? L0 + (0.45 - L0) * 0.18 : 0.45 + (L0 - 0.45) * 0.7
    const LfSkin = 0.4 + (L0 - 0.4) * 0.5
    const Lf = LfOther + (LfSkin - LfOther) * skinK
    const k = Lf / L0
    r = Math.min(1, r * k); g = Math.min(1, g * k); b = Math.min(1, b * k)
    // 2) nearest extended-palette colour (sRGB distance), pulled toward at the pixel's own value
    let best = 0, bd = 1e9
    for (let j = 0; j < EXT_SRGB.length; j++) {
      const c = EXT_SRGB[j]
      const dr = r - c[0], dg = g - c[1], db = b - c[2]
      const d = dr * dr * 0.9 + dg * dg * 1.2 + db * db * 0.8
      if (d < bd) { bd = d; best = j }
    }
    const t = EXT_SRGB[best]
    const L = lum(r, g, b), Lt = lum(t[0], t[1], t[2])
    const s = Lt > 1e-3 ? Math.min(1.5, Math.max(0.7, L / Lt)) : 1
    const chroma0 = Math.max(r, g, b) - Math.min(r, g, b)
    const pullHere = pull * (1 - 0.7 * skinK)
    r += (t[0] * s - r) * pullHere; g += (t[1] * s - g) * pullHere; b += (t[2] * s - b) * pullHere
    // near-neutral source pixels (whites, greys) stay near-neutral: no tinted casts
    if (chroma0 < 0.12) {
      const Lp = lum(r, g, b), keep = 0.35 + 0.65 * (chroma0 / 0.12)
      r = Lp + (r - Lp) * keep; g = Lp + (g - Lp) * keep; b = Lp + (b - Lp) * keep
    }
    // 3) posterize value into a few wide bands, at partial strength (texture noise near a band
    // edge then only nudges, instead of flipping between bands); skin is left smooth
    const L2 = lum(r, g, b)
    const q = (Math.floor(L2 * bands) + 0.55) / bands
    const Lq = L2 + (q - L2) * 0.6 * (1 - skinK)
    const f = L2 > 1e-3 ? Lq / L2 : 1
    px[i] = Math.max(0, Math.min(255, r * f * 255))
    px[i + 1] = Math.max(0, Math.min(255, g * f * 255))
    px[i + 2] = Math.max(0, Math.min(255, b * f * 255))
  }
  fctx.putImageData(data, 0, 0)
  const tex = new THREE.CanvasTexture(full)
  tex.colorSpace = src.colorSpace
  tex.flipY = src.flipY
  tex.wrapS = src.wrapS; tex.wrapT = src.wrapT
  tex.channel = src.channel
  tex.offset.copy(src.offset); tex.repeat.copy(src.repeat); tex.rotation = src.rotation; tex.center.copy(src.center)
  tex.matrixAutoUpdate = src.matrixAutoUpdate
  tex.anisotropy = Math.max(1, src.anisotropy)
  tex.name = `${src.name || 'map'}_aura_painted`
  return tex
}

function acquireBake(src: THREE.Texture, keep: boolean) {
  let entry = bakeCache.get(src.uuid)
  if (!entry) {
    const texture = bakeTexture(src)
    if (!texture) return null
    entry = { texture, refs: 0, keep }
    bakeCache.set(src.uuid, entry)
  }
  entry.keep ||= keep
  entry.refs++
  return { key: src.uuid, texture: entry.texture }
}
function releaseBake(key: string) {
  const entry = bakeCache.get(key)
  if (!entry || --entry.refs > 0 || entry.keep) return
  entry.texture.dispose()
  bakeCache.delete(key)
}

// ---------------------------------------------------------------------------- shader layer
const STY_FRAG_HEAD = /* glsl */ `
uniform vec3 stySun;
uniform vec3 styKey;
uniform vec3 styRimCol;
uniform vec3 stySunCol;
uniform float styRim;
uniform float styMapBias;
// Shadow tone like the world (value down, hue nudged cool, saturation kept), done
// without an HSV round-trip: HSV hue is unstable on near-neutral whites and painted
// rainbow speckles on the sneakers / shorts. Value scales down; a slight cool tint
// plus a small chroma boost stands in for the hue nudge.
vec3 sty_shade(vec3 c) {
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  vec3 chroma = c - l;
  vec3 s = l + chroma * 1.08;
  // cool tint like the world for most colours; warm, true-to-tone shade for skin-like warm hues
  float warmK = smoothstep( 0.02, 0.1, c.r - c.b ) * step( c.b, c.g );
  return s * mix( vec3( 0.71, 0.73, 0.82 ), vec3( 0.76, 0.72, 0.69 ), warmK );
}
`
const STY_MAP = /* glsl */ `
#ifdef USE_MAP
  vec4 sampledDiffuseColor = texture2D( map, vMapUv, styMapBias ); // painterly: a slightly softer mip
  diffuseColor *= sampledDiffuseColor;
#endif
`
const STY_OUT = /* glsl */ `
{
  vec3 base = diffuseColor.rgb;
  // two-tone from the key (the sun's azimuth, mirrored to the camera side: the real sun sits
  // low behind the subjects, which would leave every character in shade); the sun itself
  // paints a warm rim on the edges that face it
  vec3 keyV = normalize( ( viewMatrix * vec4( styKey, 0.0 ) ).xyz );
  vec3 sunV = normalize( ( viewMatrix * vec4( stySun, 0.0 ) ).xyz );
  // wide enough that the scanned mesh's slightly noisy normals don't flicker into patches
  float lit = smoothstep( -0.16, 0.16, dot( normal, keyV ) );
  // (no shadow-map term: self-shadowing from hair / arms painted noisy patches on skin)
  vec3 col = mix( sty_shade( base ), base, lit );
  float facing = abs( dot( normal, normalize( vViewPosition ) ) );
  float rim = smoothstep( 0.62, 0.92, 1.0 - facing );
  float hl = smoothstep( 0.03, 0.2, dot( totalEmissiveRadiance, vec3( 0.3333 ) ) ); // hover highlight from setLook()
  float sunRim = rim * smoothstep( 0.1, 0.4, dot( normal, sunV ) );
  col = mix( col, stySunCol, sunRim * 0.55 );
  col = mix( col, styRimCol, rim * clamp( styRim + 0.55 * hl, 0.0, 1.0 ) );
  col += hl * 0.04;
  outgoingLight = col;
}
#include <opaque_fragment>
gl_FragColor.a = 0.5; // composite tag: painted character (no tone map, solid ink, calm creases)
`

const STY_WING_VERT_HEAD = /* glsl */ `
attribute float styWing;
varying float vStyWing;
`
const STY_WING_FRAG = /* glsl */ `
#ifdef STY_WING
  // wings: stylised panes in the palette (pale sage-white, a touch of sky), inked at the edge by the composite
  float wingK = smoothstep( 0.35, 0.65, vStyWing );
  base = mix( base, styPane * ( 0.92 + 0.08 * dot( base, vec3( 0.333 ) ) ), wingK * 0.85 );
#endif
`

type Patched = { prevCompile: THREE.Material['onBeforeCompile']; prevKey: THREE.Material['customProgramCacheKey']; prevNormal: THREE.Texture | null; prevMap: THREE.Texture | null; prevToneMapped: boolean }

const isLit = (m: THREE.Material) =>
  (m as THREE.MeshToonMaterial).isMeshToonMaterial || (m as THREE.MeshStandardMaterial).isMeshStandardMaterial ||
  (m as THREE.MeshLambertMaterial).isMeshLambertMaterial || (m as THREE.MeshPhongMaterial).isMeshPhongMaterial

export type ToonStylizeMaterialOptions = {
  /** Geometry carries a `styWing` attribute (0..1, see markWingWeights): paint those parts as wing panes. */
  wings?: boolean
  /** The map belongs to a session-cached asset: keep its bake after the last user (see bakeCache). */
  keepBake?: boolean
}

/**
 * Per-vertex wing weight from skin weights: the summed influence of bones whose
 * names match `re` (the fairy's skeletal wings: wing_L, wing_L_tip, wing_R,
 * wing_R_tip). Stored once on the (shared) geometry as `styWing`.
 */
export function markWingWeights(mesh: THREE.SkinnedMesh, re = /wing_[LR]/i) {
  const geo = mesh.geometry
  if (geo.getAttribute('styWing')) return true
  const si = geo.getAttribute('skinIndex'), sw = geo.getAttribute('skinWeight')
  if (!si || !sw || !mesh.skeleton) return false
  const wingIdx = new Set<number>()
  mesh.skeleton.bones.forEach((b, i) => { if (re.test(b.name)) wingIdx.add(i) })
  if (!wingIdx.size) return false
  const out = new Float32Array(si.count)
  for (let v = 0; v < si.count; v++) {
    let w = 0
    for (let k = 0; k < 4; k++) if (wingIdx.has(si.getComponent(v, k))) w += sw.getComponent(v, k)
    out[v] = w
  }
  geo.setAttribute('styWing', new THREE.BufferAttribute(out, 1))
  return true
}

/** Stylize one lit material (no-op if already stylized). dispose() restores it and releases its bake. */
export function toonStylizeMaterial(m: THREE.Material, opts: ToonStylizeMaterialOptions = {}): ToonStylizeHandle {
  if (!isLit(m) || m.userData.toonStylized) return { dispose() {} }
  const mat = m as THREE.MeshToonMaterial
  const rec: Patched = { prevCompile: m.onBeforeCompile, prevKey: m.customProgramCacheKey, prevNormal: mat.normalMap ?? null, prevMap: mat.map ?? null, prevToneMapped: m.toneMapped }
  m.userData.toonStylized = true
  let bakeKey: string | null = null
  if (mat.map) {
    const b = acquireBake(mat.map, !!opts.keepBake)
    if (b) { mat.map = b.texture; bakeKey = b.key }
  }
  if (mat.normalMap) mat.normalMap = null // no fine shading (keeps the ink pass calm)
  const uniforms = {
    stySun: { value: AURA_SUN_DIR.clone() },
    styKey: { value: new THREE.Vector3(AURA_SUN_DIR.x, 0.55, -AURA_SUN_DIR.z).normalize() },
    stySunCol: { value: new THREE.Color(AURA_PALETTE.sunset) },
    styRimCol: { value: new THREE.Color(AURA_PALETTE.eggshell) },
    styPane: { value: new THREE.Color(AURA_PALETTE.eggshell).lerp(new THREE.Color(AURA_PALETTE.sage), 0.32) },
    styRim: { value: RIM },
    styMapBias: { value: 0 },
  }
  m.toneMapped = false // direct-render fallback: keep the palette exact too
  if (opts.wings) m.defines = { ...(m.defines ?? {}), STY_WING: '' }
  const prev = rec.prevCompile
  m.onBeforeCompile = (shader, renderer) => {
    prev.call(m, shader, renderer) // keep earlier patches (outlines, ...)
    Object.assign(shader.uniforms, uniforms)
    if (opts.wings) {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\n' + STY_WING_VERT_HEAD)
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vStyWing = styWing;')
    }
    let fs = shader.fragmentShader
    fs = fs.replace('#include <common>', '#include <common>\n' + STY_FRAG_HEAD + (opts.wings ? 'varying float vStyWing;\nuniform vec3 styPane;\n' : ''))
    fs = fs.replace('#include <map_fragment>', STY_MAP)
    fs = fs.replace('#include <opaque_fragment>', STY_OUT.replace('  vec3 base = diffuseColor.rgb;\n', '  vec3 base = diffuseColor.rgb;\n' + STY_WING_FRAG))
    shader.fragmentShader = fs
  }
  const prevKey = rec.prevKey
  m.customProgramCacheKey = () => `${prevKey.call(m)}|aura-toon-v5${opts.wings ? 'w' : ''}`
  m.needsUpdate = true

  let done = false
  const handle = {
    dispose() {
      if (done) return
      done = true
      m.onBeforeCompile = rec.prevCompile
      m.customProgramCacheKey = rec.prevKey
      if ('map' in mat) mat.map = rec.prevMap
      if ('normalMap' in mat) mat.normalMap = rec.prevNormal
      m.toneMapped = rec.prevToneMapped
      if (m.defines) delete m.defines.STY_WING
      delete m.userData.toonStylized
      m.needsUpdate = true
      if (bakeKey) releaseBake(bakeKey)
    },
  }
  return handle
}

/**
 * Stylize every lit material under `root` (once per material). Returns a handle
 * whose dispose() restores the materials and releases the baked textures.
 */
export function toonStylize(root: THREE.Object3D): ToonStylizeHandle {
  const handles: ToonStylizeHandle[] = []
  const seen = new Set<THREE.Material>()
  root.traverse(o => {
    const mesh = o as THREE.Mesh
    if (!mesh.isMesh || !mesh.material) return
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (seen.has(m)) continue
      seen.add(m)
      handles.push(toonStylizeMaterial(m))
    }
  })
  return { dispose() { handles.forEach(h => h.dispose()); handles.length = 0 } }
}
