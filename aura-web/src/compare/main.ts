/**
 * Dev-only side-by-side check of Aura's splat renderer (three/lightweightSplat.ts) against a
 * reference renderer (GaussianSplats3D), on the same .splat and the same camera.
 * Open /compare.html on the Vite dev server; not part of the app build.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { DropInViewer, SceneFormat } from '@mkkellogg/gaussian-splats-3d'
import { createLightweightSplat } from '../three/lightweightSplat'
import './compare.css'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const statusEl = $<HTMLParagraphElement>('status')
const setStatus = (text: string) => { statusEl.textContent = text }

// One camera drives both views; both canvases are the same size, so the image is pixel-aligned.
const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.01, 500)
camera.position.set(0, 0, 5)

function makeView(host: HTMLElement) {
  // preserveDrawingBuffer lets the comparison read pixels back; antialias off on both sides.
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true })
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.setPixelRatio(1)
  host.appendChild(renderer.domElement)
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  const controls = new OrbitControls(camera, renderer.domElement)
  return { host, renderer, scene, controls }
}
const auraView = makeView($('aura'))
const refView = makeView($('ref'))

const resize = () => {
  const w = auraView.host.clientWidth, h = auraView.host.clientHeight
  for (const v of [auraView, refView]) v.renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}
new ResizeObserver(resize).observe(auraView.host)
resize()

let auraSplat: ReturnType<typeof createLightweightSplat> | null = null
let refViewer: DropInViewer | null = null
let refUrl: string | null = null
let buffer: ArrayBuffer | null = null
let fileName = ''
let bounds = { center: new THREE.Vector3(), radius: 1 }

/** Robust bounds (2nd–98th percentile of centres) so floaters don't skew the framing. */
function splatBounds(data: ArrayBuffer, flip: boolean) {
  const view = new DataView(data)
  const total = Math.floor(data.byteLength / 32)
  const step = Math.max(1, Math.floor(total / 20_000))
  const axes: number[][] = [[], [], []]
  for (let i = 0; i < total; i += step) {
    const x = view.getFloat32(i * 32, true), y = view.getFloat32(i * 32 + 4, true), z = view.getFloat32(i * 32 + 8, true)
    axes[0].push(x); axes[1].push(flip ? -y : y); axes[2].push(flip ? -z : z)
  }
  const lo = new THREE.Vector3(), hi = new THREE.Vector3()
  axes.forEach((v, k) => {
    v.sort((a, b) => a - b)
    lo.setComponent(k, v[Math.floor(v.length * 0.02)])
    hi.setComponent(k, v[Math.floor(v.length * 0.98)])
  })
  return { center: lo.clone().add(hi).multiplyScalar(0.5), radius: Math.max(hi.distanceTo(lo) / 2, 1e-3) }
}

function frameCamera(azimuthDeg = 35, elevationDeg = 20) {
  const az = THREE.MathUtils.degToRad(azimuthDeg), el = THREE.MathUtils.degToRad(elevationDeg)
  const dist = bounds.radius * 2.4
  camera.position.set(
    bounds.center.x + dist * Math.cos(el) * Math.sin(az),
    bounds.center.y + dist * Math.sin(el),
    bounds.center.z + dist * Math.cos(el) * Math.cos(az),
  )
  camera.near = bounds.radius / 200
  camera.far = bounds.radius * 50
  camera.updateProjectionMatrix()
  for (const v of [auraView, refView]) { v.controls.target.copy(bounds.center); v.controls.update() }
}

async function load() {
  if (!buffer) return
  const flip = $<HTMLInputElement>('flip').checked
  const cap = Number($<HTMLSelectElement>('cap').value) || Number.MAX_SAFE_INTEGER
  $<HTMLButtonElement>('run').disabled = true
  setStatus(`Loading ${fileName}…`)

  if (auraSplat) { auraView.scene.remove(auraSplat.object); auraSplat.dispose() }
  auraSplat = createLightweightSplat(buffer, cap)
  if (flip) auraSplat.object.rotation.set(Math.PI, 0, 0)
  auraView.scene.add(auraSplat.object)
  $('aura-info').textContent = `${auraSplat.renderedSplats.toLocaleString()} / ${auraSplat.totalSplats.toLocaleString()} splats`

  if (refViewer) { refView.scene.remove(refViewer); await refViewer.dispose() }
  if (refUrl) URL.revokeObjectURL(refUrl)
  refUrl = URL.createObjectURL(new Blob([buffer]))
  // Shared memory needs cross-origin isolation headers the dev server doesn't send, so the
  // reference sorts with plain worker messages (its documented fallback).
  refViewer = new DropInViewer({ sharedMemoryForWorkers: false, gpuAcceleratedSort: false, integerBasedSort: false, sphericalHarmonicsDegree: 0, antialiased: false })
  refView.scene.add(refViewer)
  await refViewer.addSplatScene(refUrl, {
    format: SceneFormat.Splat, showLoadingUI: false, splatAlphaRemovalThreshold: 1,
    rotation: flip ? [1, 0, 0, 0] : [0, 0, 0, 1], // 180° about X as an [x, y, z, w] quaternion
  })

  bounds = splatBounds(buffer, flip)
  frameCamera()
  $<HTMLButtonElement>('run').disabled = false
  setStatus(`${fileName} loaded. Orbit either view; both share one camera.`)
}

async function loadBuffer(data: ArrayBuffer, name: string) {
  buffer = data; fileName = name
  try { await load() } catch (e) { setStatus(`Could not load: ${e instanceof Error ? e.message : e}`) }
}

$<HTMLInputElement>('file').addEventListener('change', async e => {
  const file = (e.currentTarget as HTMLInputElement).files?.[0]
  if (file) await loadBuffer(await file.arrayBuffer(), file.name)
})
for (const id of ['cap', 'flip']) $(id).addEventListener('change', () => void load())

// Render loop with per-view FPS (averaged over ~1 s). "Mode" renders one side only for fair FPS.
let frames = 0, since = performance.now()
const fpsEls = { aura: $('aura-fps'), ref: $('ref-fps') }
const tick = () => {
  const mode = $<HTMLSelectElement>('mode').value
  if (mode !== 'ref') auraView.renderer.render(auraView.scene, camera)
  if (mode !== 'aura') refView.renderer.render(refView.scene, camera)
  frames++
  const now = performance.now()
  if (now - since > 1000) {
    const fps = `${Math.round(frames * 1000 / (now - since))} fps`
    fpsEls.aura.textContent = mode === 'ref' ? 'paused' : fps
    fpsEls.ref.textContent = mode === 'aura' ? 'paused' : fps
    frames = 0; since = now
  }
  requestAnimationFrame(tick)
}
requestAnimationFrame(tick)

// ---- Image comparison -------------------------------------------------------------------
const wait = (ms: number) => new Promise(r => setTimeout(r, ms))

function readPixels(canvas: HTMLCanvasElement, w: number, h: number) {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(canvas, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h).data
}

function compareImages(a: Uint8ClampedArray, b: Uint8ClampedArray) {
  let se = 0, ae = 0, covA = 0, covB = 0
  const px = a.length / 4
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) { const d = a[i + k] - b[i + k]; se += d * d; ae += Math.abs(d) }
    if (a[i] + a[i + 1] + a[i + 2] > 12) covA++
    if (b[i] + b[i + 1] + b[i + 2] > 12) covB++
  }
  const mse = se / (px * 3)
  return { psnr: mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse), mae: ae / (px * 3), covA: covA / px, covB: covB / px }
}

$('run').addEventListener('click', async () => {
  const run = $<HTMLButtonElement>('run')
  run.disabled = true
  const mode = $<HTMLSelectElement>('mode')
  const prevMode = mode.value
  mode.value = 'both'
  const w = 480, h = 360
  const rows: { view: string; r: ReturnType<typeof compareImages>; a: Uint8ClampedArray; b: Uint8ClampedArray }[] = []
  for (let i = 0; i < 8; i++) {
    const az = i * 45, el = i % 2 ? 35 : 10
    setStatus(`Comparing view ${i + 1} / 8…`)
    frameCamera(az, el)
    await wait(900) // let both renderers finish re-sorting for the new viewpoint
    const a = readPixels(auraView.renderer.domElement, w, h)
    const b = readPixels(refView.renderer.domElement, w, h)
    rows.push({ view: `az ${az}° · el ${el}°`, r: compareImages(a, b), a, b })
  }
  mode.value = prevMode
  frameCamera()

  const fmt = (n: number, d = 1) => Number.isFinite(n) ? n.toFixed(d) : '∞'
  const mean = (f: (r: ReturnType<typeof compareImages>) => number) => rows.reduce((s, x) => s + f(x.r), 0) / rows.length
  $('rows').innerHTML = rows.map(({ view, r }) =>
    `<tr><td>${view}</td><td>${fmt(r.psnr)} dB</td><td>${fmt(r.mae, 2)}</td><td>${fmt(r.covA * 100)}%</td><td>${fmt(r.covB * 100)}%</td></tr>`).join('')
    + `<tr><td><b>Mean</b></td><td><b>${fmt(mean(r => r.psnr))} dB</b></td><td><b>${fmt(mean(r => r.mae), 2)}</b></td><td>${fmt(mean(r => r.covA) * 100)}%</td><td>${fmt(mean(r => r.covB) * 100)}%</td></tr>`

  // Heatmap of the worst view: |difference| ×4, so small colour shifts are visible.
  const worst = rows.reduce((m, x) => (x.r.psnr < m.r.psnr ? x : m))
  const diff = $<HTMLCanvasElement>('diff')
  diff.width = w; diff.height = h
  const img = new ImageData(w, h)
  for (let i = 0; i < worst.a.length; i += 4) {
    for (let k = 0; k < 3; k++) img.data[i + k] = Math.min(255, Math.abs(worst.a[i + k] - worst.b[i + k]) * 4)
    img.data[i + 3] = 255
  }
  diff.getContext('2d')!.putImageData(img, 0, 0)
  $('results').hidden = false
  setStatus(`Comparison done · mean PSNR ${fmt(mean(r => r.psnr))} dB (worst: ${worst.view}).`)
  run.disabled = false
  ;(window as unknown as { __cmp: unknown }).__cmp = rows.map(({ view, r }) => ({ view, ...r }))
})

// Console hook for scripted checks: __frame(azimuthDeg, elevationDeg) sets the shared camera.
;(window as unknown as { __frame: typeof frameCamera }).__frame = frameCamera

const urlParam = new URLSearchParams(location.search).get('url')
if (urlParam) {
  setStatus(`Fetching ${urlParam}…`)
  fetch(urlParam).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer() })
    .then(data => loadBuffer(data, urlParam.split('/').pop() || 'scene.splat'))
    .catch(e => setStatus(`Could not fetch ${urlParam}: ${e instanceof Error ? e.message : e}`))
}
