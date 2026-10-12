import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { createLightweightSplat } from '../three/lightweightSplat'
import { prefersReducedMotion } from '../motionPrefs'
import './scene-lab-hero-splat.css'

/** Same asset as Scene Lab's "Demo scene" button: a synthetic Gaussian-splat room (Y-up, floor at y=0). */
const SPLAT_URL = `${import.meta.env.BASE_URL || '/'}demo/aura-scene-demo.splat`
/** Seconds per full ambient drift cycle. */
const DRIFT_SECONDS = 48
const MAX_DPR = 1.5
/** Below this width the backdrop renders a single still frame (no loop). */
const SMALL_SCREEN = 650

/**
 * Decorative Gaussian-splat backdrop behind the Scene Lab hero. Own renderer/context,
 * no controls, pointer-events none. Fetches lazily once the hero is near the viewport,
 * pauses offscreen / on hidden tabs, renders one still frame under reduced motion or on
 * small screens, and silently stays invisible (flat page background) if WebGL or the
 * asset fails.
 */
export function SceneLabHeroSplat() {
  const root = useRef<HTMLDivElement>(null)
  const mount = useRef<HTMLDivElement>(null)
  const [ready, setReady] = useState(false)

  // Stretch the layer from the page top to just past the hero (padding-agnostic, no 100vw).
  useEffect(() => {
    const layer = root.current
    const hero = layer?.parentElement
    const page = hero?.parentElement
    if (!layer || !hero || !page) return
    const fit = () => {
      // Anchor to the page top even if the hero itself later becomes a positioned box.
      const pageTop = page.getBoundingClientRect().top
      const parentTop = (layer.offsetParent ?? page).getBoundingClientRect().top
      layer.style.top = `${Math.round(pageTop - parentTop)}px`
      layer.style.height = `${Math.round(hero.getBoundingClientRect().bottom - pageTop + 40)}px`
    }
    const ro = new ResizeObserver(fit)
    ro.observe(hero); ro.observe(page)
    fit()
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = mount.current
    if (!el) return
    let disposed = false
    let raf = 0
    let renderer: THREE.WebGLRenderer | null = null
    let splat: ReturnType<typeof createLightweightSplat> | null = null
    let visible = false
    let started = false
    const abort = new AbortController()
    const still = prefersReducedMotion() || window.innerWidth < SMALL_SCREEN

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 60)
    const look = new THREE.Vector3()
    const clock = new THREE.Clock(false)
    let t = DRIFT_SECONDS * 0.12

    // Slow drift through the room: a gentle arc + bob, always looking at the back wall.
    const place = (time: number) => {
      const a = (time / DRIFT_SECONDS) * Math.PI * 2
      camera.position.set(Math.sin(a) * 1.1, 1.45 + Math.sin(a * 2) * 0.12, 3.6 + Math.cos(a) * 0.35)
      look.set(Math.sin(a + 0.6) * 0.45, 1.05, -1.2)
      camera.lookAt(look)
    }

    const draw = () => { if (renderer) { place(t); renderer.render(scene, camera) } }

    const resize = () => {
      if (!renderer) return
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight)
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      // Keep the room framed on tall/narrow layouts.
      camera.fov = w / h < 1 ? 64 : 50
      // Wide layouts: push the room toward the right so the copy column stays clear.
      if (w / h > 1.2) camera.setViewOffset(w, h, -Math.round(w * 0.26), 0, w, h)
      else camera.clearViewOffset()
      camera.updateProjectionMatrix()
      if (still || !visible) draw()
    }
    const ro = new ResizeObserver(resize)

    const shouldRun = () => !still && visible && !document.hidden && !!renderer
    const tick = () => {
      if (disposed) return
      if (!shouldRun()) { raf = 0; clock.stop(); return }
      raf = requestAnimationFrame(tick)
      t += Math.min(0.1, clock.getDelta())
      draw()
    }
    const sync = () => {
      if (shouldRun() && !raf) { clock.start(); raf = requestAnimationFrame(tick) }
    }

    const start = async () => {
      started = true
      try {
        const res = await fetch(SPLAT_URL, { signal: abort.signal })
        if (!res.ok) return
        const buffer = await res.arrayBuffer()
        if (disposed) return
        renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'low-power' })
        renderer.outputColorSpace = THREE.SRGBColorSpace
        renderer.setClearColor(0x000000, 0)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR))
        renderer.domElement.setAttribute('aria-hidden', 'true')
        el.appendChild(renderer.domElement)
        splat = createLightweightSplat(buffer, still ? 40_000 : 140_000)
        scene.add(splat.object)
        ro.observe(el)
        resize()
        draw()
        setReady(true)
        sync()
      } catch {
        // Silent fallback: the layer stays transparent over the flat page background.
      }
    }

    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (visible && !started) void start()
      sync()
    }, { rootMargin: '200px 0px' })
    io.observe(el)
    const onVisibility = () => sync()
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      disposed = true
      abort.abort()
      cancelAnimationFrame(raf)
      io.disconnect(); ro.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
      splat?.dispose()
      if (renderer) { renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove() }
    }
  }, [])

  return <div ref={root} className={`scene-lab-hero-splat${ready ? ' is-ready' : ''}`} aria-hidden="true">
    <div ref={mount} className="scene-lab-hero-splat__canvas" />
    <div className="scene-lab-hero-splat__scrim" />
  </div>
}
