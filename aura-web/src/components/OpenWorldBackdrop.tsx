import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { createAuraRenderer, createAuraWorld, prefersReducedMotion } from '../three/auraWorld'

/** Seconds per full slow orbit of the camera around the meadow. */
const ORBIT_SECONDS = 140

/**
 * The Aura open world as a full-bleed backdrop (no character): a camera drifts slowly
 * around the fairy meadow while content sits on top. Pauses offscreen; still frame
 * under reduced motion.
 */
export function OpenWorldBackdrop({ className = 'fx-cta__world' }: { className?: string }) {
  const mount = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = mount.current
    if (!el) return
    const reduced = prefersReducedMotion()
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 400)
    const renderer = createAuraRenderer({ alpha: false, powerPreference: 'low-power' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75))
    renderer.domElement.setAttribute('aria-hidden', 'true')
    el.appendChild(renderer.domElement)
    const world = createAuraWorld(scene, renderer, { reducedMotion: reduced, ring: false, contactShadow: false, props: { density: 0.8, keepOut: 1.2 } })
    world.observe(el)

    const look = new THREE.Vector3(0, 0.9, 0)
    const place = (t: number) => {
      const a = (t / ORBIT_SECONDS) * Math.PI * 2
      // Low, wide orbit: horizon (sky) fills the top where the heading sits.
      camera.position.set(Math.cos(a) * 7.5, 1.35 + Math.sin(a * 2) * 0.12, Math.sin(a) * 7.5)
      camera.lookAt(look)
    }

    let w = 0, h = 0
    const fit = () => {
      const nw = el.clientWidth, nh = el.clientHeight
      if (!nw || !nh || (nw === w && nh === h)) return
      w = nw; h = nh
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      if (reduced) { place(ORBIT_SECONDS * 0.15); world.render(camera, 0) }
    }
    const resize = new ResizeObserver(fit)
    resize.observe(el)

    // When this section closes the page, the world also runs behind the footer card
    // (its cream frame and rounded corners show meadow instead of page background).
    const section = el.parentElement
    const page = section?.parentElement
    const footer = page?.nextElementSibling
    const underFooter = section === page?.lastElementChild && footer?.classList.contains('fx-footer') ? footer as HTMLElement : null
    const extend = () => { if (underFooter) el.style.bottom = `-${underFooter.offsetHeight}px` }
    const footerResize = new ResizeObserver(extend)
    if (underFooter) footerResize.observe(underFooter)
    extend()
    fit()

    const clock = new THREE.Clock()
    let elapsed = ORBIT_SECONDS * 0.15
    let frame = 0
    let disposed = false
    const tick = () => {
      if (disposed) return
      frame = requestAnimationFrame(tick)
      const dt = Math.min(clock.getDelta(), 0.05)
      if (!world.visible) return
      elapsed += dt
      place(elapsed)
      world.render(camera, dt)
    }
    if (reduced) { place(elapsed); world.render(camera, 0) } else tick()

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      footerResize.disconnect()
      el.style.bottom = ''
      world.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
    }
  }, [])
  return <div className={className} ref={mount} />
}
