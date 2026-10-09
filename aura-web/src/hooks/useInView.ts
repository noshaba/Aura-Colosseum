import { useEffect, useRef, useState } from 'react'

type Callback = (isIntersecting: boolean) => void

/** One IntersectionObserver per threshold, shared by every element that asks for it. */
const observers = new Map<number, { io: IntersectionObserver; callbacks: Map<Element, Callback> }>()

function observe(el: Element, threshold: number, cb: Callback) {
  let shared = observers.get(threshold)
  if (!shared) {
    const callbacks = new Map<Element, Callback>()
    const io = new IntersectionObserver(entries => entries.forEach(e => callbacks.get(e.target)?.(e.isIntersecting)), { threshold })
    shared = { io, callbacks }
    observers.set(threshold, shared)
  }
  shared.callbacks.set(el, cb)
  shared.io.observe(el)
  return () => {
    shared.callbacks.delete(el)
    shared.io.unobserve(el)
    if (shared.callbacks.size === 0) { shared.io.disconnect(); observers.delete(threshold) }
  }
}

/**
 * Tracks whether an element is at least `threshold` visible. Used to play/pause
 * looping micro-animations only while their section is on screen (mirrors the
 * reference's IntersectionObserver play/pause at a 0.3 threshold). Observers are
 * shared per threshold, so 40+ icons cost one observer.
 */
export function useInView<T extends Element>(threshold = 0.3) {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') { setInView(true); return }
    return observe(el, threshold, setInView)
  }, [threshold])
  return [ref, inView] as const
}
