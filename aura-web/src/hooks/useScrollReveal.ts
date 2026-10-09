import { useEffect } from 'react'
import { prefersReducedMotion } from '../motionPrefs'

/** Major blocks that fade in and rise as they enter the viewport. */
export const REVEAL_SELECTOR = [
  '.fx-section__title',
  '.fx-learning',
  '.fx-flow',
  '.fx-about',
  '.fx-cta',
  '.fx-cases',
  '.method-grid',
  '.fx-footer__content',
].join(',')

/** Never reveal-animate these or anything inside them (interactive / generator UI). */
const EXCLUDE_SELECTOR = '.aura-generator-stage, .fx-hero__asset, .fx-pair, .fx-hero'

/**
 * Scroll reveal, reveal-once. Elements stay visible by default: the hidden state
 * only applies after this hook adds `fx-reveal` on mount, and only to elements
 * that start below the fold. Skipped entirely under prefers-reduced-motion or
 * without IntersectionObserver. Re-scans when matching nodes are added later
 * (page switch, async content).
 */
export function useScrollReveal(selector = REVEAL_SELECTOR) {
  useEffect(() => {
    if (typeof window === 'undefined' || typeof IntersectionObserver === 'undefined') return
    if (prefersReducedMotion()) return

    const seen = new WeakSet<Element>()
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (!e.isIntersecting) continue
        e.target.classList.add('is-revealed')
        io.unobserve(e.target)
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' })

    const scan = () => {
      document.querySelectorAll(selector).forEach(el => {
        if (seen.has(el)) return
        seen.add(el)
        if (el.closest(EXCLUDE_SELECTOR)) return
        // Already on screen at mount: leave it alone (no flash, no motion above the fold).
        if (el.getBoundingClientRect().top < window.innerHeight) return
        el.classList.add('fx-reveal')
        io.observe(el)
      })
    }
    scan()

    let raf = 0
    const mo = new MutationObserver(() => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(scan)
    })
    mo.observe(document.body, { childList: true, subtree: true })

    return () => {
      cancelAnimationFrame(raf)
      mo.disconnect()
      io.disconnect()
    }
  }, [selector])
}
