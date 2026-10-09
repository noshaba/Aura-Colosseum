import { useEffect, useState } from 'react'

const SCROLLED_AT = 300
const TOGGLE_DELTA = 100

/**
 * Scroll-direction-aware navbar state:
 * - `scrolled` once the page passes 300px (logo shrinks to the mark),
 * - `collapsed` after scrolling down 100px past the last toggle point (menu pill
 *   folds to zero width, logo fades), and re-expands after scrolling up 100px.
 */
export function useSmartNav() {
  const [scrolled, setScrolled] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  useEffect(() => {
    let last = window.scrollY
    let toggleAt = last
    let isCollapsed = false
    const onScroll = () => {
      const y = Math.max(0, window.scrollY)
      const down = y > last
      if (y >= SCROLLED_AT) setScrolled(true)
      if (y === 0) { setScrolled(false); isCollapsed = false; setCollapsed(false); toggleAt = 0 }
      if (down && !isCollapsed && y - toggleAt >= TOGGLE_DELTA) { isCollapsed = true; toggleAt = y; setCollapsed(true) }
      else if (down && isCollapsed) toggleAt = Math.max(toggleAt, y)
      if (!down && isCollapsed && toggleAt - y >= TOGGLE_DELTA) { isCollapsed = false; toggleAt = y; setCollapsed(false) }
      else if (!down && !isCollapsed) toggleAt = Math.min(toggleAt, y)
      last = y
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  return { scrolled, collapsed }
}
