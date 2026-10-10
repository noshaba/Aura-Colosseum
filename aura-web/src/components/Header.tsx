import { useEffect, useRef, useState } from 'react'
import { shortAddress } from '../solana'
import { useSmartNav } from '../hooks/useSmartNav'
import { CharacterSwitch } from './CharacterSwitch'
import type { Page } from '../types'
import './nav-panel.css'

type Props = {
  page: Page
  onPage: (page: Page) => void
  onWallet: () => void
  walletAddress: string | null
}

/** Relative luminance (0..1) of a computed rgb()/rgba() colour; null when transparent. */
function luminance(color: string): number | null {
  const m = color.match(/rgba?\(([^)]+)\)/)
  if (!m) return null
  const [r, g, b, a = 1] = m[1].split(/[\s,/]+/).filter(Boolean).map(Number)
  if (a < 0.5) return null
  const lin = (c: number) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4 }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

/** True when the first opaque background under (x, y), outside the nav, is dark. */
function isDarkBehind(x: number, y: number) {
  const hit = document.elementsFromPoint(x, y).find(el => !el.closest('.fx-nav'))
  for (let el: Element | null = hit ?? null; el; el = el.parentElement) {
    const l = luminance(getComputedStyle(el).backgroundColor)
    if (l != null) return l < 0.18
  }
  return false
}

const nav: { id: Page; label: string }[] = [
  { id: 'studio', label: 'Motion studio' },
  { id: 'model', label: 'Aura model' },
  { id: 'curation', label: 'Curation' },
  { id: 'methodology', label: 'Methodology' },
]

export function Header({ page, onPage, onWallet, walletAddress }: Props) {
  const base = import.meta.env.BASE_URL || '/'
  const { scrolled, collapsed } = useSmartNav()
  const [open, setOpen] = useState(false)
  const menu = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    const onDown = (e: PointerEvent) => { if (!menu.current?.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown)
    }
  }, [open])

  // While open, tag each pill with the tone of what sits behind it so it can flip to
  // the dark variant (nav-panel.css). The home hero's split halves are canvas/gradient,
  // which this can't read; HeroArena tags those pills via data-hero-side instead.
  useEffect(() => {
    if (!open) return
    let raf = 0
    const sample = () => {
      raf = 0
      menu.current?.querySelectorAll<HTMLElement>('.fx-nav__pill').forEach(pill => {
        const r = pill.getBoundingClientRect()
        if (isDarkBehind(r.left + r.width / 2, r.top + r.height / 2)) pill.dataset.bg = 'dark'
        else delete pill.dataset.bg
      })
    }
    const schedule = () => { if (!raf) raf = requestAnimationFrame(sample) }
    schedule()
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
    }
  }, [open])

  const go =(id: Page) => { onPage(id); setOpen(false); window.scrollTo({ top: 0 }) }
  const cls = ['fx-nav', scrolled && 'is-scrolled', collapsed && 'is-collapsed', open && 'is-open'].filter(Boolean).join(' ')
  const walletLabel = walletAddress ? shortAddress(walletAddress) : 'Connect wallet'

  return (
    <header className={cls}>
      <div className="fx-nav__inner">
        <button className="fx-nav__logo" onClick={() => go('studio')} aria-label="Aura Technologies home">
          <img src={`${base}brand/aura-logo-horizontal.png`} alt="Aura Technologies" />
          {/* Light copy, shown only over the dark half of the home hero (hero-arena.css). */}
          <img className="fx-nav__logo-alt" src={`${base}brand/aura-logo-horizontal.png`} alt="" aria-hidden="true" />
        </button>
        {/* Always visible; the only character control in the nav. */}
        <CharacterSwitch />
        <div className="fx-nav__side" ref={menu}>
          <button className="fx-nav__burger" aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open} aria-controls="fx-nav-panel" onClick={() => setOpen(v => !v)}>
            <span className="fx-nav__burger-line is-top" />
            <span className="fx-nav__burger-line is-mid" />
            <span className="fx-nav__burger-line is-bottom" />
          </button>
          <nav id="fx-nav-panel" className="fx-nav__panel" aria-label="Primary navigation" aria-hidden={!open}>
            {nav.map((item) => (
              <button key={item.id} tabIndex={open ? 0 : -1} className={page === item.id ? 'fx-nav__link fx-nav__pill is-current' : 'fx-nav__link fx-nav__pill'} onClick={() => go(item.id)}>
                {item.label}
              </button>
            ))}
            <button tabIndex={open ? 0 : -1} className={walletAddress ? 'fx-btn fx-btn--primary fx-nav__pill fx-nav__cta is-connected' : 'fx-btn fx-btn--primary fx-nav__pill fx-nav__cta'} onClick={() => { setOpen(false); onWallet() }}>
              <img className="fx-nav__cta-mark" src={`${base}brand/aura-mark.png`} alt="" aria-hidden="true" />
              <span>{walletLabel}</span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  )
}
