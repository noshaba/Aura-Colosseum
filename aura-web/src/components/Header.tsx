import { useEffect, useRef, useState } from 'react'
import { shortAddress } from '../solana'
import { useSmartNav } from '../hooks/useSmartNav'
import { CharacterPicker } from './CharacterPicker'
import { CharacterSwitch } from './CharacterSwitch'
import type { Page } from '../types'

type Props = {
  page: Page
  onPage: (page: Page) => void
  onWallet: () => void
  walletAddress: string | null
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

  const go = (id: Page) => { onPage(id); setOpen(false); window.scrollTo({ top: 0 }) }
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
        {/* Always visible; cycles the character (the dropdown picker below shares the store). */}
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
            <CharacterPicker className="fx-nav__pill" tabbable={open} />
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
