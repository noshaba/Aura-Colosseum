import { shortAddress } from '../solana'
import type { Page } from '../types'

type Props = {
  page: Page
  onPage: (page: Page) => void
  onWallet: () => void
  walletAddress: string | null
}

const nav: { id: Page; label: string }[] = [
  { id: 'studio', label: 'Motion studio' },
  { id: 'methodology', label: 'Methodology' },
]

export function Header({ page, onPage, onWallet, walletAddress }: Props) {
  const base = import.meta.env.BASE_URL || '/'

  return (
    <header className="site-header">
      <button className="brand" onClick={() => onPage('studio')} aria-label="Aura Technologies home">
        <img className="brand-logo" src={`${base}brand/aura-logo-horizontal.png`} alt="Aura Technologies" />
      </button>
      <nav className="main-nav" aria-label="Primary navigation">
        {nav.map((item) => (
          <button
            key={item.id}
            className={page === item.id ? 'nav-item active' : 'nav-item'}
            onClick={() => onPage(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <button className={walletAddress ? 'wallet-button connected' : 'wallet-button'} onClick={onWallet}>
        <span className="wallet-fairy" aria-hidden="true">
          <img src={`${base}brand/aura-mark.png`} alt="" />
        </span>
        <span className="wallet-button-label">{walletAddress ? shortAddress(walletAddress) : 'Connect wallet'}</span>
      </button>
    </header>
  )
}
