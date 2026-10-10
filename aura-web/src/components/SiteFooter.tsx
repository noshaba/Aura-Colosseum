import type { Page } from '../types'
import { OpenWorldBackdrop } from './OpenWorldBackdrop'

/** `world`: draw the open world behind the footer. The homepage passes false: its closing
 *  section's world already extends under the footer. */
type Props = { onPage: (page: Page) => void; onWallet: () => void; world?: boolean }

export function SiteFooter({ onPage, onWallet, world = true }: Props) {
  const go = (page: Page) => { onPage(page); window.scrollTo({ top: 0 }) }
  return (
    <footer className={world ? 'fx-footer has-world' : 'fx-footer'}>
      {world && <OpenWorldBackdrop className="fx-footer__world" />}
      <div className="fx-footer__content">
        <div className="fx-footer__grid">
          <div className="fx-footer__left">
            <p className="fx-footer__statement">Aura, a human-guided selection layer for generated humanoid motion.</p>
            <p className="fx-footer__muted">Physical AI selection study 001.<br />Evidence on Solana devnet.</p>
          </div>
          <div className="fx-footer__cols">
            <div className="fx-footer__col">
              <span className="fx-footer__label">Explore</span>
              <button className="fx-link" onClick={() => go('studio')}>Motion studio</button>
              <button className="fx-link" onClick={() => go('model')}>Aura model</button>
              <button className="fx-link" onClick={() => go('curation')}>Curation market</button>
              <button className="fx-link" onClick={() => go('methodology')}>Methodology</button>
            </div>
            <div className="fx-footer__col">
              <span className="fx-footer__label">Modes</span>
              <a className="fx-link" href="?judge=0">Live workspace <span aria-hidden="true">→</span></a>
              <a className="fx-link" href="?judge=1">Judge mode <span aria-hidden="true">→</span></a>
            </div>
            <div className="fx-footer__col">
              <span className="fx-footer__label">Network</span>
              <span className="fx-footer__item">Solana devnet</span>
              <button className="fx-link" onClick={onWallet}>Connect wallet</button>
            </div>
            <button className="fx-footer__contact fx-underline" onClick={() => go('methodology')}>Read the methods</button>
          </div>
        </div>
        <div className="fx-footer__legal">
          <span>© {new Date().getFullYear()} Aura Technologies</span>
          <span>Built on NVIDIA Kimodo · Unitree G1</span>
          <span>Diagnostics, not physical certification</span>
        </div>
      </div>
    </footer>
  )
}
