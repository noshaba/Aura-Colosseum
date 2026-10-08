import { useMemo, useState } from 'react'
import { detectedWallets, explorerAddressUrl, shortAddress } from '../solana'
import type { WalletConnection, WalletKind } from '../solana'

type Props = {
  wallet: WalletConnection | null
  balance: number | null
  busy: boolean
  error: string
  onConnect: (kind: WalletKind) => Promise<void>
  onDisconnect: () => Promise<void>
  onAirdrop: () => Promise<void>
  onRefresh: () => Promise<void>
  onClose: () => void
}

export function WalletModal({ wallet, balance, busy, error, onConnect, onDisconnect, onAirdrop, onRefresh, onClose }: Props) {
  const wallets = useMemo(detectedWallets, [])
  const [localMessage, setLocalMessage] = useState('')

  const run = async (action: () => Promise<void>, success = '') => {
    setLocalMessage('')
    try {
      await action()
      if (success) setLocalMessage(success)
    } catch {
      // Parent owns the detailed error string.
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="wallet-modal" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="wallet-title">
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="modal-kicker">SOLANA DEVNET · REAL WALLET</div>
        <h2 id="wallet-title">{wallet ? 'Wallet connected.' : 'Connect your contribution.'}</h2>

        {wallet ? (
          <>
            <p>Your browser wallet signs contribution receipts. Aura submits the signed transaction to Solana devnet; no private key is stored by the site.</p>
            <div className="wallet-connected-card">
              <span>{wallet.label}</span>
              <strong>{shortAddress(wallet.address)}</strong>
              <small>{balance == null ? 'Devnet balance unavailable' : `${balance.toFixed(4)} devSOL`}</small>
            </div>
            <div className="wallet-actions-row">
              <button className="connect-browser" disabled={busy} onClick={() => run(onAirdrop, 'Requested 0.05 devSOL from the public devnet faucet.')}>{busy ? 'Working…' : 'Get devnet SOL'}</button>
              <button className="outline-action" disabled={busy} onClick={() => run(onRefresh)}>Refresh</button>
            </div>
            <a className="wallet-explorer-link" href={explorerAddressUrl(wallet.address)} target="_blank" rel="noreferrer">View wallet on Solana Explorer ↗</a>
            <button className="wallet-disconnect" disabled={busy} onClick={() => run(onDisconnect)}>Disconnect wallet</button>
          </>
        ) : (
          <>
            <p>Connect a browser wallet to sign an on-chain receipt for your human motion evaluations. Transactions are sent to Solana devnet only.</p>
            <div className="wallet-provider-list">
              {wallets.map((candidate) => (
                <button
                  key={candidate.kind}
                  className="connect-browser wallet-provider"
                  disabled={busy || !candidate.installed}
                  onClick={() => run(() => onConnect(candidate.kind), `${candidate.label} connected.`)}
                >
                  <span>{candidate.label}</span>
                  <small>{candidate.installed ? 'Detected' : 'Not installed'}</small>
                </button>
              ))}
            </div>
            <div className="wallet-install-note">Phantom or Solflare browser extension required for signing. A pasted address cannot sign an on-chain contribution.</div>
          </>
        )}

        {(error || localMessage) && <div className={error ? 'wallet-message error' : 'wallet-message'}>{error || localMessage}</div>}
        <small>Devnet SOL has no monetary value. The site never asks for a seed phrase or private key.</small>
      </div>
    </div>
  )
}
