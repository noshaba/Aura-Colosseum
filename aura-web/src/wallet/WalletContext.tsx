import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { WalletModal } from '../components/WalletModal'
import { connectWallet, detectedWallets, disconnectWallet, getDevnetBalance, requestDevnetSol } from '../solana'
import type { WalletConnection, WalletKind } from '../solana'

/** Single source of truth for the browser wallet: header pill, modal, Curation and every lab share it. */
export type WalletApi = {
  wallet: WalletConnection | null
  balance: number | null
  busy: boolean
  error: string
  connect: (kind: WalletKind) => Promise<WalletConnection>
  /** Returns the connected wallet, or connects the user's preferred (last chosen) installed wallet. */
  ensureWallet: () => Promise<WalletConnection>
  disconnect: () => Promise<void>
  airdrop: () => Promise<void>
  refresh: () => Promise<void>
  openModal: () => void
}

const Ctx = createContext<WalletApi | null>(null)
const KIND_KEY = 'aura.wallet.kind'
const readKind = (): WalletKind | null => { try { const v = localStorage.getItem(KIND_KEY); return v === 'phantom' || v === 'solflare' ? v : null } catch { return null } }
const writeKind = (k: WalletKind | null) => { try { k ? localStorage.setItem(KIND_KEY, k) : localStorage.removeItem(KIND_KEY) } catch { /* private mode */ } }
const label = (k: WalletKind) => (k === 'phantom' ? 'Phantom' : 'Solflare')

export function useWallet(): WalletApi {
  const v = useContext(Ctx)
  if (!v) throw new Error('useWallet must be used inside <WalletProvider>')
  return v
}

export function WalletProvider({ children }: { children: ReactNode }) {
  const [wallet, setWallet] = useState<WalletConnection | null>(null)
  const [balance, setBalance] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)
  const walletRef = useRef<WalletConnection | null>(null)
  walletRef.current = wallet

  const refreshFor = useCallback(async (w: WalletConnection | null) => {
    if (!w) { setBalance(null); return }
    try { const b = await getDevnetBalance(w.address); if (walletRef.current?.address === w.address || walletRef.current === null) setBalance(b) } catch { setBalance(null) }
  }, [])
  const refresh = useCallback(() => refreshFor(walletRef.current), [refreshFor])

  const connect = useCallback(async (kind: WalletKind) => {
    setBusy(true); setError('')
    try {
      const w = await connectWallet(kind)
      writeKind(kind); walletRef.current = w; setWallet(w); void refreshFor(w)
      return w
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to connect wallet'); throw e
    } finally { setBusy(false) }
  }, [refreshFor])

  const ensureWallet = useCallback(async () => {
    if (walletRef.current) return walletRef.current
    const installed = detectedWallets().filter(x => x.installed)
    const pref = readKind()
    const pick = installed.find(x => x.kind === pref) || installed[0]
    if (!pick) { setOpen(true); throw new Error('Install Phantom or Solflare (see the wallet dialog), then try again.') }
    return connect(pick.kind)
  }, [connect])

  const disconnect = useCallback(async () => {
    setBusy(true)
    try { await disconnectWallet(walletRef.current) } catch { /* wallet may already be disconnected */ }
    writeKind(null); walletRef.current = null; setWallet(null); setBalance(null); setBusy(false)
  }, [])

  const airdrop = useCallback(async () => {
    const w = walletRef.current; if (!w) return
    setBusy(true); setError('')
    try { await requestDevnetSol(w.address); await refreshFor(w) }
    catch (e) { setError(e instanceof Error ? e.message : 'Devnet airdrop failed'); throw e }
    finally { setBusy(false) }
  }, [refreshFor])

  // Silent session restore: only succeeds if the user already trusted this site.
  useEffect(() => {
    let cancelled = false
    const attempt = async () => {
      if (walletRef.current) return
      const pref = readKind()
      const order = detectedWallets().filter(x => x.installed).sort((a, b) => (a.kind === pref ? -1 : b.kind === pref ? 1 : 0))
      for (const c of order) {
        try {
          const provider = c.kind === 'phantom' ? (window.phantom?.solana || window.solana) : window.solflare
          if (!provider) continue
          const res = await provider.connect({ onlyIfTrusted: true })
          const pk = (res && 'publicKey' in res && res.publicKey) || provider.publicKey
          if (pk && !cancelled && !walletRef.current) {
            const w: WalletConnection = { kind: c.kind, label: label(c.kind), address: pk.toString(), provider }
            walletRef.current = w; setWallet(w); void refreshFor(w); return
          }
        } catch { /* not trusted yet: stay disconnected */ }
      }
    }
    void attempt()
    const t = window.setTimeout(() => void attempt(), 1200) // extensions can inject after load
    return () => { cancelled = true; window.clearTimeout(t) }
  }, [refreshFor])

  // Follow account switches / disconnects made inside the wallet extension.
  useEffect(() => {
    const p = wallet?.provider
    if (!wallet || !p?.on) return
    const onAccount = (pk: { toString(): string } | null) => {
      if (!pk) { writeKind(null); walletRef.current = null; setWallet(null); setBalance(null); return }
      const next = { ...wallet, address: pk.toString() }
      walletRef.current = next; setWallet(next); void refreshFor(next)
    }
    const onDisc = () => { walletRef.current = null; setWallet(null); setBalance(null) }
    p.on('accountChanged', onAccount); p.on('disconnect', onDisc)
    return () => { const off = p.off || p.removeListener; off?.call(p, 'accountChanged', onAccount); off?.call(p, 'disconnect', onDisc) }
  }, [wallet?.provider, wallet?.address]) // eslint-disable-line react-hooks/exhaustive-deps

  const api = useMemo<WalletApi>(() => ({ wallet, balance, busy, error, connect, ensureWallet, disconnect, airdrop, refresh, openModal: () => setOpen(true) }),
    [wallet, balance, busy, error, connect, ensureWallet, disconnect, airdrop, refresh])

  return (
    <Ctx.Provider value={api}>
      {children}
      {open && <WalletModal wallet={wallet} balance={balance} busy={busy} error={error} onClose={() => setOpen(false)}
        onConnect={async k => { await connect(k) }} onDisconnect={disconnect} onAirdrop={airdrop} onRefresh={refresh} />}
    </Ctx.Provider>
  )
}
