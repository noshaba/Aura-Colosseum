import { useState } from 'react'
import { Header } from './components/Header'
import { MotionStudio } from './components/MotionStudio'
import { Methodology } from './components/Methodology'
import { AuraModel } from './components/AuraModel'
import { CurationMarket } from './components/CurationMarket'
import { WalletModal } from './components/WalletModal'
import { SiteFooter } from './components/SiteFooter'
import { LoadingScreen } from './components/LoadingScreen'
import { useScrollReveal } from './hooks/useScrollReveal'
import { connectWallet, disconnectWallet, getDevnetBalance, requestDevnetSol } from './solana'
import type { WalletConnection, WalletKind } from './solana'
import type { Page } from './types'

export default function App(){
  const judgeMode=import.meta.env.VITE_AURA_JUDGE_MODE==='true'
  const [page,setPage]=useState<Page>('studio');const [walletOpen,setWalletOpen]=useState(false);const [wallet,setWallet]=useState<WalletConnection|null>(null);const [balance,setBalance]=useState<number|null>(null);const [busy,setBusy]=useState(false);const [error,setError]=useState('')
  useScrollReveal()
  const refresh=async(w=wallet)=>{if(!w){setBalance(null);return}try{setBalance(await getDevnetBalance(w.address))}catch{setBalance(null)}}
  const onConnect=async(kind:WalletKind)=>{setBusy(true);setError('');try{const w=await connectWallet(kind);setWallet(w);await refresh(w)}catch(e){setError(e instanceof Error?e.message:'Unable to connect wallet');throw e}finally{setBusy(false)}}
  const onDisconnect=async()=>{setBusy(true);try{await disconnectWallet(wallet);setWallet(null);setBalance(null)}finally{setBusy(false)}}
  const onAirdrop=async()=>{if(!wallet)return;setBusy(true);setError('');try{await requestDevnetSol(wallet.address);await refresh(wallet)}catch(e){setError(e instanceof Error?e.message:'Devnet airdrop failed');throw e}finally{setBusy(false)}}
  return <div className="app fx-app"><LoadingScreen/><Header page={page} onPage={setPage} onWallet={()=>setWalletOpen(true)} walletAddress={wallet?.address||null}/>{page==='studio'?<MotionStudio/>:page==='model'?<AuraModel/>:page==='curation'?<main className="fx-page fx-curation-page">{judgeMode?<section className="curation-market" aria-label="Solana curation market"><div className="card-kicker">CURATION MARKET · SOLANA DEVNET</div><h2>Pay for judgment. <em>Keep the evidence.</em></h2><p className="curation-intro">The live workspace lets requesters publish hashed motion-curation tasks, collect same-prompt human comparisons, and pay accepted curators directly in SOL. Aura verifies the bounty terms and payout transaction through Solana RPC before recording the evidence.</p><div className="generated-library-warning">Judge Mode is static and does not connect wallets or the local Aura API. Open the live workspace to create bounties, submit paid curations, and verify devnet transactions.</div></section>:<CurationMarket/>}</main>:<Methodology/>}<SiteFooter world={page!=='studio'} onPage={setPage} onWallet={()=>setWalletOpen(true)}/>{walletOpen&&<WalletModal wallet={wallet} balance={balance} busy={busy} error={error} onClose={()=>setWalletOpen(false)} onConnect={onConnect} onDisconnect={onDisconnect} onAirdrop={onAirdrop} onRefresh={()=>refresh()}/>}</div>
}
