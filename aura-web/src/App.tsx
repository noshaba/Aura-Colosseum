import { useState } from 'react'
import { Header } from './components/Header'
import { MotionStudio } from './components/MotionStudio'
import { AuraModel } from './components/AuraModel'
import { SceneLab } from './components/SceneLab'
import { CurationMarket } from './components/CurationMarket'
import { WalletProvider, useWallet } from './wallet/WalletContext'
import { SiteFooter } from './components/SiteFooter'
import { LoadingScreen } from './components/LoadingScreen'
import { useScrollReveal } from './hooks/useScrollReveal'
import type { Page } from './types'

export default function App(){return <WalletProvider><AppInner/></WalletProvider>}

function AppInner(){
  const judgeMode=import.meta.env.VITE_AURA_JUDGE_MODE==='true'
  const [page,setPage]=useState<Page>('studio');const {wallet,openModal}=useWallet()
  useScrollReveal()
  return <div className="app fx-app"><LoadingScreen/><Header page={page} onPage={setPage} onWallet={openModal} walletAddress={wallet?.address||null}/>{page==='studio'?<MotionStudio/>:page==='scene'?<SceneLab/>:page==='model'?<AuraModel/>:<main className="fx-page fx-curation-page">{judgeMode?<section className="curation-market" aria-label="Solana curation market"><div className="card-kicker">CURATION MARKET · SOLANA DEVNET</div><h2><span>Pay for judgment.</span> <em>Keep the evidence.</em></h2><p className="curation-intro">The live workspace lets requesters publish hashed motion-curation tasks, collect same-prompt human comparisons, and pay accepted curators directly in SOL. Aura verifies the bounty terms and payout transaction through Solana RPC before recording the evidence.</p><div className="generated-library-warning">Judge Mode is static and does not connect wallets or the local Aura API. Open the live workspace to create bounties, submit paid curations, and verify devnet transactions.</div></section>:<CurationMarket/>}</main>}<SiteFooter world={page!=='studio'} onPage={setPage} onWallet={openModal}/></div>
}
