import { useState } from 'react'
import { Header } from './components/Header'
import { MotionStudio } from './components/MotionStudio'
import { Methodology } from './components/Methodology'
import { WalletModal } from './components/WalletModal'
import { CornerOrnaments } from './components/CornerOrnaments'
import { connectWallet, disconnectWallet, getDevnetBalance, requestDevnetSol } from './solana'
import type { WalletConnection, WalletKind } from './solana'
import type { Page } from './types'

export default function App(){
  const [page,setPage]=useState<Page>('studio');const [walletOpen,setWalletOpen]=useState(false);const [wallet,setWallet]=useState<WalletConnection|null>(null);const [balance,setBalance]=useState<number|null>(null);const [busy,setBusy]=useState(false);const [error,setError]=useState('')
  const refresh=async(w=wallet)=>{if(!w){setBalance(null);return}try{setBalance(await getDevnetBalance(w.address))}catch{setBalance(null)}}
  const onConnect=async(kind:WalletKind)=>{setBusy(true);setError('');try{const w=await connectWallet(kind);setWallet(w);await refresh(w)}catch(e){setError(e instanceof Error?e.message:'Unable to connect wallet');throw e}finally{setBusy(false)}}
  const onDisconnect=async()=>{setBusy(true);try{await disconnectWallet(wallet);setWallet(null);setBalance(null)}finally{setBusy(false)}}
  const onAirdrop=async()=>{if(!wallet)return;setBusy(true);setError('');try{await requestDevnetSol(wallet.address);await refresh(wallet)}catch(e){setError(e instanceof Error?e.message:'Devnet airdrop failed');throw e}finally{setBusy(false)}}
  return <div className="app"><CornerOrnaments/><Header page={page} onPage={setPage} onWallet={()=>setWalletOpen(true)} walletAddress={wallet?.address||null}/>{page==='studio'?<MotionStudio/>:<Methodology/>}{walletOpen&&<WalletModal wallet={wallet} balance={balance} busy={busy} error={error} onClose={()=>setWalletOpen(false)} onConnect={onConnect} onDisconnect={onDisconnect} onAirdrop={onAirdrop} onRefresh={()=>refresh()}/>}</div>
}
