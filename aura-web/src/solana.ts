import { Buffer } from 'buffer'
import { Connection, LAMPORTS_PER_SOL, PublicKey, SystemProgram, Transaction, TransactionInstruction } from '@solana/web3.js'
export type WalletKind = 'phantom' | 'solflare'
export type SolanaWalletProvider = {
  publicKey?: { toString(): string }
  isConnected?: boolean
  connect(options?: { onlyIfTrusted?: boolean }): Promise<{ publicKey?: { toString(): string } } | void>
  disconnect?(): Promise<void>
  signTransaction?(transaction: any): Promise<any>
  signAndSendTransaction?(transaction: any, options?: Record<string, unknown>): Promise<{ signature?: string } | string>
}
export type WalletConnection = { kind: WalletKind; label: string; address: string; provider: SolanaWalletProvider }

declare global {
  interface Window {
    phantom?: { solana?: SolanaWalletProvider & { isPhantom?: boolean } }
    solana?: SolanaWalletProvider & { isPhantom?: boolean }
    solflare?: SolanaWalletProvider & { isSolflare?: boolean }
  }
}

export const DEVNET_RPC = import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com'
export const MEMO_PROGRAM_ID = 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr'
function providerFor(kind:WalletKind):SolanaWalletProvider|null{if(kind==='phantom')return window.phantom?.solana||(window.solana?.isPhantom?window.solana:null)||null;return window.solflare||null}
export function detectedWallets(){return[{kind:'phantom' as const,label:'Phantom',installed:Boolean(providerFor('phantom'))},{kind:'solflare' as const,label:'Solflare',installed:Boolean(providerFor('solflare'))}]}
export async function connectWallet(kind:WalletKind):Promise<WalletConnection>{const provider=providerFor(kind);if(!provider)throw new Error(`${kind==='phantom'?'Phantom':'Solflare'} was not detected in this browser.`);const response=await provider.connect();const publicKey=response&&'publicKey' in response&&response.publicKey?response.publicKey:provider.publicKey;if(!publicKey)throw new Error('The wallet connected but did not return a public key.');return{kind,label:kind==='phantom'?'Phantom':'Solflare',address:publicKey.toString(),provider}}
export async function disconnectWallet(wallet:WalletConnection|null){await wallet?.provider.disconnect?.()}
function connection(){return new Connection(DEVNET_RPC,'confirmed')}
export async function getDevnetBalance(address:string){return await connection().getBalance(new PublicKey(address),'confirmed')/LAMPORTS_PER_SOL}
export async function requestDevnetSol(address:string){const rpc=connection();const signature=await rpc.requestAirdrop(new PublicKey(address),Math.round(.05*LAMPORTS_PER_SOL));const latest=await rpc.getLatestBlockhash('confirmed');await rpc.confirmTransaction({signature,...latest},'confirmed');return signature}
export function explorerTransactionUrl(signature:string){return `https://explorer.solana.com/tx/${signature}?cluster=devnet`}
export function explorerAddressUrl(address:string){return `https://explorer.solana.com/address/${address}?cluster=devnet`}
export function shortAddress(address:string){return address.length>12?`${address.slice(0,5)}…${address.slice(-4)}`:address}

async function memoTransaction(wallet:WalletConnection,payload:unknown){
  const rpc=connection();const signer=new PublicKey(wallet.address)
  const bytes=Buffer.from(JSON.stringify(payload),'utf8');if(bytes.byteLength>850)throw new Error('On-chain receipt is too large for one memo transaction.')
  const ix=new TransactionInstruction({programId:new PublicKey(MEMO_PROGRAM_ID),keys:[{pubkey:signer,isSigner:true,isWritable:false}],data:bytes})
  const latest=await rpc.getLatestBlockhash('confirmed');if(!wallet.provider.signTransaction)throw new Error('Wallet must support signTransaction for explicit devnet submission.')
  const tx=new Transaction({feePayer:signer,recentBlockhash:latest.blockhash}).add(ix);const signed=await wallet.provider.signTransaction(tx);const signature=await rpc.sendRawTransaction(signed.serialize(),{skipPreflight:false});await rpc.confirmTransaction({signature,...latest},'confirmed');return signature
}

export type G1ScreenReceipt={signature:string;walletAddress:string;motionId:string;nativeSha256:string;reportSha256:string;method:string;createdAt:number;cluster:'devnet'}
export async function commitG1ScreenOnChain(wallet:WalletConnection,motionId:string,nativeSha256:string,reportSha256:string,method:string):Promise<G1ScreenReceipt>{if(![nativeSha256,reportSha256].every(s=>/^[0-9a-f]{64}$/.test(s)))throw new Error('Invalid motion or report SHA-256 digest');const signature=await memoTransaction(wallet,{app:'AURA',v:1,kind:'g1-kinematic-screen',motion:motionId,dataSha256:nativeSha256,reportSha256,method});return{signature,walletAddress:wallet.address,motionId,nativeSha256,reportSha256,method,cluster:'devnet',createdAt:Date.now()}}

export type PreferenceChainReceipt={signature:string;walletAddress:string;preferenceId:number;evidenceSha256:string;createdAt:number;cluster:'devnet'}
export async function commitPreferenceOnChain(wallet:WalletConnection,preferenceId:number,evidenceSha256:string):Promise<PreferenceChainReceipt>{if(!Number.isSafeInteger(preferenceId)||!/^[0-9a-f]{64}$/.test(evidenceSha256))throw new Error('Invalid preference evidence');const signature=await memoTransaction(wallet,{app:'AURA',v:1,kind:'g1-human-preference',preferenceId,evidenceSha256});return{signature,walletAddress:wallet.address,preferenceId,evidenceSha256,createdAt:Date.now(),cluster:'devnet'}}

export type DownstreamBenchmarkReceipt={signature:string;walletAddress:string;reportSha256:string;benchmarkVersion:string;createdAt:number;cluster:'devnet'}
export async function commitDownstreamBenchmarkOnChain(wallet:WalletConnection,reportSha256:string,benchmarkVersion:string):Promise<DownstreamBenchmarkReceipt>{if(!/^[0-9a-f]{64}$/.test(reportSha256))throw new Error('Invalid benchmark report SHA-256');const signature=await memoTransaction(wallet,{app:'AURA',v:1,kind:'g1-downstream-imitation-benchmark',reportSha256,method:benchmarkVersion});return{signature,walletAddress:wallet.address,reportSha256,benchmarkVersion,createdAt:Date.now(),cluster:'devnet'}}

export type ConstraintReportReceipt={signature:string;walletAddress:string;reportSha256:string;method:string;createdAt:number;cluster:'devnet'}
export async function commitConstraintReportOnChain(wallet:WalletConnection,reportSha256:string,method:string):Promise<ConstraintReportReceipt>{if(!/^[0-9a-f]{64}$/.test(reportSha256))throw new Error('Invalid constraint report SHA-256');if(!method||method.length>80)throw new Error('Invalid constraint method');const signature=await memoTransaction(wallet,{app:'AURA',v:1,kind:'g1-constraint-selection',reportSha256,method});return{signature,walletAddress:wallet.address,reportSha256,method,createdAt:Date.now(),cluster:'devnet'}}

export type BountyPostReceipt={signature:string;walletAddress:string;bountyId:string;bountySha256:string;rewardLamports:number;targetCurations:number;createdAt:number;cluster:'devnet'}
/** Publishes terms on devnet. Funds are not escrowed in this hackathon version. */
export async function commitCurationBountyOnChain(wallet:WalletConnection,bountyId:string,bountySha256:string,rewardLamports:number,targetCurations:number):Promise<BountyPostReceipt>{if(!/^bounty_[a-f0-9]{16}$/.test(bountyId)||!/^[0-9a-f]{64}$/.test(bountySha256))throw new Error('Invalid bounty evidence');if(!Number.isSafeInteger(rewardLamports)||rewardLamports<10_000)throw new Error('Invalid reward amount');const signature=await memoTransaction(wallet,{app:'AURA',v:1,kind:'curation-bounty',bountyId,bountySha256,rewardLamports,targetCurations});return{signature,walletAddress:wallet.address,bountyId,bountySha256,rewardLamports,targetCurations,createdAt:Date.now(),cluster:'devnet'}}

export type CurationPaymentReceipt={signature:string;payer:string;curator:string;curationId:number;bountyId:string;evidenceSha256:string;paidLamports:number;createdAt:number;cluster:'devnet'}
/** Direct SOL payout + evidence hash in the same transaction. Requester approval is required; this is not escrow. */
export async function payCurationOnChain(wallet:WalletConnection,curatorAddress:string,paidLamports:number,curationId:number,bountyId:string,evidenceSha256:string):Promise<CurationPaymentReceipt>{
  if(!Number.isSafeInteger(paidLamports)||paidLamports<10_000)throw new Error('Invalid payout amount');if(!Number.isSafeInteger(curationId)||curationId<1)throw new Error('Invalid curation ID');if(!/^bounty_[a-f0-9]{16}$/.test(bountyId)||!/^[0-9a-f]{64}$/.test(evidenceSha256))throw new Error('Invalid curation evidence')
  const rpc=connection();const payer=new PublicKey(wallet.address),curator=new PublicKey(curatorAddress)
  const memo=Buffer.from(JSON.stringify({app:'AURA',v:1,kind:'curation-payout',bountyId,curationId,evidenceSha256,paidLamports}),'utf8');if(memo.byteLength>850)throw new Error('Payout receipt is too large')
  const transferIx=SystemProgram.transfer({fromPubkey:payer,toPubkey:curator,lamports:paidLamports});const memoIx=new TransactionInstruction({programId:new PublicKey(MEMO_PROGRAM_ID),keys:[{pubkey:payer,isSigner:true,isWritable:false}],data:memo})
  const latest=await rpc.getLatestBlockhash('confirmed');if(!wallet.provider.signTransaction)throw new Error('Wallet must support signTransaction for explicit devnet submission');const tx=new Transaction({feePayer:payer,recentBlockhash:latest.blockhash}).add(transferIx,memoIx);const signed=await wallet.provider.signTransaction(tx);const signature=await rpc.sendRawTransaction(signed.serialize(),{skipPreflight:false});await rpc.confirmTransaction({signature,...latest},'confirmed');return{signature,payer:wallet.address,curator:curatorAddress,curationId,bountyId,evidenceSha256,paidLamports,createdAt:Date.now(),cluster:'devnet'}
}
