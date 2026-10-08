# Aura architecture

## Live path

```text
Browser / Aura Web
  ├─ Text2Motion Aura iframe ───────────────► Text2Motion Aura GPU service
  ├─ Motion Library ──────────────► local Aura API
  ├─ Discovery / Constraints ─────► SQLite + NPZ files
  ├─ Downstream Benchmark ────────► deterministic Python experiment
  └─ Curation Market ─────────────► Phantom/Solflare ─► Solana devnet
                                      │
                                      └─ SOL transfer + evidence memo
```

Text2Motion Aura writes robot-native motion files to `aura-motion-library/`. The local API never receives wallet private keys. Wallet signing happens in the browser.

## Judge Mode

```text
static Vite site
   ↓
bundled Text2Motion Aura example previews
   ↓
browser-local pairwise votes
   ↓
browser-local logistic preference model
```

Judge Mode deliberately omits claims about benchmark performance. It exists so the product is clickable without GPU/model weights.

## Solana market design

Current hackathon flow:

1. Server creates a local bounty ID and hashes immutable terms.
2. Requester signs a devnet Memo containing the bounty ID/hash/reward terms.
3. Curator submits a pairwise comparison; Aura creates a deterministic curation evidence hash.
4. Requester accepts work and signs one transaction containing:
   - `SystemProgram.transfer` of SOL to the curator.
   - Memo with bounty ID, curation ID and evidence hash.
5. The browser submits only the transaction signature to the local coordinator.
6. The coordinator calls Solana RPC `getTransaction` and refuses to record the payment unless it verifies:
   - transaction success and confirmation;
   - requester as signer and fee payer;
   - exact curator recipient and lamport amount;
   - exact Aura payout evidence memo;
   - unused transaction signature.
7. Only then is the curation marked paid. Bounty-posting memos are verified the same way before the bounty can accept curations.

This makes payments and auditability dependent on Solana while avoiding unaudited escrow custody. It is not a trustless marketplace yet. Curator wallet ownership and Sybil resistance remain roadmap items.
