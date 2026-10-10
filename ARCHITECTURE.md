# Aura architecture

## Live path

```text
Browser / Aura Web
  ├─ Prompt + batch request ───────────────► local Aura API
  │                                          │
  │                                          └─ NVIDIA Kimodo G1 generator (GPU)
  │                                               ├─ fresh seed per batch
  │                                               └─ hash-deduplicated NPZ candidates
  ├─ Human A/B preferences ───────────────► SQLite + immutable motion hashes
  │                                          ├─ per-evaluator pair deduplication
  │                                          └─ training-data diagnostics
  ├─ Aura Reward Model ───────────────────► temporal Transformer reward + rankings
  ├─ AMP-inspired prior ──────────────────► trajectory prior learned from valid preferences
  ├─ Discovery / Constraints ─────────────► baseline selector + explicit G1 checks
  ├─ Downstream Benchmark ────────────────► deterministic Python experiment
  └─ Curation Market ─────────────────────► Phantom/Solflare ─► Solana devnet
                                             │
                                             └─ SOL transfer + evidence memo
```

NVIDIA Kimodo is the upstream motion generator. The internal `text2motion-aura/` folder/package name is retained only as a compatibility namespace from an earlier integration rename; it is not an Aura-authored generation model.

Generated robot-native motion files are written to `aura-motion-library/`. Default live generation uses a fresh random batch seed and rejects duplicate NPZ hashes before a candidate is admitted to the library. Passing an explicit seed remains available for reproducible experiments.

Preference rows store the motion hashes seen at voting time. The training pipeline rechecks those hashes, rejects changed/identical motions, and de-duplicates repeated votes by evaluator + content pair before either learned model trains.

The local API never receives wallet private keys. Wallet signing happens in the browser.

## Judge Mode

```text
static Vite site
   ↓
bundled NVIDIA Kimodo example previews
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
