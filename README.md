# Aura — Human-guided discovery for robot motion data

Aura is a hackathon prototype for selecting useful synthetic humanoid motion data. NVIDIA Kimodo generates Unitree G1 candidates; Aura adds explicit kinematic constraints, human pairwise preferences, a neural Aura Motion Reward Model, an interpretable selector baseline, an equal-budget downstream benchmark, and Solana-backed curation payments/provenance.

**Upstream generation attribution:** motion generation is provided by NVIDIA Kimodo. The `text2motion-aura/` name is retained only as an internal compatibility namespace from an earlier integration rename; it is not presented as an Aura-authored generation model.

## Core claim

Aura does **not** claim that a kinematic motion is physically safe or executable. The experiment it is designed to test is narrower:

> Given a same-prompt cohort of generated G1 motions, can human preference learning help select demonstrations that perform better than random selection on an unseen offline imitation benchmark?

The code preserves negative results and hashes the evidence needed to reproduce each stage.

## Product flow

```text
NVIDIA Kimodo generation
      ↓
robot-native NPZ library
      ↓
explicit kinematic constraints
      ↓
human A/B comparisons
      ↓
Aura Motion Reward Model
(temporal Transformer + pairwise reward loss)
      ↓
ranked / selected demonstrations
      ↓
Aura vs equal-budget random benchmark
      ↓
report + dataset hashes
      ↓
Solana provenance + direct curation payouts
```

## Installation on a new machine

See **`INSTALL.md`** for the complete Linux/NVIDIA setup guide. A bootstrap helper is also provided:

```bash
./scripts/install_new_machine.sh
```

Review the CUDA/PyTorch compatibility notes in `INSTALL.md` before using the helper on a machine with a different NVIDIA driver.

## Two ways to run Aura

### 1. Judge Mode — no GPU or Python server

Judge Mode is a static interactive demo using **NVIDIA Kimodo's bundled G1 examples**. It lets a judge play motions, make pairwise choices, and train a small preference model locally in the browser.

These examples are from different tasks and are **not experimental evidence**.

```bash
cd aura-web
npm install
npm run judge
```

Open the Vite URL. You can also append `?judge=1` to any build.

### 2. Live Mode — real generation + experiment + curation market

Prerequisites: a working NVIDIA Kimodo environment, NVIDIA GPU/CUDA for generation, Node/npm, and optionally Phantom or Solflare for devnet transactions.

```bash
conda activate <your-aura-env>
export PYTHONNOUSERSITE=1
export AURA_G1_FINISH=all-gold

cd text2motion-aura
python -m pip install -e .
```

Aura uses NVIDIA Kimodo directly as the upstream generation library; the standalone Kimodo/Viser viewer is not required. Use the helper launcher:

```bash
./scripts/run_live.sh
```

Manual services are:

```text
text2motion_aura_textencoder                         # CPU text encoder
python aura_library_server.py             # Aura API + persistent/lazy G1 generator on GPU
cd aura-web && npm run dev                # browser UI
```

`aura_library_server.py` loads the NVIDIA Kimodo G1 motion model on the first Generate request and keeps it in memory for later generations. Set `AURA_TEXT2MOTION_STEPS` (default `30`) to change the streamlined generator's denoising-step count.

By default both browser and server use `https://api.devnet.solana.com`. If you use a custom devnet RPC, set both sides consistently before starting:

```bash
export AURA_SOLANA_RPC_URL=https://your-devnet-rpc.example
export VITE_SOLANA_RPC_URL=https://your-devnet-rpc.example
```

The browser submits signatures only; the Python coordinator independently fetches the confirmed transaction from its configured RPC before recording bounty publication or payment.

## What is actually implemented

- A single prompt + candidate-count control embedded directly in the top Motion-vs-Motion arena, calling NVIDIA Kimodo without embedding the full Kimodo/Viser viewer.
- Automatic saving of NVIDIA Kimodo G1 generations as NPZ plus rotation-aware G1 preview, with newest output auto-playing in the main viewer.
- Preloaded G1 starter/reference comparisons inside the top Motion-vs-Motion arena so first-time visitors can rate immediately without GPU generation. In live mode, those exact trajectories are materialized as robot-native NPZ records and their A/B choices contribute to Aura's broad motion-preference reward model; they remain tagged separately from same-prompt generated comparisons and are not prompt-compliance evidence.
- Reproducible G1 kinematic screening with report/file SHA-256 hashes.
- Native task-constraint engine; no BioIK or AI4Animation dependency.
- Persistent pairwise human comparisons in SQLite.
- **Aura Motion Reward Model:** compact temporal Transformer trained directly from same-prompt human A/B preferences with Bradley–Terry reward loss.
- Regularized pairwise logistic preference model retained as an interpretable baseline with grouped holdout handling.
- Active-pair suggestion and learned candidate ranking.
- Selector-unseen downstream benchmark against repeated equal-budget random baselines.
- Solana devnet memo attestations for preferences, screens, constraints, and benchmark reports.
- **Curation market:** requester publishes hashed bounty terms; curators submit comparisons; accepted curations are paid in SOL; the transfer and curation evidence hash are recorded in the same transaction. The local coordinator independently verifies the posted terms and every claimed payout against Solana RPC before recording them.
- Static Judge Mode requiring neither CUDA nor the Python API.


## Aura Motion Reward Model and RLHF

`aura_reward_model.py` is Aura's primary learned model. It consumes fixed-length G1 trajectory sequences containing root-relative joint positions, joint velocities, root velocity/height, and global joint rotations when available. A compact temporal Transformer produces one scalar reward per motion. Human A/B labels train it with the standard pairwise reward-model objective:

```text
P(A preferred to B) = sigmoid(r(A) - r(B))
```

This is accurately described as **human-preference reward modeling**, which is a core component of RLHF. The current prototype is **not full RLHF** because the NVIDIA Kimodo generator and robot policy are not updated with reinforcement learning against this reward. Full RLHF would add a trainable policy/generator plus an RL stage (for example in a physics simulator for a G1 control policy).

The reward model runs CPU-first so it does not compete with the G1 generator for GPU memory. It becomes trainable after at least six valid unique generated-motion comparisons covering at least four distinct motion hashes. The UI hides learned scores while a person is voting and reveals the model ranking only after the human decision, reducing label bias.

## Curation market scope

The hackathon market is intentionally non-custodial. Aura **does not escrow funds**. A requester approves each direct payout. Before marking a bounty published or a curation paid, the server fetches the devnet transaction and checks success, requester signer/fee payer, exact recipient and lamports where applicable, and the expected Aura memo. Transaction signatures cannot be reused across Aura records. This makes Solana the actual payment rail and audit trail while avoiding an unaudited custody contract.

A production version would add an audited Solana program for escrow, claim rules, disputes, curator reputation, and Sybil resistance.

## The real experiment you still need to run

1. Pick one exact G1 prompt.
2. Generate 12–20 candidates with that exact prompt.
3. Collect genuine pairwise comparisons from more than one person if possible.
4. Train/freeze the Aura selector.
5. Generate several **fresh same-prompt candidates after training** and do not vote on them.
6. Run the downstream benchmark. Those fresh motions must remain selector-unseen.
7. Export the report and report whatever the numbers say, including a loss to random selection.
8. Optionally anchor the final report hash on Solana devnet.

See `EXPERIMENT_PROTOCOL.md` for the submission-grade protocol.

## Tests

```bash
./scripts/test.sh
```

The tests cover preference persistence/training, native constraints, curation-bounty bookkeeping, raw Solana transaction parsing/verification, HTTP integration with a mocked Solana RPC, and downstream benchmark holdout logic. Real browser-wallet signing and CUDA generation still require the user's machine.

## Repository layout

```text
aura-web/                 React/Vite product UI + Judge Mode
aura_library_server.py    local motion/preference/market coordinator
aura_preference.py        pairwise learning
aura_constraints.py       native G1 constraint checks
aura_downstream_benchmark.py
aura_g1_metrics.py
aura_bounties.py          local bounty + curation ledger
text2motion-aura/              compatibility path for vendored NVIDIA Kimodo source used by Aura
scripts/                  local run/test helpers
```

## Claims and limitations

Aura's current automated measurements are kinematic proxies. They are not dynamics, collision checks, real-robot task success, or safety certification. The downstream benchmark is an offline imitation proxy, not physical G1 training. Solana receipts prove that the verified transaction contained the specified signer/payment/evidence fields; they do not prove that the underlying motion judgment or benchmark conclusion is correct.

## Third-party software and assets

See `THIRD_PARTY.md`. NVIDIA Kimodo's code and model weights/assets have separate licensing terms. Aura is an independent integration and is not an NVIDIA or Unitree product.

## What still requires the project owner

The repository can supply the software, but it cannot manufacture external evidence. Before submission, the project owner should:

- Run NVIDIA Kimodo on the working GPU and generate the real same-prompt cohort.
- Obtain genuine human comparisons (preferably from multiple people).
- Run the final frozen benchmark and keep the result even if it is negative.
- Connect/fund devnet wallets and run at least one real bounty posting + curator payout end-to-end against devnet. The backend verification logic is implemented and tested with transaction fixtures, but the real wallet flow still needs your machine.
- Deploy `aura-web` in Judge Mode to a public static host, or provide an equivalent clickable build.
- Record the final live GPU/wallet walkthrough.
- Confirm rights to any Aura brand/decorative assets supplied outside the third-party repositories.
- Gather at least a small amount of customer/user feedback if possible.

The local Python API intentionally binds to `127.0.0.1`. Do not expose it to the public internet without adding authentication, authorization, rate limiting, HTTPS, persistence/backups, and stronger marketplace/Sybil controls.

## Solana client dependency

`@solana/web3.js` is now imported from the app dependency graph and bundled by Vite. Aura no longer depends on the unpkg CDN at runtime. Run `npm install` once in `aura-web/` before building; the package versions are pinned in `package.json`.

## Generated G1 playback

The live Motion Studio opens directly on a Motion-vs-Motion comparison using six preloaded AIST++ clips retargeted to the G1. After Aura generates a batch, the same player switches to the newly generated candidates and records same-prompt preferences. The separate AIST++ reference section has been removed.

## Batch-first interaction

The top Motion-vs-Motion arena is also the live generator. Visitors can rate bundled G1 starter/reference motions immediately, type a prompt in the arena, choose 2–6 candidates, and generate a fresh same-prompt batch without scrolling to a separate generator panel. Generated candidates replace the starter matchup and run as a winner-vs-next tournament. Live A/B choices from both starter/reference and generated motions are persisted to Aura; starter records are explicitly tagged as broad motion-quality data, while generated comparisons retain the same-prompt constraint.

The Motion Studio stays focused on generation/comparison and the Generated Motion Library. The Solana curation market is available as its own top-level **Curation** page so the payment/provenance workflow remains visible without cluttering the primary motion workflow. Constraint, discovery, benchmark, and standalone AIST reference tools remain in the repository/backend where applicable.

## AMP-inspired learned motion prior

Aura includes an optional learned G1 motion prior in `aura_motion_prior.py`. It is deliberately **not full AMP** and does not require a physics simulator. The model learns from short G1 state-transition windows (root-relative joint pose, joint velocity, root velocity, and root height) using human generated-motion preference outcomes as weak acceptance targets. After at least six generated-motion comparisons covering at least four distinct G1 motions, Aura can train the prior and score saved/generated candidates.

The UI updates this prior opportunistically after new generated-motion preferences. Scores are shown as an additional learned signal beside the pairwise human-preference workflow. They are not a physical stability, task-success, collision, safety, or real-robot score.

Full Adversarial Motion Priors would require a physics simulator and a trainable robot policy so the discriminator can provide a style reward during reinforcement learning. That remains a future validation layer rather than a hackathon claim.

## Preference-data integrity

Live generation now uses a fresh random seed for each batch unless you explicitly set `AURA_TEXT2MOTION_SEED` or send a seed in the generation request. Every candidate records its generation seed and NPZ SHA-256. If a generated NPZ duplicates an existing library motion, Aura discards it and retries with a new seed before presenting it for rating.

Generated-motion preferences are also de-duplicated. The browser keeps a local evaluator ID, and the backend refuses a second vote from that evaluator on the same content pair. Training re-verifies the saved motion hashes and reports stored, valid, duplicate, rejected, unique-pair and unique-motion counts at `/preferences/diagnostics` and on the **Aura model** page.

A skipped matchup is not written to the training dataset. This is intentional: ambiguous or uniformly bad candidates should not create a forced preference label.
