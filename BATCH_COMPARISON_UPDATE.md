# Aura batch generation + motion-vs-motion update

The top Motion Studio interaction now treats human comparison as the primary output of generation.

## Generation

- Every prompt generates **at least two** G1 motion candidates.
- The UI exposes a **2–6 candidate slider** (default: 2).
- Candidates are generated sequentially with fresh per-batch seeds. Aura records each candidate seed and rejects duplicate NPZ hashes before admitting a candidate to the comparison, while reusing the loaded NVIDIA Kimodo model to keep peak VRAM manageable on 16 GB GPUs.
- All candidates share a `batch_id`, exact prompt, and candidate index in their saved metadata.

## Loading stage

While the request is running, the main player shows Aura's AIST++ retargeted loading animation with end-effector trails. The centered overlay reads **AURA IS GENERATING** and shows the candidate count and elapsed time.

## Motion vs motion

When a batch completes, Aura immediately displays two generated G1 motions side by side. The user picks Motion A or Motion B. The preference is written to Aura's real preference database.

For batches larger than two, Aura runs a simple tournament: the preferred motion advances and faces the next candidate until one batch winner remains. This yields `N-1` human comparisons for a batch of `N` candidates.

The batch winner is a human preference result, not a physical-safety or task-success claim.
