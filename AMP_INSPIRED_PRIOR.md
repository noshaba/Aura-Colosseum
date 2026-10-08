# Aura AMP-inspired motion prior

Aura now contains a small learned motion-prior model that operates directly on G1 trajectory transitions.

## What it learns

For each G1 transition, Aura uses:

- root-relative 34-joint positions,
- 34-joint world-space velocities,
- root velocity,
- root height.

Human A/B preference outcomes are aggregated into smoothed per-motion acceptance targets. A two-hidden-layer discriminator is trained on local transition windows and produces a 0–1 learned-prior score for generated G1 motions.

## Why this is AMP-inspired, not full AMP

The design borrows the useful AMP idea of learning a discriminator on local motion transitions rather than only hand-authored metrics. However:

- no robot policy is trained,
- no adversarial policy/discriminator loop runs,
- no physics engine is involved,
- no dynamic stability or task-success claim is made.

Full AMP would require a physics simulator (for example Isaac Lab/Gym or MuJoCo), a G1 policy, task rewards, and discriminator style rewards during RL.

## API

- `POST /prior/train` trains/updates the learned prior after enough generated-motion comparisons exist.
- `GET /prior` returns model metadata and scores for saved G1 motions.

The main comparison UI attempts a lightweight prior update after generated-motion votes once enough data is available.
