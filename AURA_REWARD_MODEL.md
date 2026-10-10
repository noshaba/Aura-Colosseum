# Aura Motion Reward Model

Aura's primary learned model is a small temporal Transformer trained from real
same-prompt human A/B preferences over generated Unitree G1 trajectories.

## What it learns

For a motion trajectory `m`, the model outputs a scalar reward `r(m)`. A human
preference for motion A over motion B trains the model with the pairwise objective:

```text
P(A > B) = sigmoid(r(A) - r(B))
loss = binary_cross_entropy_with_logits(r(A) - r(B), human_choice)
```

The model consumes robot-native trajectory information rather than Aura's
handcrafted screening summaries:

- root-relative positions for all 34 G1 joints;
- joint velocities;
- root velocity and root height;
- 6D global joint rotations when the NPZ contains rotation matrices.

Each motion is resampled to a fixed sequence length and encoded by a compact
2-layer temporal Transformer. The default model is intentionally small and
trains on CPU so it does not compete with NVIDIA Kimodo for GPU VRAM.

## Is this RLHF?

This is the **reward-modeling / human-preference-learning component of RLHF**.
Calling the current system "full RLHF" would be inaccurate because Aura does not
yet update NVIDIA Kimodo or a robot control policy using reinforcement learning.

A future full-RLHF robotics loop would be:

```text
human comparisons -> Aura reward model -> robot policy in physics simulation
                                      -> RL policy update -> new trajectories
```

The current prototype stops after learning and applying the reward model for
ranking/selection.

## Minimum data

Training starts after at least six valid generated-motion comparisons spanning
at least four distinct G1 motions. More comparisons and multiple independent
raters are strongly preferred. A held-out pair metric is reported only when the
comparison graph is large enough; it is not presented as physical robot success.

## Bias control

The live A/B UI does not display Aura reward scores while the user is making the
choice. Scores/rankings are revealed after voting so the model does not influence
its own labels.

## API

```text
GET  /reward        current model metadata and motion rankings
POST /reward/train  train/retrain from persisted generated-motion preferences
```

The existing pairwise logistic selector remains in the repository as an
interpretable baseline. The AMP-inspired motion prior remains a separate auxiliary
signal. Hard kinematic constraints are also kept separate.

## Training-data integrity

Aura does not use the raw SQLite row count as its training threshold. Before training it:

1. re-hashes each referenced G1 NPZ and verifies it matches the hash stored with the human vote;
2. rejects comparisons where the two candidates have identical motion bytes;
3. de-duplicates repeated votes from the same evaluator on the same content pair;
4. counts distinct motions by NPZ hash rather than filename or database ID.

The live `/preferences/diagnostics` endpoint reports stored rows, valid comparisons, unique motion pairs, unique motion hashes, duplicate rows and rejected rows. The web model page exposes these counts directly.

The neural reward model becomes trainable at six valid comparisons spanning at least four distinct motion hashes. These are implementation minimums, not a claim that six labels are enough for a scientifically reliable model.
