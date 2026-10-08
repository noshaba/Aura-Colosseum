# Submission experiment protocol

Use this protocol for results you intend to cite. Do not cite Judge Mode or the bundled Text2Motion Aura examples as Aura performance evidence.

## Cohort

Use one exact prompt and one Text2Motion Aura model/configuration. Generate at least 12 candidates; 20 is preferable. Keep prompt, model, duration, diffusion settings and skeleton fixed. Different random seeds are fine.

## Human comparisons

Collect pairwise decisions under one written criterion. Prefer multiple independent evaluators. Do not reveal Aura's screening/ranking scores while a person is voting. Record evaluator count and total comparisons.

## Freeze

Train Aura once and record its model version, source motion IDs and dataset hash. Do not retrain after producing the final test motions.

## Selector-unseen holdout

After freezing Aura, generate fresh same-prompt motions. Do not vote on them. The benchmark code refuses a holdout that appeared in the selector's training snapshot.

## Comparison

Use the same training-set size and same downstream learner for:

- Aura-selected demonstrations.
- Random demonstration sets, repeated with the fixed benchmark seed.

Report all provided metrics and the fraction of random trials Aura beats. Do not cherry-pick only favorable metrics.

## Stronger optional validation

The offline ridge-regression imitation test is a proxy. A stronger next step is to test Aura-selected versus random demonstrations using the same policy-training recipe in a physics simulator, then compare actual task success. This repository does not claim that experiment has been done.

## Solana evidence

Anchor the final immutable report hash only after the experiment is frozen. An on-chain receipt is provenance, not validation of the scientific conclusion.
