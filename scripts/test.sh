#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python test_discovery.py
python test_preference_integrity.py
python test_starter_training.py
python test_constraints.py
python test_bounties.py
python test_solana_verification.py
python test_curation_api.py
python test_downstream_benchmark.py
python test_api.py
python test_generator_api.py
python test_generator_uniqueness.py
python test_motion_prior.py
python test_prior_api.py
python test_reward_model.py
python test_reward_api.py
