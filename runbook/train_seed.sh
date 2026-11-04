#!/bin/bash
# Usage: runbook/train_seed.sh K [--resume]
# Launches (or resumes) training shard K of 3 in the background.
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
K="$1"; shift
case "$K" in 1|2|3) ;; *) echo "usage: train_seed.sh 1|2|3 [--resume]"; exit 2;; esac
bg "train_seed$K" "$PY" multicity_backtest.py \
  --cities "$CITIES" --launch-seed "$LAUNCH_SEED" --launch-id "paper_v65_seed$K" --machine local \
  --policy-replicates 3 --training-policy-indices "$K" \
  --train-episodes-per-city 100 --training-curriculum "$CONFIG" \
  --eval-replications-per-city 20 --strategies "$STRATEGIES" \
  --no-require-convergence --visualize-eval-pairs-per-city 0 --no-policy-cache --train-only "$@"
