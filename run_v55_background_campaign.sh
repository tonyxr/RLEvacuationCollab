#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
python_bin="$project_dir/.venv/bin/python"
export MPLCONFIGDIR="/private/tmp/rlevacuation_mpl_v55"
export PYTHONUNBUFFERED=1
cd "$project_dir"
print -r -- "$$" > "$project_dir/runs/v55_background_campaign.pid"

"$python_bin" multicity_backtest.py \
  --cities state_college_pa \
  --launch-seed 5501 \
  --launch-id state_college_250_rollout_pi_v55_smoke12_eval_seed5501_20261008 \
  --machine local \
  --policy-replicates 1 \
  --train-episodes-per-city 12 \
  --training-curriculum config/state_college_training_smoke_250_rollout_pi_v55.json \
  --eval-replications-per-city 6 \
  --bootstrap-draws 2000 \
  --strategies rl,objective_greedy,risk_reduction,route_saving,heuristic,hazard_weighted,accessibility_deficit,random,initial_only,static_greedy,rl_precommit \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 2 \
  --no-policy-cache

"$python_bin" multicity_backtest.py \
  --cities state_college_pa \
  --launch-seed 5511 \
  --launch-id state_college_2500_rollout_pi_v55_full128_eval_seed5511_20261008 \
  --machine local \
  --policy-replicates 1 \
  --train-episodes-per-city 128 \
  --training-curriculum config/state_college_training_curriculum_2500_rollout_pi_v55.json \
  --eval-replications-per-city 20 \
  --bootstrap-draws 5000 \
  --strategies rl,objective_greedy,risk_reduction,route_saving,heuristic,hazard_weighted,accessibility_deficit,random,initial_only,static_greedy,rl_precommit \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 4 \
  --no-policy-cache
