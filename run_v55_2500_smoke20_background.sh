#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
python_bin="$project_dir/.venv/bin/python"
launch_seed="${RLEVAC_LAUNCH_SEED:-5622}"
launch_stamp="$(date +%Y%m%d_%H%M%S)"
launch_id="${RLEVAC_LAUNCH_ID:-state_college_2500_rollout_pi_replay12_smoke20_eval_seed${launch_seed}_${launch_stamp}}"

export MPLCONFIGDIR="/private/tmp/rlevacuation_mpl_v55_smoke20"
export PYTHONUNBUFFERED=1
cd "$project_dir"
print -r -- "$$" > "$project_dir/runs/${launch_id}.pid"

"$python_bin" multicity_backtest.py \
  --cities state_college_pa \
  --launch-seed "$launch_seed" \
  --launch-id "$launch_id" \
  --machine local \
  --policy-replicates 1 \
  --train-episodes-per-city 20 \
  --training-curriculum config/state_college_training_smoke_2500_rollout_pi_v55.json \
  --eval-replications-per-city 6 \
  --bootstrap-draws 2000 \
  --strategies rl,objective_greedy,risk_reduction,route_saving,heuristic,hazard_weighted,accessibility_deficit,random,initial_only,static_greedy,rl_precommit \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 4 \
  --no-policy-cache
