# (bash 3.2 compatible; macOS default /bin/bash)
# Source me:  source runbook/env.sh
# One place for every constant of the v65.1 paper campaign.
export REPO="$HOME/Desktop/RLEvacuationCollab"
export PY="$REPO/.venv/bin/python"
export LAUNCH_SEED=6500
export CONFIG="config/state_college_spokane_regime_v65_2500_paper.json"
export CITIES="state_college_pa,spokane_wa"
export CAMPAIGN="runs/paper_v65"
export STRATEGIES="rl,rl_early_search,rl_forecast_ablated,rl_precommit,objective_greedy,objective_greedy_legacy,heuristic,random,initial_only,static_greedy,belief_rollout"
export MPLBACKEND=Agg
cd "$REPO" || return 1

# bg NAME CMD... : run CMD detached under caffeinate; log runs/_logs/NAME.log; pid in runs/_logs/NAME.pid
bg() {
  local name="$1"; shift
  mkdir -p runs/_logs
  nohup /usr/bin/caffeinate -dimsu "$@" >> "runs/_logs/$name.log" 2>&1 &
  echo $! > "runs/_logs/$name.pid"
  echo "started $name (pid $(cat runs/_logs/$name.pid)); log: runs/_logs/$name.log"
}
alive() { local p; p=$(cat "runs/_logs/$1.pid" 2>/dev/null) && kill -0 "$p" 2>/dev/null && echo "$1: running (pid $p)" || echo "$1: NOT running"; }
