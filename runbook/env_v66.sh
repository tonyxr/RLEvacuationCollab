# (bash 3.2 compatible; macOS default /bin/bash)
# Source me from the v66 clone:  source runbook/env_v66.sh
# Never source runbook/env.sh in the clone: it cd's into the campaign checkout.
export REPO="$HOME/Desktop/RLEvacuationCollab-v66"
export CAMPAIGN_REPO="$HOME/Desktop/RLEvacuationCollab"
export PY="$REPO/.venv/bin/python"
export LAUNCH_SEED=6500
export CITIES="state_college_pa,spokane_wa"
export STRATEGIES="rl,rl_early_search,rl_forecast_ablated,rl_precommit,objective_greedy,objective_greedy_legacy,heuristic,random,initial_only,static_greedy,belief_rollout"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export V66="$CAMPAIGN_REPO/Claude outputs/v66_validity"
# Path A (default) evaluates the v65.1 checkpoints; Path B the retrained ones.
# Set V66_PATH=B before sourcing for Path B.  Each path has its own campaign.
export V66_PATH="${V66_PATH:-A}"
if [ "$V66_PATH" = B ]; then
  export CAMPAIGN=runs/paper_v66b SL_CAMPAIGN=runs/paper_v66b_site_loss SPECS=config/panels/v66b
  export TRAINING="runs/paper_v66_seed1 runs/paper_v66_seed2 runs/paper_v66_seed3"
else
  export CAMPAIGN=runs/paper_v66 SL_CAMPAIGN=runs/paper_v66_site_loss SPECS=config/panels/v66
  export TRAINING="runs/paper_v65_seed1 runs/paper_v65_seed2 runs/paper_v65_seed3"
fi
export AN="$CAMPAIGN/analysis"
export FU="$CAMPAIGN_REPO/Claude outputs/v65_followup"
cd "$REPO" || return 1
if [ "$(pwd -P)" = "$(cd "$CAMPAIGN_REPO" && pwd -P)" ]; then
  echo "refusing: this is the campaign checkout"; return 1
fi

# bg NAME CMD... : run CMD detached under caffeinate; log runs/_logs/NAME.log; pid in runs/_logs/NAME.pid
bg() {
  local name="$1"; shift
  mkdir -p runs/_logs
  nohup /usr/bin/caffeinate -dimsu "$@" >> "runs/_logs/$name.log" 2>&1 &
  echo $! > "runs/_logs/$name.pid"
  echo "started $name (pid $(cat runs/_logs/$name.pid)); log: runs/_logs/$name.log"
}
alive() { local p; p=$(cat "runs/_logs/$1.pid" 2>/dev/null) && kill -0 "$p" 2>/dev/null && echo "$1: running (pid $p)" || echo "$1: NOT running"; }

spec() { case "$1" in v66_identity) echo "config/panels/v66/$1.json" ;; *) echo "$SPECS/$1.json" ;; esac; }
# Specs whose learned arms read v65.1 checkpoints need the dynamics-change flag.
flag() { grep -q '"runs/paper_v65_' "$(spec "$1")" && echo "--allow-dynamics-change"; }
csv() {
  case "$1" in
    v66_identity) echo "runs/paper_v66/panels/$1/evaluation_episode_summary.csv" ;;
    site_loss*) echo "$SL_CAMPAIGN/panels/$1/evaluation_episode_summary.csv" ;;
    *) echo "$CAMPAIGN/panels/$1/evaluation_episode_summary.csv" ;;
  esac
}
v66_plan()    { "$PY" scripts/panel_eval.py plan    --spec "$(spec "$1")" $(flag "$1"); }
v66_status()  { "$PY" scripts/panel_eval.py status  --spec "$(spec "$1")" $(flag "$1"); }
v66_collect() { "$PY" scripts/panel_eval.py collect --spec "$(spec "$1")" $(flag "$1"); }
v66_launch() {  # usage: v66_launch PANEL_ID PLAIN_WORKERS HEAVY_WORKERS
  local s f; s="$(spec "$1")"; f="$(flag "$1")"
  bg "v66_$1" /bin/bash -c "\"$PY\" scripts/panel_eval.py run --spec \"$s\" $f --plain-workers $2 --heavy-workers $3 && \"$PY\" scripts/panel_eval.py collect --spec \"$s\" $f"
}
# v66_wait PANEL_ID : block until the panel's background job ends, then show its status.
v66_wait() {
  while kill -0 "$(cat "runs/_logs/v66_$1.pid" 2>/dev/null)" 2>/dev/null; do sleep 120; done
  v66_status "$1" | tail -n 3
}
chk() { "$PY" "$FU/followup_checks.py" "$@"; }
ana() { "$PY" "$V66/tools/v66_analysis.py" "$@"; }
v66_train() {  # usage: v66_train LAUNCH_ID CURRICULUM_PATH EPISODES_PER_CITY POLICY_INDEX [--resume]
  local id="$1" cur="$2" n="$3" k="$4"; shift 4
  bg "train_$id" "$PY" multicity_backtest.py --cities "$CITIES" --launch-seed "$LAUNCH_SEED" \
    --launch-id "$id" --machine local --policy-replicates 3 --training-policy-indices "$k" \
    --train-episodes-per-city "$n" --training-curriculum "$cur" \
    --eval-replications-per-city 20 --strategies "$STRATEGIES" --no-require-convergence \
    --visualize-eval-pairs-per-city 0 --no-policy-cache --train-only "$@"
}
