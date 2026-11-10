#!/bin/bash
# Site-loss corner case and its redo with forward-looking evacuation orders.
# bash 3.2 compatible (macOS /bin/bash). Run it from the site-loss worktree
# (~/Desktop/RLEvacuationCollab-siteloss), never from the campaign checkout.
#
# Panels (PANEL):
#   pilot | full              site_loss                    no recourse (done 2026-11-09)
#   rpilot | rfull            site_loss_recourse           off + forecast orders fL30 / fL15
#   rrollpilot | rroll        site_loss_recourse_rollout   off + fL30 + simulator-graded stopping
#
#   bash runbook/site_loss_run.sh preflight            read-only checks; ends PREFLIGHT PASS|FAIL
#   bash runbook/site_loss_run.sh resources [W]        memory/disk/process rule; ends RESOURCES OK|WAIT
#   bash runbook/site_loss_run.sh plan-PANEL           job counts only
#   bash runbook/site_loss_run.sh launch PANEL W       checks, then runs in screen v65_site_loss_PANEL
#   bash runbook/site_loss_run.sh status-PANEL
#   bash runbook/site_loss_run.sh gates-PANEL          all gates; ends GATES PASS|FAIL
#   bash runbook/site_loss_run.sh identity-main        base cells vs the main panel (after it is collected)
#   bash runbook/site_loss_run.sh stop                 interrupt the site-loss run; finished episodes stay
#   bash runbook/site_loss_run.sh PANEL W              the foreground job that 'launch' starts in screen
set -u
unset PYTHONPATH PYTHONHOME
SL="$(cd "$(dirname "$0")/.." && pwd -P)"
MAIN="${MAIN:-$HOME/Desktop/RLEvacuationCollab}"
if [ -d "$MAIN" ]; then MAIN="$(cd "$MAIN" && pwd -P)"; fi
PY="$MAIN/.venv/bin/python"
CHECKS="$SL/runbook/site_loss_checks.py"
STORE="$MAIN/runs/paper_v65_site_loss"
LOGDIR="$MAIN/runs/_logs"
G2_REFERENCE="$MAIN/runs/paper_v65/panels/g2_seed1/evaluation_episode_summary.csv"
MAIN_REFERENCE="$MAIN/runs/paper_v65/panels/main/evaluation_episode_summary.csv"
PATCH="$MAIN/Claude outputs/site_loss/0001-Corner-case-initial-site-loss-by-a-spot-fire-evaluat.patch"
FREEZE="f3edfb2e3f8ce23b44066878e77fbc75c8731c85"
RUN_PREFIX="scripts/panel_eval.py run --spec config/panels/site_loss"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ "$SL" = "$MAIN" ]; then
  echo "Refusing to run in the campaign checkout ($MAIN). Use the site-loss worktree."; exit 2
fi
cd "$SL" || exit 2

MODE="${1:-}"
case "$MODE" in
  launch) PANEL="${2:-}"; WORKERS="${3:-1}" ;;
  pilot|full|rpilot|rfull|rrollpilot|rroll) PANEL="$MODE"; WORKERS="${2:-1}" ;;
  resources) PANEL=""; WORKERS="${2:-1}" ;;
  plan-*|status-*|gates-*) PANEL="${MODE#*-}"; WORKERS=1 ;;
  identity-main) PANEL=full; WORKERS=1 ;;
  *) PANEL=""; WORKERS=1 ;;
esac
case "$WORKERS" in 1|2|3) ;; *) echo "workers must be 1, 2 or 3 (got '$WORKERS')"; exit 2 ;; esac

S1="$MAIN/runs/paper_v65_seed1"; S2="$MAIN/runs/paper_v65_seed2"; S3="$MAIN/runs/paper_v65_seed3"
EXPECTED=0; MIN_COMMON=0; OUT=""; KIND=""; PANEL_ID=""; SPEC=""; NEEDS_SEED3=0; EXTRA=()
case "$PANEL" in
  pilot)      SPEC=config/panels/site_loss.json; PANEL_ID=site_loss; KIND=site
              EXTRA=(--training "$S1" "$S2" --policies 1,2 --replications 1:2)
              EXPECTED=120; MIN_COMMON=8; OUT="$STORE/site_loss_results_pilot" ;;
  full)       SPEC=config/panels/site_loss.json; PANEL_ID=site_loss; KIND=site; NEEDS_SEED3=1
              EXTRA=(--training "$S1" "$S2" "$S3")
              EXPECTED=1400; MIN_COMMON=80; OUT="$STORE/site_loss_results_full" ;;
  rpilot)     SPEC=config/panels/site_loss_recourse.json; PANEL_ID=site_loss_recourse; KIND=recourse
              EXTRA=(--training "$S1" "$S2" --policies 1,2 --replications 1:2)
              EXPECTED=360; MIN_COMMON=8; OUT="$STORE/recourse_results_pilot" ;;
  rfull)      SPEC=config/panels/site_loss_recourse.json; PANEL_ID=site_loss_recourse; KIND=recourse; NEEDS_SEED3=1
              EXTRA=(--training "$S1" "$S2" "$S3")
              EXPECTED=4200; MIN_COMMON=80; OUT="$STORE/recourse_results_full" ;;
  rrollpilot) SPEC=config/panels/site_loss_recourse_rollout.json; PANEL_ID=site_loss_recourse_rollout
              KIND=recourse; NEEDS_SEED3=1
              EXTRA=(--training "$S1" "$S2" "$S3" --replications 1:1)
              EXPECTED=36; MIN_COMMON=4; OUT="$STORE/recourse_results_pilot" ;;
  rroll)      SPEC=config/panels/site_loss_recourse_rollout.json; PANEL_ID=site_loss_recourse_rollout
              KIND=recourse; NEEDS_SEED3=1
              EXTRA=(--training "$S1" "$S2" "$S3")
              EXPECTED=360; MIN_COMMON=40; OUT="$STORE/recourse_results_full" ;;
  "") ;;
  *) echo "panel must be pilot, full, rpilot, rfull, rrollpilot or rroll (got '$PANEL')"; exit 2 ;;
esac
ARGS=(--spec "$SPEC" --campaign "$STORE")
if [ -n "$PANEL" ]; then ARGS=("${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}); fi
SUMMARY="$STORE/panels/$PANEL_ID/evaluation_episode_summary.csv"
WHERE=()
if [ "$KIND" = recourse ]; then WHERE=(--where label_recourse=off); fi

# Python processes of the site-loss panels (the caffeinate wrapper and shells are excluded).
site_loss_pids() {
  ps axo pid=,command= | grep -F "$RUN_PREFIX" | grep -F -e "--campaign $STORE" \
    | grep -v -e caffeinate -e grep -e "bash -c" | awk '{print $1}'
}
# Any other panel_eval run on this machine (campaign panels).
other_panel_pids() {
  ps axo pid=,command= | grep -F "panel_eval.py run" | grep -v -F "config/panels/site_loss" \
    | grep -v -e caffeinate -e grep -e "bash -c" | awk '{print $1}'
}
status_of() { "$PY" "$CHECKS" status "$1" 2>&1 | tail -1; }

preflight() {
  local fail=0 graph
  check() { if [ "$2" = "$3" ]; then echo "[PASS] $1"; else echo "[FAIL] $1: got '$2', want '$3'"; fail=1; fi; }
  check "campaign HEAD is the freeze" "$(git -C "$MAIN" rev-parse HEAD 2>&1)" "$FREEZE"
  check "tag v65.1-freeze" "$(git -C "$MAIN" rev-parse 'v65.1-freeze^{commit}' 2>&1)" "$FREEZE"
  check "campaign tracked changes limited to LOG.md/TODO.md" \
        "$(git -C "$MAIN" status --porcelain --untracked-files=no | grep -v -E '^ M (LOG|TODO)\.md$')" ""
  check "worktree HEAD is the freeze (patch applied, not committed)" "$(git -C "$SL" rev-parse HEAD 2>&1)" "$FREEZE"
  check "only tracked change in the worktree is scripts/panel_eval.py" \
        "$(git -C "$SL" status --porcelain --untracked-files=no)" " M scripts/panel_eval.py"
  check "patch file present" "$(test -f "$PATCH" && echo yes)" "yes"
  check "patch fully applied in the worktree" \
        "$(git -C "$SL" apply --reverse --check "$PATCH" >/dev/null 2>&1 && echo yes)" "yes"
  check "cache is a link to the campaign cache" \
        "$(test -L "$SL/cache" && cd "$SL/cache" && pwd -P)" "$MAIN/cache"
  for graph in graph_walk_State_College__PA__USA_6cb8fa0092e2.graphml \
               graph_walk_Spokane__WA__USA_586ca7735ba9.graphml; do
    check "cache file $graph" "$(test -f "$SL/cache/$graph" && echo yes)" "yes"
  done
  check "python stack" "$("$PY" "$CHECKS" stack 2>/dev/null | tail -1)" "3.10.8 2.5.1 1.26.4"
  check "worktree modules import from the worktree" "$("$PY" "$CHECKS" imports "$SL" 2>&1 | tail -1)" "yes"
  check "worktree dynamics fingerprint equals the frozen one (32 files)" \
        "$("$PY" "$CHECKS" fingerprint "$MAIN/runs/v65.1_freeze_fingerprint.json" "$SL" 2>&1 | tail -1)" "identical"
  check "G2 reference panel collected (80 rows)" "$("$PY" "$CHECKS" rows "$G2_REFERENCE" 2>&1 | tail -1)" "80"
  check "seed 1 complete" "$(status_of "$MAIN/runs/paper_v65_seed1/experiment_manifest.json")" "training_shard_complete"
  check "seed 2 complete" "$(status_of "$MAIN/runs/paper_v65_seed2/experiment_manifest.json")" "training_shard_complete"
  echo "[INFO] seed 3: $("$PY" "$CHECKS" seed3 "$MAIN" 2>&1 | tail -1) (full-size panels need training_shard_complete)"
  echo "[INFO] site_loss full panel collected: $("$PY" "$CHECKS" rows "$STORE/panels/site_loss/evaluation_episode_summary.csv" 2>&1 | tail -1) rows"
  echo "[INFO] cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "[INFO] site-loss run processes alive: $(site_loss_pids | tr '\n' ' ')"
  echo "[INFO] site-loss screen sessions: $(screen -ls 2>/dev/null | awk '/v65_site_loss_/{print $1}' | tr '\n' ' ')"
  if [ "$fail" = 0 ]; then echo "PREFLIGHT PASS"; else echo "PREFLIGHT FAIL"; fi
  return "$fail"
}

resources() { "$PY" "$CHECKS" resources "$MAIN" "$WORKERS" $(other_panel_pids); }

launch() {
  local session="v65_site_loss_$PANEL" log="$LOGDIR/site_loss_$PANEL.log" s3
  if screen -ls 2>/dev/null | grep -q "\.$session[[:space:]]"; then
    echo "Screen session $session already exists. Do not launch twice."; return 1
  fi
  if [ -n "$(site_loss_pids)" ]; then
    echo "A site-loss run process is already alive: $(site_loss_pids | tr '\n' ' '). Not launching."; return 1
  fi
  if ! preflight > /dev/null 2>&1; then
    echo "PREFLIGHT FAIL: run 'bash runbook/site_loss_run.sh preflight' and resolve it first."; return 1
  fi
  if ! resources; then echo "Not launching: the resource rule is not met."; return 1; fi
  if [ "$NEEDS_SEED3" = 1 ]; then
    s3="$(status_of "$MAIN/runs/paper_v65_seed3/experiment_manifest.json")"
    if [ "$s3" != training_shard_complete ]; then
      echo "Seed 3 status is '$s3'; panel $PANEL needs training_shard_complete."; return 1
    fi
  fi
  mkdir -p "$LOGDIR"
  local before=0
  if [ -f "$log" ]; then before="$(wc -l < "$log" | tr -d ' ')"; fi
  screen -dmS "$session" /bin/bash -c "MAIN='$MAIN' /bin/bash '$SL/runbook/site_loss_run.sh' $PANEL $WORKERS >> '$log' 2>&1"
  sleep 5
  if screen -ls 2>/dev/null | grep -q "\.$session[[:space:]]"; then
    echo "[$(now)] LAUNCHED screen session $session (panel $PANEL, $WORKERS worker(s)); log: $log"
  elif tail -n +"$((before + 1))" "$log" 2>/dev/null | grep -q "SITE_LOSS_${PANEL}_DONE"; then
    echo "[$(now)] COMPLETED within 5 s (no episode left to run); last line: $(tail -n 1 "$log")"
  else
    echo "Screen session $session is not alive 5 s after launch and wrote no DONE line; read $log"; return 1
  fi
}

analysis() {
  if [ "$KIND" = recourse ]; then
    "$PY" scripts/site_loss_recourse_results.py --campaign "$STORE" --out "$OUT" --check
  else
    "$PY" scripts/site_loss_results.py --campaign "$STORE" --out "$OUT" --check
  fi
}

identity_g2() {
  "$PY" scripts/site_loss_identity.py --site-loss "$SUMMARY" --reference "$G2_REFERENCE" \
      --min-common "$MIN_COMMON" ${WHERE[@]+"${WHERE[@]}"}
}

run_panel() {
  local run collect=skipped results=skipped identity=skipped
  echo "[$(now)] SITE_LOSS_${PANEL}_START workers=$WORKERS expected_jobs=$EXPECTED spec=$SPEC worktree=$SL"
  /usr/bin/caffeinate -dimsu "$PY" scripts/panel_eval.py run "${ARGS[@]}" \
      --plain-workers "$WORKERS" --heavy-workers 0
  run=$?
  echo "[$(now)] run exit $run"
  if [ "$run" = 0 ]; then
    "$PY" scripts/panel_eval.py collect "${ARGS[@]}"
    collect=$?
    echo "[$(now)] collect exit $collect"
  fi
  if [ "$collect" = 0 ]; then
    analysis
    results=$?
    echo "[$(now)] results exit $results"
    identity_g2
    identity=$?
    echo "[$(now)] identity exit $identity"
  fi
  echo "[$(now)] SITE_LOSS_${PANEL}_DONE run=$run collect=$collect results=$results identity=$identity"
}

gates() {
  local identity results
  mkdir -p "$OUT"
  identity_g2 > "$OUT/identity_gate_$PANEL.txt" 2>&1
  identity=$?
  if [ "$KIND" = recourse ]; then
    analysis > "$OUT/analysis_$PANEL.txt" 2>&1
    results=$?
    "$PY" "$CHECKS" recourse_gates "$STORE" "$OUT" "$EXPECTED" "$PANEL" "$PANEL_ID" \
        "$results" "$OUT/analysis_$PANEL.txt" "$identity" "$OUT/identity_gate_$PANEL.txt"
  else
    cp "$OUT/identity_gate_$PANEL.txt" "$OUT/identity_gate.txt"
    "$PY" "$CHECKS" gates "$STORE" "$OUT" "$EXPECTED" "$PANEL" "$identity" "$OUT/identity_gate.txt"
  fi
}

identity_main() {
  if [ ! -f "$MAIN_REFERENCE" ]; then echo "The main panel is not collected yet ($MAIN_REFERENCE)."; return 1; fi
  "$PY" scripts/site_loss_identity.py --site-loss "$STORE/panels/site_loss/evaluation_episode_summary.csv" \
      --reference "$MAIN_REFERENCE" --min-common 280
}

stop_run() {
  local pids kids pid i alive="" session
  pids="$(site_loss_pids | tr '\n' ' ')"
  if [ -z "$pids" ]; then
    echo "No site-loss run process is alive."
  else
    kids=""
    for pid in $pids; do kids="$kids $(pgrep -P "$pid" | tr '\n' ' ')"; done
    echo "Interrupting site-loss run $pids (workers:$kids)"
    kill -INT $pids 2>/dev/null
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
      sleep 5
      alive=""
      for pid in $pids $kids; do if kill -0 "$pid" 2>/dev/null; then alive="$alive $pid"; fi; done
      if [ -z "$alive" ]; then break; fi
    done
    if [ -n "$alive" ]; then echo "Still alive after 60 s:$alive; sending TERM"; kill -TERM $alive 2>/dev/null; fi
  fi
  for session in $(screen -ls 2>/dev/null | awk '/v65_site_loss_/{print $1}'); do
    screen -S "$session" -X quit && echo "Closed screen session $session"
  done
  echo "Stopped. Finished episodes stay in $STORE; the same 'launch' command resumes."
}

case "$MODE" in
  preflight) preflight ;;
  resources) resources ;;
  plan-*) "$PY" scripts/panel_eval.py plan "${ARGS[@]}" ;;
  launch) launch ;;
  pilot|full|rpilot|rfull|rrollpilot|rroll) run_panel ;;
  status-*) "$PY" scripts/panel_eval.py status "${ARGS[@]}" ;;
  gates-*) gates ;;
  identity-main) identity_main ;;
  stop) stop_run ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
