#!/bin/bash
# Usage: runbook/freeze.sh   -- commit v65.1 and tag the frozen dynamics (G0)
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
set -e
if ! git diff --quiet HEAD; then
  echo "Tracked files differ from HEAD; commit the v65.1 patch first:"; git status --short --untracked-files=no; exit 1
fi
untracked=$(git ls-files --others --exclude-standard -- '*.py' 'config/*.json' | grep -v '^Claude outputs/' || true)
if [[ -n "$untracked" ]]; then
  echo "Untracked code/config (commit or move it out of the repo):"; echo "$untracked"; exit 1
fi
git tag -a v65.1-freeze -m "v65.1 frozen dynamics for the TR-C campaign (G0)"
"$PY" - <<'PY'
import json, sys
sys.path.insert(0, ".")
from multicity_backtest import POLICY_DYNAMICS_SOURCES, PROJECT_ROOT
from PolicyCache import source_fingerprint
fp = source_fingerprint(PROJECT_ROOT, POLICY_DYNAMICS_SOURCES)
open("runs/v65.1_freeze_fingerprint.json", "w").write(json.dumps(fp, indent=1, sort_keys=True))
print("fingerprint of", len(fp), "dynamics sources -> runs/v65.1_freeze_fingerprint.json")
PY
git rev-parse HEAD > runs/v65.1_freeze_commit.txt
echo "frozen at $(cat runs/v65.1_freeze_commit.txt)"
