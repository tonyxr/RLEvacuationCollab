#!/bin/bash
# Usage: runbook/panel.sh PANEL_ID [PLAIN_WORKERS] [HEAVY_WORKERS]
# Runs (or resumes) a panel in the background and collects it when finished.
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
P="$1"; W="${2:-4}"; H="${3:-1}"
[[ -f "config/panels/$P.json" ]] || { echo "no config/panels/$P.json"; exit 2; }
bg "panel_$P" /bin/bash -c "'$PY' scripts/panel_eval.py run --spec config/panels/$P.json --plain-workers $W --heavy-workers $H && '$PY' scripts/panel_eval.py collect --spec config/panels/$P.json"
