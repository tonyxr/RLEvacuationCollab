#!/bin/bash
# Usage: runbook/results.sh [--check]   -- every table and figure into runs/paper_v65/paper_results
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
"$PY" scripts/paper_results.py --campaign "$CAMPAIGN" \
  --training runs/paper_v65_seed1 runs/paper_v65_seed2 runs/paper_v65_seed3 \
  --out "$CAMPAIGN/paper_results" "$@"
