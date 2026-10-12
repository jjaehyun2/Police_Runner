#!/bin/bash
# Joint role + barricade search, then the verification report (bar_report.py).
# Usage: run_bar_pipeline.sh <run_dir> <generations>
set -u
cd "$(dirname "$0")"
RUN=$1; GENS=$2
mkdir -p "$RUN"
PY=/workspace/cluster/.venv/bin/python
echo "$(date -Is) train start" >> "$RUN/pipeline.log"
$PY bar_ga.py "$RUN" "$GENS" > "$RUN/train.out" 2>&1 || { echo "$(date -Is) train FAILED" >> "$RUN/pipeline.log"; exit 1; }
echo "$(date -Is) train done; report start" >> "$RUN/pipeline.log"
$PY bar_report.py "$RUN" > "$RUN/report.out" 2>&1 || { echo "$(date -Is) report FAILED" >> "$RUN/pipeline.log"; exit 1; }
echo "$(date -Is) report done" >> "$RUN/pipeline.log"
