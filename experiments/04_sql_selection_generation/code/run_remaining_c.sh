#!/bin/sh
# Chain C: after the nested pool is collected, re-run D5 on it so the randomness diagnostic is
# outcome-level and not merely input-level.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
C=experiments/04_sql_selection_generation/code
while pgrep -f "run_remaining_b.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== controls chain done; D5 on the second pool $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_controls.py" --stage nestedarms --dataset musicbrainz20k --split dev --workers 10 2>&1 | tee -a "$L/run_controls.log"
echo "=== NESTED ARMS COMPLETE $(date -u +%H:%M:%SZ) ==="
