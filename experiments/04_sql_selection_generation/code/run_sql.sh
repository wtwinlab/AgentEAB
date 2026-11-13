#!/bin/sh
# A4: the deferred SQL stages. Waits for the three-domain collection to finish so the two jobs do
# not contend for the recipient servers, then runs the full BIRD matrix on all 1,534 questions.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
while pgrep -f "run_all.sh" >/dev/null 2>&1; do sleep 60; done
echo "=== three-domain collection finished; starting SQL $(date -u +%H:%M:%SZ) ==="
C=experiments/04_sql_selection_generation/code/collect_sql_arms.py
python3 -u "$C" --stage peers 2>&1 | tee -a "$L/run_sql.log"
python3 -u "$C" --stage arms  2>&1 | tee -a "$L/run_sql.log"
echo "=== SQL COLLECTION COMPLETE $(date -u +%H:%M:%SZ) ==="
