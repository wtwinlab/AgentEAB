#!/bin/sh
# Chain B: the nested independent-pool diagnostic on GS and WDC,
# under the same frozen scheme used for MusicBrainz (10% of dev pairs, six models, three repeats,
# then D5 on the second pool so the diagnostic is outcome-level).
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
C=experiments/04_sql_selection_generation/code
while pgrep -f "run_gapfix_a.sh" >/dev/null 2>&1; do sleep 30; done
for d in geographic_settlements wdc_products; do
  echo "=== nested pool $d $(date -u +%H:%M:%SZ) ==="
  python3 -u "$C/collect_controls.py" --stage nestedpool --dataset $d --split dev --workers 8 2>&1 | tee -a "$L/run_gapfix.log"
  echo "=== D5 on pool2 $d $(date -u +%H:%M:%SZ) ==="
  python3 -u "$C/collect_controls.py" --stage nestedarms --dataset $d --split dev --workers 8 2>&1 | tee -a "$L/run_gapfix.log"
done
echo "=== NESTED POOLS COMPLETE $(date -u +%H:%M:%SZ) ==="
