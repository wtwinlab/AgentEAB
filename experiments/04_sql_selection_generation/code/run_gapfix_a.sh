#!/bin/sh
# Chain A: the WDC/GS evidence conditions.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
C=experiments/04_sql_selection_generation/code
for s in dev test; do
  for arm in E5f E5c; do
    echo "=== wdc $arm $s $(date -u +%H:%M:%SZ) ==="
    python3 -u "$C/collect_fixed_arms.py" --dataset wdc_products --arm $arm --split $s --workers 8 2>&1 | tee -a "$L/run_gapfix.log"
  done
done
for s in dev test; do
  echo "=== gs E5f $s $(date -u +%H:%M:%SZ) ==="
  python3 -u "$C/collect_fixed_arms.py" --dataset geographic_settlements --arm E5f --split $s --workers 8 2>&1 | tee -a "$L/run_gapfix.log"
done
echo "=== FIXED ARMS COMPLETE $(date -u +%H:%M:%SZ) ==="
