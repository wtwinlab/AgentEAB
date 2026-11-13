#!/bin/sh
# Full three-domain collection. Order: GS (dev+test) -> WDC (dev+test) -> MusicBrainz (dev+test).
# Per dataset+split: pool -> verify -> all eight conditions, each stage in one session.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code"
C=experiments/04_sql_selection_generation/code/collect_matrix.py
L=results/three_domain_four_question_2026-10-15/logs
for ds in geographic_settlements wdc_products musicbrainz20k; do
  for sp in dev test; do
    for st in pool verify arms; do
      echo "=== $ds / $sp / $st  $(date -u +%H:%M:%SZ) ==="
      python3 -u "$C" --dataset "$ds" --split "$sp" --stage "$st" 2>&1 | tee -a "$L/run_${ds}_${sp}.log"
    done
  done
  echo "=== $ds COMPLETE $(date -u +%H:%M:%SZ) ==="
done
echo "=== ALL COLLECTION COMPLETE $(date -u +%H:%M:%SZ) ==="
