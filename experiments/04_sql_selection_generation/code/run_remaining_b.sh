#!/bin/sh
# Chain B: the remaining required controls. Waits for the MusicBrainz token-cap control, then runs
# the same control on GS and WDC dev, then the nested independent-pool diagnostic.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/05_matching_downstream/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
C=experiments/04_sql_selection_generation/code
while pgrep -f "collect_controls.py --stage tokencap --dataset musicbrainz20k" >/dev/null 2>&1; do sleep 30; done
echo "=== MB token cap done; GS token cap $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_controls.py" --stage tokencap --dataset geographic_settlements --split dev --workers 8 2>&1 | tee -a "$L/run_controls.log"
echo "=== WDC token cap $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_controls.py" --stage tokencap --dataset wdc_products --split dev --workers 8 2>&1 | tee -a "$L/run_controls.log"
echo "=== nested independent pool, MusicBrainz dev $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_controls.py" --stage nestedpool --dataset musicbrainz20k --split dev --workers 8 2>&1 | tee -a "$L/run_controls.log"
echo "=== CONTROLS COMPLETE $(date -u +%H:%M:%SZ) ==="
