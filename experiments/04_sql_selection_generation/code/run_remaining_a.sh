#!/bin/sh
# Chain A: P4 Spider. Waits for MusicBrainz test E5p to finish so the recipient servers are not
# oversubscribed, then runs the Spider peers stage followed by the arms stage.
set -e
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$R"
export PYTHONPATH="experiments/06_cost_resources/code:experiments/04_sql_selection_generation/code"
L=results/three_domain_four_question_2026-10-15/logs
C=experiments/04_sql_selection_generation/code
while pgrep -f "collect_e5p_from_pool" >/dev/null 2>&1; do sleep 30; done
echo "=== E5p test done; starting Spider peers $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_sql_spider.py" --stage peers --workers 10 2>&1 | tee -a "$L/run_spider.log"
echo "=== Spider peers done; starting Spider arms $(date -u +%H:%M:%SZ) ==="
python3 -u "$C/collect_sql_spider.py" --stage arms --workers 10 2>&1 | tee -a "$L/run_spider.log"
echo "=== SPIDER COMPLETE $(date -u +%H:%M:%SZ) ==="
