#!/bin/sh
set -u
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
NA="$R/experiments/02_controlled_sharing/code"
NR="$R/results/protocol_all_matching_20261018T192619Z"
for pop in geographic_settlements wdc_products musicbrainz20k; do
  for al in llama3-8b-g1 qwen25-7b-g2; do
    python3 "$NA/collect_all_matching.py" --population $pop --protocol dmad --alias $al \
      --repeat 0 --limit 12 --workers 12 --attempt 1 \
      --out "$NR/pilot/${pop}__dmad__${al}__r0__a1.jsonl" >> "$NR/logs/pilot.log" 2>&1 &
  done
  python3 "$NA/collect_all_matching.py" --population $pop --protocol ford --repeat 0 \
    --limit 12 --workers 12 --attempt 1 \
    --out "$NR/pilot/${pop}__ford__team__r0__a1.jsonl" >> "$NR/logs/pilot.log" 2>&1 &
done
wait
echo "PILOT RETRY DONE $(date -u +%H:%M:%S)" >> "$NR/logs/pilot.log"
