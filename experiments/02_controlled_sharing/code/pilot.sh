#!/bin/sh
# Per-workflow pilot gate: message routing, each protocol's own output contract, parsing against
# raw replies, its own stop rule, gold isolation and complete logging -- on all three populations
# and both backbones, because the BCCS lesson was backbone-specific.
set -u
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
NA="$R/experiments/02_controlled_sharing/code"
NR="$R/results/protocol_all_matching_20261018T192619Z"
N=${N:-12}
for pop in geographic_settlements wdc_products musicbrainz20k; do
  for proto in madm2_naive madm2_objective du_mad bccs liang_mad dov_vote dmad; do
    for al in llama3-8b-g1 qwen25-7b-g2; do
      python3 "$NA/collect_all_matching.py" --population "$pop" --protocol "$proto" \
        --alias "$al" --repeat 0 --limit "$N" --workers 12 \
        --out "$NR/pilot/${pop}__${proto}__${al}__r0.jsonl" \
        >> "$NR/logs/pilot.log" 2>&1 &
    done
  done
  python3 "$NA/collect_all_matching.py" --population "$pop" --protocol ford --repeat 0 \
    --limit "$N" --workers 12 --out "$NR/pilot/${pop}__ford__team__r0.jsonl" \
    >> "$NR/logs/pilot.log" 2>&1 &
  wait
  echo "PILOT $pop done $(date -u +%H:%M:%S)" >> "$NR/logs/pilot.log"
done
echo "PILOT ALL DONE $(date -u +%H:%M:%S)" >> "$NR/logs/pilot.log"
