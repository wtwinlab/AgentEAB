#!/bin/sh
# Phase A: the eight workflows that passed the pilot, on all three common matching populations.
# One pool, no barrier: a finished shard is replaced immediately so no card drains mid-stage.
#   run_phase_a.sh                      run the pool
#   run_phase_a.sh --shard POP PROTO ALIAS REP
set -u
R="${ARTIFACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
NA="$R/experiments/02_controlled_sharing/code"
NR="$R/results/protocol_all_matching_20261018T192619Z"
P=${P:-8}
W=${W:-48}

if [ "${1:-}" = "--shard" ]; then
  pop="$2"; proto="$3"; al="$4"; rep="$5"
  if [ "$al" = "team" ]; then
    exec python3 "$NA/collect_all_matching.py" --population "$pop" --protocol "$proto" \
      --repeat "$rep" --workers "$W" --out "$NR/raw/${pop}__${proto}__team__r${rep}.jsonl" \
      >> "$NR/logs/phase_a.log" 2>&1
  fi
  exec python3 "$NA/collect_all_matching.py" --population "$pop" --protocol "$proto" \
    --alias "$al" --repeat "$rep" --workers "$W" \
    --out "$NR/raw/${pop}__${proto}__${al}__r${rep}.jsonl" >> "$NR/logs/phase_a.log" 2>&1
fi

LIST="$NR/logs/shards_phase_a.txt"; : > "$LIST"
# populations smallest first and protocols cheapest first, so complete cells appear early
for pop in wdc_products geographic_settlements musicbrainz20k; do
  for proto in ford madm2_objective madm2_naive du_mad bccs liang_mad dmad dov_vote; do
    if [ "$proto" = "ford" ]; then
      for rep in 0 1 2; do echo "$pop $proto team $rep" >> "$LIST"; done
      continue
    fi
    for rep in 0 1 2; do
      case $rep in
        0) A=llama3-8b-g1; B=qwen25-7b-g2 ;;
        1) A=llama3-8b-g3; B=qwen25-7b-g0 ;;
        2) A=llama3-8b-g1; B=qwen25-7b-g0 ;;
      esac
      echo "$pop $proto $A $rep" >> "$LIST"
      echo "$pop $proto $B $rep" >> "$LIST"
    done
  done
done
echo "=== PHASE A START $(date -u +%H:%M) : $(grep -c . "$LIST") shards, pool $P, workers $W ===" \
  >> "$NR/logs/phase_a.log"
xargs -P "$P" -L 1 sh "$NA/run_phase_a.sh" --shard < "$LIST"
echo "=== PHASE A COMPLETE $(date -u +%H:%M) ===" >> "$NR/logs/phase_a.log"
