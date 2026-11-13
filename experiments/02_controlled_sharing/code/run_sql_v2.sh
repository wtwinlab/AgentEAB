#!/bin/sh
set -u
R=${R:-$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)}
PY=${PY:-python3}
NA="$R/experiments/02_controlled_sharing/code"
NR="$R/results/protocol_all_matching_20261018T192619Z"
if [ "${1:-}" = "--shard" ]; then
  exec "$PY" "$NA/collect_sql_v2.py" --population sql_bird_v2 --protocol "$2" --alias "$3" \
    --repeat "$4" --workers "$5" --out "$NR/raw_sql/sql_bird_v2__$2__$3__r$4.jsonl" \
    >> "$NR/logs/sql_v2.log" 2>&1
fi
LIST="$NR/logs/shards_sql_v2.txt"; : > "$LIST"
for spec in "madm2_naive 32" "madm2_objective 32" "bccs 24" "dmad 16"; do
  set -- $spec; proto=$1; w=$2
  for rep in 0 1 2; do
    case $rep in 0) A=llama3-8b-g1; B=qwen25-7b-g2 ;;
                 1) A=llama3-8b-g3; B=qwen25-7b-g0 ;;
                 2) A=llama3-8b-g1; B=qwen25-7b-g1b ;; esac
    echo "$proto $A $rep $w" >> "$LIST"
    echo "$proto $B $rep $w" >> "$LIST"
  done
done
echo "=== SQL v2 START $(date -u +%H:%M) : $(grep -c . "$LIST") shards ===" >> "$NR/logs/sql_v2.log"
xargs -P ${P:-6} -L 1 sh "$NA/run_sql_v2.sh" --shard < "$LIST"
echo "=== SQL v2 COMPLETE $(date -u +%H:%M) ===" >> "$NR/logs/sql_v2.log"
