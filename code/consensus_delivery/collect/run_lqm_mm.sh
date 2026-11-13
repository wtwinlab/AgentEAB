#!/usr/bin/env bash
# Phase B, second protocol: MAD-M2 objective-prune with the heterogeneous LQM roster.
# SQuAD 2.0 stays excluded, carried over from BATCH2_FREEZE §3.
#
# Cells whose output file is named `*.affected-consensus-rule.*` are not skipped by the `-f` test
# below; they are collected with the released `if_reach_consensus`.
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
R="$PROJECT_ROOT/code"
OUT="$PROJECT_ROOT/results/backbone_matrix"
cd "$R" || exit 1

# The drift that cost two cells is checked before any request is sent, not after.
if ! PYTHONPATH="$R" python -m consensus_delivery.tests.test_roster_paths_match_originals \
      >"$DRIFTLOG" 2>&1; then
  echo "REFUSING to collect: the roster path no longer matches the original."
  tail -20 "$DRIFTLOG"
  exit 1
fi
echo "roster/original equivalence checked; collecting"
declare -a ROT=("llama3-8b,qwen25-7b,mistral-7b" "qwen25-7b,mistral-7b,llama3-8b" "mistral-7b,llama3-8b,qwen25-7b")
for b in gsm8k strategyqa mmlu_pro gpqa; do
  for i in 0 1 2; do
    f="$OUT/${b}100_madm2obj_LQM_rot${i}.json"
    [ -f "$f" ] && { echo "skip ${b} rot${i}"; continue; }
    echo "=== ${b} rotation ${i}: ${ROT[$i]}"
    PYTHONPATH="$R" python -m consensus_delivery.adapters.mad_m2_roster \
      --block "../data/frozen/${b}_block02.json" --roster "${ROT[$i]}" \
      --prune objective --rotation "$i" --out "$f" --limit 100 \
      --temperature 1.0 --top-p 1.0 --max-tokens 1024
  done
done
echo "=== Phase B madm2 finished"
