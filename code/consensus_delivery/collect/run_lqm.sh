#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
R="$PROJECT_ROOT/code"
OUT="$PROJECT_ROOT/results/backbone_matrix"
cd "$R" || exit 1
declare -a ROT=("llama3-8b,qwen25-7b,mistral-7b" "qwen25-7b,mistral-7b,llama3-8b" "mistral-7b,llama3-8b,qwen25-7b")
for b in gsm8k strategyqa mmlu_pro gpqa squad_v2; do
  for i in 0 1 2; do
    f="$OUT/${b}100_dumad_LQM_rot${i}.json"
    [ -f "$f" ] && { echo "skip ${b} rot${i} (collected)"; continue; }
    echo "=== ${b} rotation ${i}: ${ROT[$i]}"
    PYTHONPATH="$R" python -m consensus_delivery.adapters.du_mad_roster \
      --block "../data/frozen/${b}_block02.json" --roster "${ROT[$i]}" \
      --rotation "$i" --out "$f" --limit 100 --max-tokens 1024
  done
done
echo "=== Phase B du_mad finished"
