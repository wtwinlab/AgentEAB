#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
R="$PROJECT_ROOT/code"
OUT="$PROJECT_ROOT/results/backbone_matrix"
cd "$R" || exit 1

PROTO="${1:?need dumad or madm2obj}"; shift
[ $# -gt 0 ] || { echo "need at least one benchmark"; exit 1; }

if ! PYTHONPATH="$R" python -m consensus_delivery.tests.test_roster_paths_match_originals \
      >"$DRIFTLOG" 2>&1; then
  echo "REFUSING to collect: the roster path no longer matches the original."
  tail -20 "$DRIFTLOG"
  exit 1
fi

declare -a ROT=("llama3-8b,qwen25-7b,mistral-7b" "qwen25-7b,mistral-7b,llama3-8b" \
                "mistral-7b,llama3-8b,qwen25-7b")

for b in "$@"; do
  for i in 0 1 2; do
    f="$OUT/${b}100_${PROTO}_LQM_rot${i}.json"
    [ -f "$f" ] && { echo "skip ${b} rot${i} (already collected)"; continue; }
    lockd="$OUT/.inflight-${b}100_${PROTO}_LQM_rot${i}"
    if ! mkdir "$lockd" 2>/dev/null; then
      echo "skip ${b} rot${i} (another stream holds $lockd)"; continue
    fi
    trap 'rmdir "$lockd" 2>/dev/null' EXIT INT TERM
    echo "=== ${PROTO} ${b} rotation ${i}: ${ROT[$i]}"
    case "$PROTO" in
      dumad)
        PYTHONPATH="$R" python -m consensus_delivery.adapters.du_mad_roster \
          --block "../data/frozen/${b}_block02.json" --roster "${ROT[$i]}" \
          --rotation "$i" --out "$f" --limit 100 --max-tokens 1024 ;;
      madm2obj)
        PYTHONPATH="$R" python -m consensus_delivery.adapters.mad_m2_roster \
          --block "../data/frozen/${b}_block02.json" --roster "${ROT[$i]}" \
          --prune objective --rotation "$i" --out "$f" --limit 100 \
          --temperature 1.0 --top-p 1.0 --max-tokens 1024 ;;
      *) echo "unknown protocol ${PROTO}"; rmdir "$lockd" 2>/dev/null; exit 1 ;;
    esac
    rmdir "$lockd" 2>/dev/null
    trap - EXIT INT TERM
  done
done
echo "=== ${PROTO} subset $* finished"
