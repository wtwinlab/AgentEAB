#!/usr/bin/env bash
# Phase B on a named benchmark subset, so two streams can run without racing.
#
#   run_lqm_subset.sh <dumad|madm2obj> <benchmark> [benchmark ...]
#
# The existing `run_lqm.sh` / `run_lqm_mm.sh` walk their benchmark list in a fixed order and skip a
# cell whose output file already exists. Launching a second copy of the same wrapper would have both
# see a cell as absent and start it twice, so instead this takes the benchmarks to cover and is
# pointed at the **end** of the other stream's order. The two meet rather than overlap, and the
# `-f` test catches the handover.
#
# Rotations and settings are identical to the two single-protocol wrappers; this file exists for
# scheduling, not for a different experiment.
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
R="$PROJECT_ROOT/code"
OUT="$PROJECT_ROOT/results/backbone_matrix"
cd "$R" || exit 1

PROTO="${1:?need dumad or madm2obj}"; shift
[ $# -gt 0 ] || { echo "need at least one benchmark"; exit 1; }

# Same gate as run_lqm_mm.sh: the roster/original equivalence is checked before any request.
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
    # `run_block` refuses to overwrite, but only at START, and the output file appears only at the
    # END. So two streams that both saw a cell as absent would both run it for twenty minutes and
    # the second would replace the first, pairing one run's records with another's manifest. The
    # `-f` test above cannot close that window; this lock can. `mkdir` is atomic on every
    # filesystem this runs on, which `[ -d ]` followed by `mkdir` would not be.
    lockd="$OUT/.inflight-${b}100_${PROTO}_LQM_rot${i}"
    if ! mkdir "$lockd" 2>/dev/null; then
      echo "skip ${b} rot${i} (another stream holds $lockd)"; continue
    fi
    # Released on normal exit, on error, and on SIGTERM/SIGINT, so a killed stream does not leave a
    # cell permanently unclaimable.
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
