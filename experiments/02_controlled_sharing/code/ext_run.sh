#!/bin/sh
set -u
R=${R:?set R to the collection tree root on the inference host}
A="$R/experiments/02_controlled_sharing/code"
PY=${PY:-python3}
OUTD="$R/results/benchmark_extension_2026-10-20"
LOG="$OUTD/logs/RUN.log"
export PYTHONPATH="$A:$R/code:$R/experiments/06_cost_resources/code"
GPQA_WORKERS=${GPQA_WORKERS:-448}
SQL_WORKERS=${SQL_WORKERS:-320}

mkdir -p "$OUTD/logs"
cd "$A" || exit 1
ulimit -n 65536 2>/dev/null || true

note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ext_run] $*" | tee -a "$LOG"; }

stopped() {
  st=$("$PY" - "$R" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "results/benchmark_extension_2026-10-20/CONTROL.json"
try:
    print(json.loads(p.read_text()).get("state", "run"))
except Exception:
    print("run")
PYEOF
)
  [ "$st" = "stop" ]
}

note "=== start; gate is re-read from disk, nothing is dispatched until it opens ==="

note "stage 1: GPQA smoke (2 units per cell, 19 cells), waiting for the gate"
"$PY" ext_collect.py --run --wait-gate --autoserve --benchmark gpqa --smoke 2 \
      --workers 24 >> "$LOG" 2>&1
rc=$?
note "stage 1 exit $rc"
if [ "$rc" = "2" ]; then note "stop flag set -- halting, nothing further dispatched"; exit 2; fi
if [ "$rc" != "0" ]; then note "stage 1 failed (exit $rc) -- NOT proceeding to the full run"; exit "$rc"; fi
"$PY" ext_analyze.py gpqa >> "$LOG" 2>&1 || note "smoke report failed (non-fatal)"

if stopped; then note "stop flag set -- halting before the GPQA full run"; exit 2; fi
note "stage 2: GPQA full, $GPQA_WORKERS workers"
"$PY" ext_collect.py --run --benchmark gpqa --workers "$GPQA_WORKERS" >> "$LOG" 2>&1
rc=$?
note "stage 2 exit $rc"
if [ "$rc" = "2" ]; then note "stop flag set -- halting"; exit 2; fi
"$PY" ext_analyze.py gpqa >> "$LOG" 2>&1
note "stage 2 report written"
"$PY" ext_collect.py --manifest >> "$LOG" 2>&1

if stopped; then note "stop flag set -- halting before SQL"; exit 2; fi
note "stage 3: SQL smoke (2 units per cell) -- du_mad / ford / moa SQL prompts have never been sent"
"$PY" ext_collect.py --run --benchmark sql_bird_v2 --smoke 2 --workers 24 >> "$LOG" 2>&1
rc=$?
note "stage 3 exit $rc"
if [ "$rc" = "2" ]; then note "stop flag set -- halting"; exit 2; fi
if [ "$rc" != "0" ]; then note "SQL smoke failed (exit $rc) -- NOT spending 12,400 unit-runs"; exit "$rc"; fi
"$PY" ext_eval_sql.py --allow-partial >> "$LOG" 2>&1 || note "smoke SQL eval failed (non-fatal)"
"$PY" ext_analyze.py sql_bird_v2 >> "$LOG" 2>&1 || note "smoke SQL report failed (non-fatal)"
note "stage 3 done -- inspect the smoke report before the full SQL run is trusted"

if stopped; then note "stop flag set -- halting before the SQL full run"; exit 2; fi
note "stage 4: SQL full, $SQL_WORKERS workers"
"$PY" ext_collect.py --run --benchmark sql_bird_v2 --workers "$SQL_WORKERS" >> "$LOG" 2>&1
rc=$?
note "stage 4 exit $rc"
if [ "$rc" = "2" ]; then note "stop flag set -- halting"; exit 2; fi
"$PY" ext_eval_sql.py >> "$LOG" 2>&1 || "$PY" ext_eval_sql.py --allow-partial >> "$LOG" 2>&1
"$PY" ext_analyze.py --all >> "$LOG" 2>&1
"$PY" ext_collect.py --manifest >> "$LOG" 2>&1
note "=== all stages done ==="
