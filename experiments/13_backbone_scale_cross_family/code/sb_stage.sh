#!/bin/bash
# sb_stage.sh <tag> <ext_collect args...>  -- one detached collector stage with wall-clock stamps
set -u
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export PAM_ART=$R/data/records/protocol_all_matching
export PYTHONPATH="$PAM_ART:$R/code:$R/experiments/02_controlled_sharing/code"
OUTD=$R/data/records/backbone_scale_cross_family
LOG=$OUTD/logs/$1.log; shift
ulimit -n 65536 2>/dev/null || true
cd $(dirname $0)
echo "$(date -u +%FT%TZ) START $* SB_MAX_TOKENS=${SB_MAX_TOKENS:-1024} epoch=$(date +%s)" >> $LOG
${PYTHON:-python3} sb_run.py "$@" >> $LOG 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc epoch=$(date +%s)" >> $LOG
