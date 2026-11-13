#!/bin/bash
# bm_stage.sh <tag> <bm_run args...>   one collector stage; env BM_* selects conditions, protocols, lane, ports, shard
set -u
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export PAM_ART=$R/data/records/protocol_all_matching
export PYTHONPATH="$PAM_ART:$R/code:$R/experiments/02_controlled_sharing/code"
OUTD=${BM_OUT:-$R/experiments/17_budget_matched_debates/data}
mkdir -p $OUTD/logs
LOG=$OUTD/logs/$1.log; shift
ulimit -n 65536 2>/dev/null || true
cd $(dirname $0)
echo "$(date -u +%FT%TZ) START $* lane=${BM_LANE_TAG:-} arms=${BM_ARMS:-} protos=${BM_PROTOS:-} bb=${BM_BACKBONES:-} rep=${BM_REPEATS:-2} shard=${BM_UNIT_SHARD:-} epoch=$(date +%s)" >> $LOG
${PYTHON:-python3} bm_run.py "$@" >> $LOG 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc epoch=$(date +%s)" >> $LOG
exit $rc
