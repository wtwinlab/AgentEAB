#!/bin/bash
# cn_stage.sh <tag> <cn_run args...>   env: SB_BACKBONES SB_PROTOS SB_MAX_TOKENS SB_LANE_TAG [SB_*] [CN_*]
set -u
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export PAM_ART=$R/data/records/protocol_all_matching
export PYTHONPATH="$PAM_ART:$R/code:$R/experiments/02_controlled_sharing/code"
OUTD=${CN_OUT:-$R/experiments/14_nopeer_collection/data/sql}
mkdir -p $OUTD/logs
LOG=$OUTD/logs/$1.log; shift
ulimit -n 65536 2>/dev/null || true
cd $(dirname $0)
echo "$(date -u +%FT%TZ) START $* bb=${SB_BACKBONES:-} protos=${SB_PROTOS:-} mt=${SB_MAX_TOKENS:-1024} lane=${SB_LANE_TAG:-} epoch=$(date +%s)" >> $LOG
${PYTHON:-python3} cn_run.py "$@" >> $LOG 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc epoch=$(date +%s)" >> $LOG
exit $rc
