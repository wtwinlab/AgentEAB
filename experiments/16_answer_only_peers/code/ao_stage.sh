#!/bin/bash
# ao_stage.sh <tag> <ao_run args...>   one collector stage; env AO_* selects conditions, lane, ports and shard
set -u
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export PAM_ART=${PAM_ART:-$R/data/records/protocol_all_matching}
export PYTHONPATH="$PAM_ART:$R/code:$R/experiments/02_controlled_sharing/code"
OUTD=${AO_OUT:-$R/experiments/16_answer_only_peers/data}
mkdir -p $OUTD/logs
LOG=$OUTD/logs/$1.log; shift
ulimit -n 65536 2>/dev/null || true
cd $(dirname $0)
echo "$(date -u +%FT%TZ) START $* lane=${AO_LANE_TAG:-} arms=${AO_ARMS:-} rep=${AO_REPEATS:-2} shard=${AO_UNIT_SHARD:-}" >> $LOG
${PYTHON:-python3} ao_run.py "$@" >> $LOG 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc" >> $LOG
exit $rc
