#!/bin/bash
# ao_lane.sh <i> <n>   one lane: question shard i of n, all four conditions, two repeats, up to three attempts
#   (an attempt resumes from the records already written). AO_PORT_L8 / AO_PORT_Q7 name this lane's servers.
set -u
cd $(dirname $0)
: "${AO_PORT_L8:?set AO_PORT_L8}" "${AO_PORT_Q7:?set AO_PORT_Q7}"
export AO_UNIT_SHARD=$1/$2 AO_LANE_TAG=g$1
for attempt in 1 2 3; do
  AO_ARMS=FULL,NOPEERC,NOPEER,ANSWER_ONLY AO_REPEATS=2 \
    bash ao_stage.sh ao_g$1 ext_collect --run --benchmark sql_bird_v2 --workers ${W:-256} && break
done
