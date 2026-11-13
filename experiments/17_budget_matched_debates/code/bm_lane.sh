#!/bin/bash
set -u
cd $(dirname $0)
g=$1; n=$2
: "${BM_PORT_L8:?set BM_PORT_L8}" "${BM_PORT_Q7:?set BM_PORT_Q7}"
export BM_UNIT_SHARD=$g/$n BM_LANE_TAG=g$g
W=${W:-256}
st() { local tag=$1 arms=$2 rep=$3; for attempt in 1 2 3; do
  BM_ARMS=$arms BM_REPEATS=$rep bash bm_stage.sh ${tag}_g$g ext_collect --run --benchmark sql_bird_v2 --workers $W && break; done; }
st full FULL 2
st drafts DRAFTS20,T07_DRAFTS20 1
st rounds3 R3_FULL 2
st rounds4 R4_FULL 2
st agents5 A5_FULL 2
