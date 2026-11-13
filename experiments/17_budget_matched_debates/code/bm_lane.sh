#!/bin/bash
# bm_lane.sh <g> <n>   lane g of n: question shard g/n on its own Llama-3-8B and Qwen2.5-7B servers
#   (ports BM_PORT_L8, BM_PORT_Q7). Stages: the 3x2 debate (2 repeats), the 20-draft conditions
#   (1 repeat), 3 and 4 rounds (2 repeats), 5 agents (2 repeats). A stage is retried up to three
#   times; the collector resumes from the records on disk.
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
