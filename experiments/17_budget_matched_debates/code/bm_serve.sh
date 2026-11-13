#!/bin/bash
# Serving for this collection: Llama-3-8B-Instruct and Qwen2.5-7B-Instruct on vLLM 0.30.0
# (image d076d77d8dad), --max-model-len 8192 and --max-num-seqs 512. Only containers named bm-* are
# created or removed here.
#   bash bm_serve.sh up   <l8|q7> <gpu> <port>
#   bash bm_serve.sh down <l8|q7> <gpu>
#   bash bm_serve.sh status
set -u
: "${HF_HOME:=$HOME/.cache/huggingface}"
OUTD=${BM_OUT:-${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/experiments/17_budget_matched_debates/data}
LOG=$OUTD/logs/serve.log
mkdir -p $OUTD/logs
ts() { date -u +%FT%TZ; }
IMG=${IMG:-d076d77d8dad}
UTIL=${UTIL:-0.39}

up() {
  local m=$1 g=$2 port=$3 model served
  case $m in
    l8) model=meta-llama/Meta-Llama-3-8B-Instruct; served=llama3-8b-instruct ;;
    q7) model=Qwen/Qwen2.5-7B-Instruct; served=qwen2.5-7b-instruct ;;
    *) echo "unknown $m"; return 2 ;;
  esac
  local name=bm-$m-g$g
  ss -ltn | grep -q ":$port " && { echo "$(ts) ABORT: port $port busy" | tee -a $LOG; return 2; }
  local cmd=(docker run -d --name $name --gpus "\"device=$g\"" --ipc=private --shm-size=8g --restart=no
             -v $HF_HOME:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -p 127.0.0.1:$port:8000 $IMG
             --model $model --served-model-name $served
             --gpu-memory-utilization $UTIL --max-model-len 8192 --max-num-seqs 512)
  echo "$(ts) LAUNCH ${cmd[*]}" >> $LOG
  "${cmd[@]}" >> $LOG 2>&1 || { echo "$(ts) ABORT: docker run $name" | tee -a $LOG; return 2; }
  for k in $(seq 1 90); do
    curl -sf http://127.0.0.1:$port/v1/models >/dev/null && break
    docker ps --format '{{.Names}}' | grep -q "^$name$" || { echo "$(ts) ABORT: $name exited" | tee -a $LOG; return 2; }
    sleep 10
  done
  curl -sf http://127.0.0.1:$port/v1/models >/dev/null || { echo "$(ts) ABORT: $name not ready" | tee -a $LOG; return 2; }
  echo "$(ts) ready $name GPU$g port $port" | tee -a $LOG
}

down() {
  local name=bm-$1-g$2
  docker logs $name > $OUTD/logs/vllm_$name.log 2>&1
  docker rm -f $name >/dev/null 2>&1 && echo "$(ts) removed $name" | tee -a $LOG
}

case ${1:-status} in
  up) up $2 $3 $4 ;;
  down) down $2 $3 ;;
  status) docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' | grep '^bm-' ;;
esac
