#!/bin/bash
# Start or remove one local vLLM server for this collection.
#   bash ao_serve.sh up   <l8|q7> <gpu> <port>
#   bash ao_serve.sh down <l8|q7> <gpu>
# Image vLLM 0.30.0 (vllm/vllm-openai, id d076d77d8dad), checkpoints from the Hugging Face cache (HF_HOME),
# --max-model-len 8192, --max-num-seqs 512, the served names the collector expects.
set -u
IMG=${AO_IMAGE:-d076d77d8dad}
HF=${HF_HOME:?set HF_HOME to the Hugging Face cache}
UTIL=${UTIL:-0.39}
up() {
  local m=$1 g=$2 port=$3 model served
  case $m in
    l8) model=meta-llama/Meta-Llama-3-8B-Instruct; served=llama3-8b-instruct ;;
    q7) model=Qwen/Qwen2.5-7B-Instruct; served=qwen2.5-7b-instruct ;;
    *) echo "unknown $m"; return 2 ;;
  esac
  docker run -d --name ao-$m-g$g --gpus "\"device=$g\"" --ipc=private --shm-size=8g --restart=no \
    -v $HF:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -p 127.0.0.1:$port:8000 $IMG \
    --model $model --served-model-name $served \
    --gpu-memory-utilization $UTIL --max-model-len 8192 --max-num-seqs 512
  for k in $(seq 1 90); do
    curl -sf http://127.0.0.1:$port/v1/models >/dev/null && { echo "ready ao-$m-g$g on port $port"; return 0; }
    sleep 10
  done
  echo "ao-$m-g$g not ready"; return 2
}
down() { docker rm -f ao-$1-g$2; }
case ${1:-} in
  up) up $2 $3 $4 ;;
  down) down $2 $3 ;;
  *) echo "usage: ao_serve.sh up <l8|q7> <gpu> <port> | down <l8|q7> <gpu>"; exit 2 ;;
esac
