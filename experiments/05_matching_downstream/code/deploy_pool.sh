#!/bin/sh
set -e
HOST=${REMOTE_SSH_TARGET:-user@inference-host}
IMG="vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
LC=${HFCACHE:-$HOME/.cache/huggingface}

run() {  # name gpu port model served util
  ssh -o BatchMode=yes "$HOST" "docker run -d --name $1 --gpus device=$2 -p $3:8000 \
    -v $LC:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 $IMG \
    --model $4 --served-model-name $5 --gpu-memory-utilization $6 \
    --max-model-len 4096 --max-num-seqs 64" >/dev/null \
    && echo "  started $1 (gpu $2, port $3)" || echo "  FAILED $1"
}

run dsr-qwen7b    1 9201 Qwen/Qwen2.5-7B-Instruct            qwen2.5-7b     0.30
run dsr-llama8b   4 9202 meta-llama/Meta-Llama-3-8B-Instruct llama-3-8b     0.30
run dsr-mistral7b 5 9203 mistralai/Mistral-7B-Instruct-v0.3  mistral-7b     0.30
run dsr-qwen14b   2 9204 Qwen/Qwen2.5-14B-Instruct-AWQ       qwen2.5-14b    0.30
run dsr-qwen32b   2 9205 Qwen/Qwen2.5-32B-Instruct-AWQ       qwen2.5-32b    0.45
run dsr-qwen72b   3 9206 Qwen/Qwen2.5-72B-Instruct-AWQ       qwen2.5-72b    0.85
