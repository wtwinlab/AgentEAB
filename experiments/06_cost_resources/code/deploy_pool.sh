#!/bin/sh
# Model pool for the data-engineering experiments. ThriftLLM's premise is choosing among models of
# different cost and quality, so the pool spans 7B to 72B rather than six models of one size.
# All weights are already on the host. GPU 0 carries another user's allocation and is never
# addressed; ccf_*, monitoring-*, pg-agentmem, n8n, audit-env and watchtower are left alone.
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
run de-qwen7b    1 9001 Qwen/Qwen2.5-7B-Instruct            qwen2.5-7b     0.28
run de-llama8b   1 9002 meta-llama/Meta-Llama-3-8B-Instruct llama-3-8b     0.28
run de-mistral7b 1 9003 mistralai/Mistral-7B-Instruct-v0.3  mistral-7b     0.28
run de-qwen14b   2 9004 Qwen/Qwen2.5-14B-Instruct-AWQ       qwen2.5-14b    0.30
run de-qwen32b   2 9005 Qwen/Qwen2.5-32B-Instruct-AWQ       qwen2.5-32b    0.45
run de-qwen72b   3 9006 Qwen/Qwen2.5-72B-Instruct-AWQ       qwen2.5-72b    0.85
