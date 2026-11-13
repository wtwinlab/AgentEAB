#!/bin/sh
# Serving pool for the downstream-replication round. Distinct `dsr-` prefix and ports 9201-9206 so
# nothing collides with another user's services or with this project's earlier `de-*` round.
#
# GPU 0 carries another user's allocation and is never addressed. ccf_*, monitoring-*, pg-agentmem,
# audit-env, n8n and watchtower are left alone. No prune, ever.
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

# One model per GPU. Packing three onto GPU 1 at 0.28 each failed with
# "Available KV cache memory: -6.18 GiB": vLLM's utilisation fraction is of TOTAL device memory and
# accounts for memory already taken, so three concurrent initialisations race for one budget. This
# is the same failure the earlier data-engineering round hit; one model per GPU is the fix.
run dsr-qwen7b    1 9201 Qwen/Qwen2.5-7B-Instruct            qwen2.5-7b     0.30
run dsr-llama8b   4 9202 meta-llama/Meta-Llama-3-8B-Instruct llama-3-8b     0.30
run dsr-mistral7b 5 9203 mistralai/Mistral-7B-Instruct-v0.3  mistral-7b     0.30
# These three came up on the first attempt and are not redeployed.
run dsr-qwen14b   2 9204 Qwen/Qwen2.5-14B-Instruct-AWQ       qwen2.5-14b    0.30
run dsr-qwen32b   2 9205 Qwen/Qwen2.5-32B-Instruct-AWQ       qwen2.5-32b    0.45
run dsr-qwen72b   3 9206 Qwen/Qwen2.5-72B-Instruct-AWQ       qwen2.5-72b    0.85
