#!/bin/bash
: "${HF_HOME:=$HOME/.cache}"
: "${SPWR_WORKDIR:?set SPWR_WORKDIR to the work directory on the GPU host}"
# Host-side run: serve Qwen2.5-7B on GPU2, pilot + gate, main, then remove the container.
# The container is removed on every exit path (trap), so no idle server is left behind.
cd ${SPWR_WORKDIR}
NAME=cis2-spwr-q25
GPU=2
PORT=8051
RESERVE=28754
IMG="vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
LOG=logs/chain.log
ts() { date -u +%FT%TZ; }
cleanup() { docker rm -f $NAME >/dev/null 2>&1 && echo "$(ts) removed $NAME" >> $LOG; }
trap cleanup EXIT

free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "$(ts) GPU$GPU free ${free} MiB before start" >> $LOG
if [ "$free" -lt $((47500 + RESERVE)) ]; then echo "$(ts) ABORT: not enough free memory" >> $LOG; exit 2; fi

docker run -d --name $NAME --gpus device=$GPU -p $PORT:8000 \
  -v ${HF_HOME}/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 $IMG \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name qwen2.5-7b-instruct --gpu-memory-utilization 0.33 \
  --max-model-len 8192 --max-num-seqs 128 >> $LOG 2>&1 || { echo "$(ts) docker run failed" >> $LOG; exit 1; }
echo "$(ts) started $NAME" >> $LOG

python3 -B code/run_recipient.py --stage pilot --port $PORT >> logs/pilot.log 2>&1
rc=$?
echo "$(ts) pilot rc=$rc" >> $LOG
nvidia-smi -i $GPU --query-gpu=memory.used,memory.free --format=csv,noheader >> $LOG
[ $rc -ne 0 ] && { echo "$(ts) STOP: pilot gate failed" >> $LOG; exit 3; }

python3 -B code/run_recipient.py --stage main --port $PORT >> logs/main.log 2>&1
rc=$?
echo "$(ts) main rc=$rc" >> $LOG
python3 -B code/run_recipient.py --stage main --port $PORT >> logs/main.log 2>&1   # resume pass, no repeats
echo "$(ts) resume pass rc=$?" >> $LOG
sha256sum raw/*.jsonl > logs/raw.sha256
echo "$(ts) done" >> $LOG
