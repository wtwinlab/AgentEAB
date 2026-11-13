#!/bin/bash
: "${HF_HOME:=$HOME/.cache}"
# Serving for the strong-backbone BIRD run. Only containers named cis2-sb-* are
# ever created or removed here. Usage:
#   bash sb_serve.sh up q3|l70        # start one backbone (data-parallel over its two cards)
#   bash sb_serve.sh down q3|l70      # remove that backbone's container
#   bash sb_serve.sh status
set -u
OUTD=${OUTD:-${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/data/records/backbone_scale_cross_family}
LOG=$OUTD/logs/serve.log
mkdir -p $OUTD/logs
ts() { date -u +%FT%TZ; }
W=/weights
IMG=vllm/vllm-openai:v0.32.0
HUB=${HF_HOME}/huggingface/hub
Q3="$W/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
GO="/weights_home/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
Q36="$W/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
L70="$W/models--casperhansen--llama-3.3-70b-instruct-awq/snapshots/64d255621f40b42adaf6d1f32a47e1d4534c0f14"
Q72="$W/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
Q32A="$W/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
Q14A="$W/models--Qwen--Qwen2.5-14B-Instruct-AWQ/snapshots/539535859b135b0244c91f3e59816150c8056698"
Q38="$W/models--Qwen--Qwen3.8-27B/snapshots/8c0b3d30f3ae96f5c7f774de704bf0ada83e422a"
NEMO="$W/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/snapshots/e09a4cedb1787db1760375ee5539f222cae0c672"
GLM="$W/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
MIS31="$W/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a"
GO20="$W/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
DP=${DP:-2}; TP=${TP:-1}; MNS=${MNS:-128}
RESERVE=28754
TOTAL=143145
UTIL=${UTIL:-0.75}

spec() {  # name gpus port model served extra
  case $1 in
    q3)  echo "cis2-sb-q3 ${GPUS:-0,1} 8231 $Q3 qwen3-32b" ;;
    l70) echo "cis2-sb-l70 ${GPUS:-2,3} 8232 $L70 llama3.3-70b-awq" ;;
    go)  echo "cis2-sb-go ${GPUS:-0,1} 8233 $GO gpt-oss-120b" ;;
    l70b) echo "cis2-sb-l70b ${GPUS:-0} 8253 $L70 llama3.3-70b-awq" ;;
    gem) echo "cis2-sb-gem ${GPUS:-0} 8250 $W/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52 gemma-4-26b-a4b-it" ;;
    agtb) echo "cis2-sb-agtb ${GPUS:-3} 8254 $W/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/$(ls $HUB/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/ | head -1) agentar-sql-32b" ;;
    agtc) echo "cis2-sb-agtc ${GPUS:-0} 8255 $W/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/$(ls $HUB/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/ | head -1) agentar-sql-32b" ;;
    agtd) echo "cis2-sb-agtd ${GPUS:-1} 8256 $W/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/$(ls $HUB/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/ | head -1) agentar-sql-32b" ;;
    agt) echo "cis2-sb-agt ${GPUS:-2} 8251 $W/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/$(ls $HUB/models--antgroup--Agentar-Scale-SQL-Generation-32B/snapshots/ | head -1) agentar-sql-32b" ;;
    mmx) echo "cis2-sb-mmx ${GPUS:-2,3} 8252 $W/models--QuantTrio--MiniMax-M2.5-AWQ/snapshots/bbe738792c897da97ae0dfe5a47bd28a3a1ea10e minimax-m2.5-awq" ;;
    q38c) echo "cis2-sb-q38c ${GPUS:-3} 8249 $Q38 qwen3.8-27b" ;;
    q38b) echo "cis2-sb-q38b ${GPUS:-0} 8248 $Q38 qwen3.8-27b" ;;
    nemob) echo "cis2-sb-nemob ${GPUS:-0,1} 8247 $NEMO nemotron-3.5-bolt" ;;
    q3b) echo "cis2-sb-q3b ${GPUS:-3} 8246 $Q3 qwen3-32b" ;;
    q36) echo "cis2-sb-q36 ${GPUS:-2,3} 8234 $Q36 qwen3.6-35b-a3b" ;;
    q72) echo "cis2-sb-q72 ${GPUS:-2,3} 8235 $Q72 qwen2.5-72b-awq" ;;
    q32a) echo "cis2-sb-q32a ${GPUS:-0,1} 8236 $Q32A qwen2.5-32b-awq" ;;
    q14a) echo "cis2-sb-q14a ${GPUS:-0,1} 8237 $Q14A qwen2.5-14b-awq" ;;
    q38) echo "cis2-sb-q38 ${GPUS:-0} 8238 $Q38 qwen3.8-27b" ;;
    nemo) echo "cis2-sb-nemo ${GPUS:-2,3} 8239 $NEMO nemotron-3.5-bolt" ;;
    glm) echo "cis2-sb-glm ${GPUS:-0} 8240 $GLM glm-4.7-flash" ;;
    mis31) echo "cis2-sb-mis31 ${GPUS:-0} 8241 $MIS31 mistral-small-3.1-24b" ;;
    q7x) echo "cis2-sb-q7x ${GPUS:-0} 8245 /weights_home/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 qwen2.5-7b-instruct" ;;
    glmt) echo "cis2-sb-glmt ${GPUS:-0} 8244 $GLM glm-4.7-flash-think" ;;
    mis32) echo "cis2-sb-mis32 ${GPUS:-0} 8243 $W/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/$(ls $HUB/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/ | head -1) mistral-small-3.2-24b" ;;
    go20) echo "cis2-sb-go20 ${GPUS:-0} 8242 $GO20 gpt-oss-20b" ;;
    *) echo "unknown backbone $1" >&2; exit 2 ;;
  esac
}

up() {
  read -r name gpus port model served <<< "$(spec $1)"
  local extra=""
  case $1 in q3|q3b|q36|q38|q38b|q38c|nemo|nemob) extra='{"enable_thinking": false}' ;;
             glm) extra='{"enable_thinking": false, "clear_thinking": true}' ;; esac
  for g in ${gpus//,/ }; do
    local free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    local need=$(python3 -c "print(int($UTIL*$TOTAL)+$RESERVE)")
    echo "$(ts) GPU$g free $free need $need for $name" | tee -a $LOG
    [ "$free" -lt "$need" ] && { echo "$(ts) ABORT: GPU$g below need for $name" | tee -a $LOG; return 2; }
  done
  ss -ltn | grep -q ":$port " && { echo "$(ts) ABORT: port $port busy" | tee -a $LOG; return 2; }
  local denv=()
  case $1 in nemo|nemob) denv=(-e VLLM_USE_V2_MODEL_RUNNER=1) ;; esac
  local cmd=(docker run -d "${denv[@]}" --name $name --gpus "\"device=$gpus\"" --ipc=host -p 127.0.0.1:$port:8000
             -v $HUB:/weights:ro -v ${HF_HUB}:/weights_home:ro $IMG --model $model --served-model-name $served
             --max-model-len 16384 --max-num-seqs $MNS --gpu-memory-utilization $UTIL
             --data-parallel-size $DP --tensor-parallel-size $TP --enable-prefix-caching)
  [ -n "$extra" ] && cmd+=(--default-chat-template-kwargs "$extra")
  # model-specific serving switches: GLM peer-text a reasoning trace ~1% of calls even with
  # enable_thinking=false, so its parser moves it out of `content`; Mistral-Small-3.2 ships no HF
  # image-processor config, so it is served in its native mistral format
  case $1 in
    glm|glmt) cmd+=(--reasoning-parser glm47) ;;
    mis32) cmd+=(--tokenizer-mode mistral --config-format mistral --load-format mistral) ;;
    nemo|nemob) cmd+=(--trust-remote-code --attention-backend FLASHINFER --mamba-backend flashinfer --mamba-cache-mode align) ;;
    # ^ model-card recipe flags (https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16); env below
    mmx) cmd+=(--reasoning-parser minimax_m2 --trust-remote-code) ;;
  esac
  echo "$(ts) LAUNCH ${cmd[*]}" >> $LOG
  "${cmd[@]}" >> $LOG 2>&1 || { echo "$(ts) ABORT: docker run $name" | tee -a $LOG; return 2; }
  echo "$(ts) started $name GPUs $gpus port $port" | tee -a $LOG
  for k in $(seq 1 120); do
    curl -sf http://127.0.0.1:$port/v1/models >/dev/null && break
    docker ps --format '{{.Names}}' | grep -q "^$name$" || { echo "$(ts) ABORT: $name exited" | tee -a $LOG; docker logs --tail 40 $name >> $LOG 2>&1; return 2; }
    sleep 10
  done
  curl -sf http://127.0.0.1:$port/v1/models >/dev/null || { echo "$(ts) ABORT: $name not ready" | tee -a $LOG; return 2; }
  echo "$(ts) ready $name" | tee -a $LOG
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a $LOG
}

down() {
  read -r name gpus port model served <<< "$(spec $1)"
  case $name in cis2-sb-*) ;; *) echo "refusing to remove $name"; return 2 ;; esac
  docker logs $name > $OUTD/logs/vllm_$name.log 2>&1; echo "$(ts) saved docker logs of $name ($(wc -l < $OUTD/logs/vllm_$name.log) lines)" | tee -a $LOG
  docker rm -f $name >/dev/null 2>&1 && echo "$(ts) removed $name (GPUs $gpus, port $port)" | tee -a $LOG
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a $LOG
}

status() {
  docker ps -a --format '{{.Names}}\t{{.Status}}' | grep '^cis2-sb-'
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
}

case ${1:-status} in
  up) up $2 ;;
  down) down $2 ;;
  status) status ;;
esac
