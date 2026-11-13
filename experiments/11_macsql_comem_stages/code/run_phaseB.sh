#!/bin/bash
: "${HF_HOME:=$HOME/.cache}"
# Phase B of the external-stage run: Qwen2.5-72B-AWQ (weights under ${HF_HOME}) with the same
# models, revisions and server arguments as phase A. GPU0: Qwen2.5-72B-AWQ for the three Spider
# shards. GPU g in 1..3: starts when "GPU$g phase A finished" appears; Qwen2.5-72B-AWQ for BIRD
# shard g; GPU3 also Qwen2.5-7B for the ComEM selector. Each server is removed when its work ends;
# gates use the benchmark's own Decomposer signature.
set -u
: "${EXT_WORKDIR:?set EXT_WORKDIR to the MAC-SQL and ComEM checkouts, venvs, run/ inputs}"
: "${HF_HUB:=$HOME/.cache/huggingface/hub}"
E=${EXT_WORKDIR}; F=$E/formal; LOG=$F/logs/run.log; RB=$F/raw_b
mkdir -p $RB
ts() { date -u +%FT%TZ; }
W=/weights
Q72="$W/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
Q7="$W/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
start_server() {  # name gpu port model served util
  local name=$1 g=$2 port=$3 model=$4 served=$5 util=$6
  echo "$(ts) B GPU$g free $(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits) for $name" >> $LOG
  ss -ltn | grep -q ":$port " && { echo "$(ts) B ABORT: port $port busy" >> $LOG; return 2; }
  docker run -d --name $name --gpus device=$g --ipc=host -p 127.0.0.1:$port:8000 -v ${HF_HOME}/huggingface/hub:/weights:ro \
    vllm/vllm-openai:v0.32.0 --model $model --served-model-name $served --max-model-len 32768 --max-num-seqs 96 \
    --gpu-memory-utilization $util >> $LOG 2>&1 || { echo "$(ts) B ABORT: docker run $name" >> $LOG; return 2; }
  echo "$(ts) B started $name GPU$g:$port $served util=$util" >> $LOG
  for k in $(seq 1 180); do
    curl -sf http://127.0.0.1:$port/v1/models >/dev/null && break
    docker ps --format '{{.Names}}' | grep -q "^$name$" || { echo "$(ts) B ABORT: $name exited" >> $LOG; return 2; }
    sleep 10
  done
  curl -sf http://127.0.0.1:$port/v1/models >/dev/null || { echo "$(ts) B ABORT: $name not ready" >> $LOG; return 2; }
  echo "$(ts) B ready $name" >> $LOG; nvidia-smi -i $g --query-gpu=index,memory.used,memory.free --format=csv,noheader >> $LOG
}
stop_server() { docker rm -f $1 >/dev/null 2>&1 && echo "$(ts) B removed $1" >> $LOG; }
proxy() {  # listen upstream logname tag cap
  echo "$4" > $RB/$3.tag
  $E/venv_macsql/bin/python -u $E/run/proxy.py --listen $1 --upstream http://127.0.0.1:$2 --cap $5 --log $RB/$3.jsonl \
    --question-file $RB/$3.tag >> $F/logs/proxyB_$3.log 2>&1 &
  echo $!
}
shard() {  # bench k proxy_port
  local b=$1 k=$2 pp=$3 O=$F/out/macsql_q72_${1}_s${2}
  mkdir -p $O
  local db=./data/dev_databases tj=$E/run/dev_tables.json sig="decompose the question"
  [ $b = spider ] && { db=./data/spider_database; tj=$E/run/spider_tables.json; sig="then generate the corresponding SQL"; }
  python3 -c "import json;json.dump(json.load(open('$F/data/${b}_s${k}.json'))[:1],open('$O/gate_input.json','w'))"
  for inp in $O/gate_input.json $F/data/${b}_s${k}.json; do
    echo "$(ts) M q72 $b s$k run $(basename $inp) (B)" >> $LOG
    ( cd $E/macsql && OPENAI_API_BASE=http://127.0.0.1:$pp/v1 OPENAI_API_KEY=EMPTY MACSQL_MODEL=qwen2.5-72b timeout 36000 \
        $E/venv_macsql/bin/python -u run.py --dataset_name $b --dataset_mode dev --input_file $inp --db_path $db \
        --tables_json_path $tj --output_file $O/output.jsonl --log_file $O/log.txt >> $O/stdout.txt 2>&1 )
    local rc=$?; echo "$(ts) M q72 $b s$k end $(basename $inp) rc=$rc n=$(wc -l < $O/output.jsonl) (B)" >> $LOG
    [ $rc -eq 124 ] && echo "$(ts) M q72 $b s$k SHARD_TIMEOUT (B)" >> $LOG
    if [ $inp = $O/gate_input.json ]; then
      if [ "$(wc -l < $O/output.jsonl)" -ge 1 ] && grep -q "$sig" $O/api_trace.json 2>/dev/null; then
        echo "$(ts) M q72 $b s$k GATE PASS (B)" >> $LOG
      else echo "$(ts) M q72 $b s$k GATE FAIL (B): shard stops" >> $LOG; return 1; fi
    fi
  done
}
gpu0() {
  start_server cis2-fxb-q72-g0 0 8205 $Q72 qwen2.5-72b 0.70 || return 2
  local pids="$(proxy 8206 8205 q72_spider_s1 M:q72:spider:s1 20000) $(proxy 8207 8205 q72_spider_s2 M:q72:spider:s2 20000) $(proxy 8208 8205 q72_spider_s3 M:q72:spider:s3 20000)"
  sleep 3
  shard spider 1 8206 & local a=$!; shard spider 2 8207 & local b=$!; shard spider 3 8208 & local c=$!
  wait $a $b $c; kill $pids 2>/dev/null; stop_server cis2-fxb-q72-g0; echo "$(ts) B GPU0 finished" >> $LOG
}
gpuN() {  # g
  local g=$1 sp=82${1}5 pb=82${1}6 pc=82${1}8 p7=82${1}9
  until grep -q "GPU$g phase A finished" $LOG; do sleep 15; done
  local util=0.75; [ $g = 3 ] && util=0.50
  start_server cis2-fxb-q72-g$g $g $sp $Q72 qwen2.5-72b $util || return 2
  local pids="$(proxy $pb $sp q72_bird_s$g M:q72:bird:s$g 20000)" c=""
  if [ $g = 3 ]; then
    if start_server cis2-fxb-q7-g3 3 $p7 $Q7 qwen2.5-7b 0.20; then
      pids="$pids $(proxy $pc $p7 q7_comem C:q7 8000)"; sleep 3
      O=$F/out/comem_q7; mkdir -p $O/work/models/hf_models
      ln -sfn ${HF_HOME}/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 $O/work/models/hf_models/flan-t5-xl
      ( echo "$(ts) C q7 start (B)" >> $LOG
        cd $O/work && OPENAI_BASE_URL=http://127.0.0.1:$pc/v1 OPENAI_API_KEY=EMPTY CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
          PYTHONPATH=$E/comem timeout 10800 $E/venv_comem/bin/python -u $E/run/run_comem_formal.py $E/comem/data/llm4em \
          $E/run/comem_dev_nolabel.json qwen2.5-7b $O > $O/stdout.txt 2>&1
        echo "$(ts) C q7 end rc=$? (B)" >> $LOG; grep -E "GATE|done" $O/stdout.txt | sed "s/^/C q7 /" >> $LOG
        stop_server cis2-fxb-q7-g3 ) &
      c=$!
    fi
  fi
  sleep 3
  shard bird $g $pb
  [ -n "$c" ] && wait $c
  kill $pids 2>/dev/null; stop_server cis2-fxb-q72-g$g; echo "$(ts) B GPU$g finished" >> $LOG
}
echo "$(ts) B phase B (R3) start" >> $LOG
gpu0 & gpuN 1 & gpuN 2 & gpuN 3 &
wait
until grep -q " done$" $LOG; do sleep 30; done
(cd $E/data/dev_databases && sha256sum */*.sqlite; cd $E/data/spider_database && sha256sum */*.sqlite) > $F/logs/db_after_B.sha256
sha256sum $F/raw/*.jsonl $RB/*.jsonl > $F/logs/raw_all.sha256
for n in $(docker ps -a --format '{{.Names}}' | grep -E '^cis2-fx-(q72|q7)-g'); do docker rm $n >/dev/null && echo "$(ts) removed placeholder $n" >> $LOG; done
echo "$(ts) B all done" >> $LOG
