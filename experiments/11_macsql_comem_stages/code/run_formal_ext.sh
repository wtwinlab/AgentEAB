#!/bin/bash
: "${HF_HOME:=$HOME/.cache}"
# Formal external-stage run. One worker per GPU (1,2,3). Phase A: Qwen3-32B server,
# that GPU's BIRD and Spider MAC-SQL shards (one proxy each); GPU3 also runs ComEM (selector
# Qwen3-32B, ranker on GPU3). Phase B: Qwen2.5-72B-AWQ for the same shards; GPU3 also runs a
# Qwen2.5-7B server for the ComEM selector. Each server is removed when its phase work ends.
set -u
: "${EXT_WORKDIR:?set EXT_WORKDIR to the MAC-SQL and ComEM checkouts, venvs, run/ inputs}"
: "${HF_HUB:=$HOME/.cache/huggingface/hub}"
E=${EXT_WORKDIR}; F=$E/formal; LOG=$F/logs/run.log
mkdir -p $F/logs $F/raw $F/out
ts() { date -u +%FT%TZ; }
W=/weights
Q3="$W/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
Q72="$W/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
Q7="$W/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
start_server() {  # name gpu port model served util maxlen extra
  local name=$1 g=$2 port=$3 model=$4 served=$5 util=$6 ml=$7 extra=$8
  local free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  local need=$(python3 -c "print(int($util*143145)+28754)")
  echo "$(ts) GPU$g free $free need $need for $name" >> $LOG
  [ "$free" -lt "$need" ] && { echo "$(ts) ABORT: GPU$g below need for $name" >> $LOG; return 2; }
  ss -ltn | grep -q ":$port " && { echo "$(ts) ABORT: port $port busy" >> $LOG; return 2; }
  docker run -d --name $name --gpus device=$g --ipc=host -p 127.0.0.1:$port:8000 -v ${HF_HUB}:/weights:ro \
    vllm/vllm-openai:v0.32.0 --model $model --served-model-name $served --max-model-len $ml --max-num-seqs 96 \
    --gpu-memory-utilization $util $extra >> $LOG 2>&1 || { echo "$(ts) ABORT: docker run $name" >> $LOG; return 2; }
  echo "$(ts) started $name GPU$g:$port $served util=$util" >> $LOG
  for k in $(seq 1 180); do curl -sf http://127.0.0.1:$port/v1/models >/dev/null && break; sleep 10; done
  curl -sf http://127.0.0.1:$port/v1/models >/dev/null || { echo "$(ts) ABORT: $name not ready" >> $LOG; return 2; }
  echo "$(ts) ready $name" >> $LOG; nvidia-smi -i $g --query-gpu=index,memory.used,memory.free --format=csv,noheader >> $LOG
}
stop_server() {
  docker rm -f $1 >/dev/null 2>&1 && echo "$(ts) removed $1" >> $LOG
}
proxy() {  # listen upstream log tag cap
  echo "$4" > $F/raw/$4.tag
  $E/venv_macsql/bin/python -u $E/run/proxy.py --listen $1 --upstream http://127.0.0.1:$2 --cap $5 --log $F/raw/$3.jsonl \
    --question-file $F/raw/$4.tag >> $F/logs/proxy_$3.log 2>&1 &
  echo $!
}
shard() {  # model_tag bench k proxy_port
  local mt=$1 b=$2 k=$3 pp=$4 O=$F/out/macsql_${1}_${2}_s${3}
  mkdir -p $O
  local db=./data/dev_databases tj=$E/run/dev_tables.json ds=bird
  [ $b = spider ] && { db=./data/spider_database; tj=$E/run/spider_tables.json; ds=spider; }
  python3 -c "import json;json.dump(json.load(open('$E/formal/data/${b}_s${k}.json'))[:1],open('$O/gate_input.json','w'))"
  for inp in $O/gate_input.json $E/formal/data/${b}_s${k}.json; do
    echo "$(ts) M $mt $b s$k run $(basename $inp)" >> $LOG
    ( cd $E/macsql && OPENAI_API_BASE=http://127.0.0.1:$pp/v1 OPENAI_API_KEY=EMPTY MACSQL_MODEL=$5 timeout 36000 \
        $E/venv_macsql/bin/python -u run.py --dataset_name $ds --dataset_mode dev --input_file $inp --db_path $db \
        --tables_json_path $tj --output_file $O/output.jsonl --log_file $O/log.txt >> $O/stdout.txt 2>&1 )
    local rc=$?; echo "$(ts) M $mt $b s$k end $(basename $inp) rc=$rc n=$(wc -l < $O/output.jsonl)" >> $LOG
    [ $rc -eq 124 ] && echo "$(ts) M $mt $b s$k SHARD_TIMEOUT" >> $LOG
    if [ $inp = $O/gate_input.json ]; then
      if [ "$(wc -l < $O/output.jsonl)" -ge 1 ] && grep -q "decompose the question" $O/api_trace.json 2>/dev/null; then
        echo "$(ts) M $mt $b s$k GATE PASS" >> $LOG
      else echo "$(ts) M $mt $b s$k GATE FAIL: shard stops" >> $LOG; return 1; fi
    fi
  done
}
comem() {  # tag selector_model proxy_port gpu
  local O=$F/out/comem_$1; mkdir -p $O/work/models/hf_models
  ln -sfn ${HF_HOME}/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 $O/work/models/hf_models/flan-t5-xl
  echo "$(ts) C $1 start" >> $LOG
  ( cd $O/work && OPENAI_BASE_URL=http://127.0.0.1:$3/v1 OPENAI_API_KEY=EMPTY CUDA_VISIBLE_DEVICES=$4 HF_HUB_OFFLINE=1 \
      PYTHONPATH=$E/comem timeout 10800 $E/venv_comem/bin/python -u $E/run/run_comem_formal.py $E/comem/data/llm4em \
      $E/run/comem_dev_nolabel.json $2 $O > $O/stdout.txt 2>&1 )
  echo "$(ts) C $1 end rc=$?" >> $LOG; grep -E "GATE|done" $O/stdout.txt | sed "s/^/C $1 /" >> $LOG
}
worker() {  # gpu
  local g=$1 sp=81${1}5 pb=81${1}6 ps=81${1}7 pc=81${1}8 p7=81${1}9 pids
  # Phase A
  local util=0.75; [ $g = 3 ] && util=0.70
  start_server cis2-fx-q3-g$g $g $sp $Q3 qwen3-32b $util 40960 "--reasoning-parser qwen3" || return 2
  pids="$(proxy $pb $sp q3_bird_s$g M:q3:bird:s$g 20000) $(proxy $ps $sp q3_spider_s$g M:q3:spider:s$g 20000)"
  [ $g = 3 ] && pids="$pids $(proxy $pc $sp q3_comem C:q3 8000)"
  sleep 3
  shard q3 bird $g $pb qwen3-32b & local a=$!
  shard q3 spider $g $ps qwen3-32b & local b=$!
  local c=""; [ $g = 3 ] && { comem q3 qwen3-32b $pc 3 & c=$!; }
  wait $a $b $c; kill $pids 2>/dev/null; stop_server cis2-fx-q3-g$g
  echo "$(ts) GPU$g phase A finished" >> $LOG
  # Phase B
  util=0.75; [ $g = 3 ] && util=0.50
  start_server cis2-fx-q72-g$g $g $sp $Q72 qwen2.5-72b $util 32768 "" || return 2
  pids="$(proxy $pb $sp q72_bird_s$g M:q72:bird:s$g 20000) $(proxy $ps $sp q72_spider_s$g M:q72:spider:s$g 20000)"
  c=""
  if [ $g = 3 ]; then
    start_server cis2-fx-q7-g3 3 $p7 $Q7 qwen2.5-7b 0.20 32768 "" && { pids="$pids $(proxy $pc $p7 q7_comem C:q7 8000)"; sleep 3; comem q7 qwen2.5-7b $pc 3 & c=$!; }
  fi
  sleep 3
  shard q72 bird $g $pb qwen2.5-72b & a=$!
  shard q72 spider $g $ps qwen2.5-72b & b=$!
  if [ -n "$c" ]; then wait $c; stop_server cis2-fx-q7-g3; fi
  wait $a $b; kill $pids 2>/dev/null; stop_server cis2-fx-q72-g$g
  echo "$(ts) GPU$g phase B finished" >> $LOG
}
cleanup() { for n in $(docker ps -a --format '{{.Names}}' | grep '^cis2-fx-'); do stop_server $n; done; pkill -f "$F/raw/" 2>/dev/null; }
trap cleanup EXIT
echo "$(ts) formal start" >> $LOG
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
(cd $E/data/dev_databases && sha256sum */*.sqlite; cd $E/data/spider_database && sha256sum */*.sqlite) > $F/logs/db_before.sha256
worker 1 & worker 2 & worker 3 &
wait
(cd $E/data/dev_databases && sha256sum */*.sqlite; cd $E/data/spider_database && sha256sum */*.sqlite) > $F/logs/db_after.sha256
sha256sum $F/raw/*.jsonl > $F/logs/raw.sha256
echo "$(ts) done" >> $LOG
