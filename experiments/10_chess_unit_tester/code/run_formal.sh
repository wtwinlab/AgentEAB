#!/bin/bash
# CHESS B1 formal run: 3 Qwen3-32B replicas (GPU1/2/3), one logging proxy
# per replica (no call cap), 3 database shards run in parallel, each question one independent CHESS
# invocation with one worker, in question-id order within its shard. Interface gate after the
# first question of each shard. Servers and proxies are removed on exit (also on failure).
set -u
: "${CHESS_WORKDIR:?set CHESS_WORKDIR to the CHESS checkout, venv, run/ inputs}"
: "${HF_HUB:=$HOME/.cache/huggingface/hub}"
B=${CHESS_WORKDIR}
cd $B
LOG=$B/logs/run_formal.log
ts() { date -u +%FT%TZ; }
GPUS=(1 2 3); SPORTS=(8065 8075 8085); PPORTS=(8066 8076 8086)
NEED=$((107900 + 28754))
PIDS=""
cleanup() {
  for p in $PIDS; do kill $p 2>/dev/null; done
  for i in 0 1 2; do docker rm -f cis2-chess-f$((i+1)) >/dev/null 2>&1 && echo "$(ts) removed cis2-chess-f$((i+1))" >> $LOG; done
  nvidia-smi -i 1,2,3 --query-gpu=index,memory.used,memory.free --format=csv,noheader >> $LOG
}
trap cleanup EXIT
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
for i in 0 1 2; do
  g=${GPUS[$i]}
  free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "$(ts) GPU$g free ${free} MiB (need $NEED)" >> $LOG
  [ "$free" -lt "$NEED" ] && { echo "$(ts) ABORT: GPU$g below need" >> $LOG; exit 2; }
done
for i in 0 1 2; do
  docker run -d --name cis2-chess-f$((i+1)) --gpus device=${GPUS[$i]} --ipc=host -p 127.0.0.1:${SPORTS[$i]}:8000 \
    -v ${HF_HUB}:/weights:ro vllm/vllm-openai:v0.32.0 \
    --model /weights/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
    --served-model-name qwen3-32b --reasoning-parser qwen3 --max-model-len 40960 --max-num-seqs 96 \
    --gpu-memory-utilization 0.75 >> $LOG 2>&1 || { echo "$(ts) ABORT: docker run $i failed" >> $LOG; exit 2; }
  echo "$(ts) started cis2-chess-f$((i+1)) GPU${GPUS[$i]}:${SPORTS[$i]}" >> $LOG
done
for i in 0 1 2; do
  for k in $(seq 1 120); do curl -sf http://127.0.0.1:${SPORTS[$i]}/v1/models >/dev/null && break; sleep 10; done
  curl -sf http://127.0.0.1:${SPORTS[$i]}/v1/models >/dev/null || { echo "$(ts) ABORT: server $i not ready" >> $LOG; exit 2; }
done
echo "$(ts) servers ready" >> $LOG
nvidia-smi -i 1,2,3 --query-gpu=index,memory.used,memory.free --format=csv,noheader >> $LOG
. $B/venv/bin/activate
mv $B/chess/results $B/chess/results_r3 2>/dev/null
for i in 0 1 2; do
  echo "F:s$((i+1)):idle" > $B/run/current_question_s$((i+1))
  python -u $B/run/proxy.py --listen ${PPORTS[$i]} --upstream http://127.0.0.1:${SPORTS[$i]} --cap 1000000000 \
    --log $B/raw/formal_s$((i+1)).jsonl --question-file $B/run/current_question_s$((i+1)) >> $B/logs/proxy_formal_s$((i+1)).log 2>&1 &
  PIDS="$PIDS $!"
done
sleep 3
shard() {
  s=$1; i=$((s-1))
  QS=$(python -c "
import json
sh=json.load(open('$B/run/shards.json'))['shards'][$i]
t=[x for x in json.load(open('$B/run/tasks_147.json')) if x['db_id'] in sh]
print(' '.join(str(x['question_id']) for x in sorted(t, key=lambda x: x['question_id'])))")
  first=1
  for qid in $QS; do
    python -c "import json;t=[x for x in json.load(open('$B/run/tasks_147.json')) if x['question_id']==$qid];json.dump(t,open('$B/run/f_$qid.json','w'))"
    echo "F:q$qid" > $B/run/current_question_s$s
    echo "$(ts) s$s start q$qid" >> $LOG
    t0=$(date +%s)
    ( cd $B/chess && LOCAL_LLM_BASE=http://127.0.0.1:${PPORTS[$i]}/v1 timeout 3600 python -u src/main.py --data_mode dev \
        --data_path $B/run/f_$qid.json --config ./run/configs/CHESS_IR_CG_UT_local.yaml --num_workers 1 \
        --pick_final_sql true > $B/logs/formal_q$qid.out 2>&1 )
    rc=$?
    echo "$(ts) s$s end q$qid rc=$rc wall=$(( $(date +%s) - t0 ))s" >> $LOG
    [ $rc -eq 124 ] && echo "$(ts) s$s TASK_TIMEOUT q$qid" >> $LOG
    if [ $first -eq 1 ]; then
      first=0
      python - <<EOF >> $LOG 2>&1 || { echo "$(ts) s$s GATE FAILED: shard stops" >> $LOG; return 1; }
import glob, json, sys
h = sorted(glob.glob("$B/chess/results/dev/CHESS_IR_CG_UT/f_$qid/*/${qid}_*.json"))
names = [x.get("tool_name") for x in json.load(open(h[-1]))] if h else []
ok = bool(h) and any(n in ("generate_unit_test", "evaluate") for n in names)
print("s$s GATE", "PASS" if ok else "FAIL", names)
sys.exit(0 if ok else 1)
EOF
    fi
  done
  echo "F:s$s:done" > $B/run/current_question_s$s
  echo "$(ts) s$s finished" >> $LOG
}
shard 1 & S1=$!
shard 2 & S2=$!
shard 3 & S3=$!
wait $S1 $S2 $S3
echo "$(ts) all shards finished" >> $LOG
cd $B/data/dev_databases && sha256sum */*.sqlite > $B/logs/db_formal_after.sha256
sha256sum $B/raw/formal_s*.jsonl > $B/logs/formal_raw.sha256
echo "$(ts) done" >> $LOG
