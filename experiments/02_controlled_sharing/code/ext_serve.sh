#!/bin/sh
# Starts or stops the Llama-3-8B inference server used by the benchmark extension.
#
#   sh ext_serve.sh status      # which aliases the extension queue needs, and which answer
#   sh ext_serve.sh up          # start ext-llama3-8b on GPU 0, sized from the free memory there
#   sh ext_serve.sh down        # stop and remove the container this script created
#
# The server uses the same checkpoint, image and served name as the main collection. The script
# manages only the container `ext-llama3-8b`; `up` is idempotent, and it never prunes images,
# volumes, weights or data.
set -u
NAME=ext-llama3-8b
PORT=9211
DEV=0
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
SERVED=llama3-8b-instruct
IMAGE=vllm/vllm-openai:latest
HFCACHE=${HFCACHE:-$HOME/.cache/huggingface}
UTIL=0.55

# the aliases the extension's queue resolves to, and the port each uses after ROUTES.json
probe() {
  for spec in "9211 llama3-8b-g1 (llama3-8b panel + both rosters)" \
              "9221 qwen25-7b-g1b / qwen25-7b-g2 (qwen panel + both rosters)" \
              "9224 gemma4-g5 (MoA roster only)"; do
    p=$(echo "$spec" | cut -d' ' -f1)
    rest=$(echo "$spec" | cut -d' ' -f2-)
    got=$(curl -s -m 8 "http://127.0.0.1:$p/v1/models" | head -c 400)
    if [ -n "$got" ]; then
      id=$(echo "$got" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
      echo "  $p  UP    $id    $rest"
    else
      echo "  $p  DOWN  -                     $rest"
    fi
  done
}

case "${1:-status}" in
  status)
    echo "endpoints the extension queue needs:"
    probe
    echo
    echo "containers (only ext-* are this script's):"
    docker ps -a --format "  {{.Names}}  {{.Status}}  {{.Ports}}" | grep -E "ext-|cx-" || echo "  none"
    echo
    echo "GPU $DEV tenants, left untouched:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $DEV \
      | sed 's/^/  /' || true
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i $DEV \
      | sed 's/^/  card: /'
    ;;
  up)
    if [ "$DEV" = "3" ]; then echo "refusing: GPU 3 is off limits to this run"; exit 2; fi
    if curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
      echo "port $PORT already answering -- nothing to do"
      exit 0
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "removing this script's previous $NAME (not answering)"
      docker rm -f "$NAME" >/dev/null
    fi
    echo "GPU $DEV before start:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $DEV | sed 's/^/  /'
    docker run -d --name "$NAME" --gpus "\"device=$DEV\"" \
      --ipc=private --shm-size=8g --restart=no \
      -v "$HFCACHE:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
      -p "$PORT:8000" "$IMAGE" \
      --model "$MODEL" --served-model-name "$SERVED" \
      --gpu-memory-utilization "$UTIL" --max-model-len 8192 --max-num-seqs 512 \
      || { echo "docker run failed"; exit 1; }
    printf "waiting for %s on %s" "$SERVED" "$PORT"
    i=0
    while [ $i -lt 120 ]; do
      if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
        echo " -- up"
        probe
        exit 0
      fi
      printf "."
      i=$((i + 1))
      sleep 5
    done
    echo " -- TIMED OUT after 10 min; last 40 log lines:"
    docker logs --tail 40 "$NAME" 2>&1
    exit 1
    ;;
  down)
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "$NAME does not exist -- nothing of this script's to remove"
      exit 0
    fi
    docker rm -f "$NAME" >/dev/null && echo "removed $NAME; port $PORT and GPU $DEV memory freed"
    echo "left running (not this script's): $(docker ps --format '{{.Names}}' | grep -E '^cx-' \
      | tr '\n' ' ')"
    ;;
  *)
    echo "usage: sh ext_serve.sh [status|up|down]"; exit 1 ;;
esac
