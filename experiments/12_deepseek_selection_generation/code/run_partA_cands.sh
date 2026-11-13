#!/bin/bash
# Part A: s2..s5 for every question (resumes over the pilot), both benchmarks, one endpoint each.
cd "$(dirname "$0")/.."
for b in bird spider; do
  GEN_HOST=127.0.0.1 GEN_PORT=${PORT_BIRD:-8305} ; [ $b = spider ] && GEN_PORT=${PORT_SPIDER:-8305}
  GEN_HOST=127.0.0.1 GEN_PORT=$GEN_PORT python3 code/gen_candidates_avail.py --benchmark $b --family qwen3-32b --samples 6 \
    --out raw/avail_cand_$b.jsonl --max-tokens 12288 --concurrency 64 --max-calls 6200 > logs/partA_cand_$b.log 2>&1 &
done
wait
date -u +%FT%TZ > logs/partA_cand_done.txt
