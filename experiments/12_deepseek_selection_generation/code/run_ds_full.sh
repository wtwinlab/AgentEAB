#!/bin/bash
# Full DeepSeek-V4-Pro collection: selection (Sal) and generation (G), 2,568 questions each, cap $25 per
# condition at peak prices. Needs DEEPSEEK_API_KEY in the environment.
cd "$(dirname "$0")/.."
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
date -u +%FT%TZ > logs/ds_full_start.txt
for arm in Sal G; do
  python3 code/collect_hosted.py --input data/ds_inputs_${arm}_P0.jsonl --model deepseek-v4-pro --out raw/main_${arm}.jsonl \
    --max-tokens 32768 --concurrency 16 --usd-per-m-in 1.32 --usd-per-m-cached 0.044 --usd-per-m-out 3.96 --max-usd 25 \
    > logs/ds_full_${arm}.log 2>&1 &
done
wait
date -u +%FT%TZ > logs/ds_full_end.txt
