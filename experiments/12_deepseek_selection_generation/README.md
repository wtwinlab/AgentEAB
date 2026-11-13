# 12 DeepSeek-V4-Pro selection and generation on the fixed pool

Produces the DeepSeek-V4-Pro inputs of Table IX, Section VIII-A and supplementary Tables S28, S32 and
S37 (paper objects `tab_strength`, `tab_replacement-cost`, `text_generation-beyond-pool`): one selection
call (the selector given the generation prompt) and one generation call per question on the four-model
BIRD (1,534) and Spider (1,034) pool, and the availability study with six Qwen3-32B samples per question.

## Code

| File | Role |
|---|---|
| `collect_hosted.py`, `run_ds_full.sh` | hosted OpenAI-compatible collection with a per-attempt call log and a spending cap; needs `DEEPSEEK_API_KEY` |
| `gen_candidates_avail.py`, `collect_local.py`, `run_partA_cands.sh` | Qwen3-32B samples for the availability study from a local vLLM server (`GEN_HOST`, `GEN_PORT`) |
| `build_avail.py`, `analyze_avail.py` | parse, execute read-only and label the samples; availability analysis |
| `score_ds.py`, `analyze_ds.py` | score the selection and generation outputs against the stored candidate labels and references; contrasts with voting |

## Run

```
export DEEPSEEK_API_KEY=...
bash code/run_ds_full.sh
python3 code/score_ds.py raw/main_Sal.jsonl raw/main_G.jsonl output/ds_per_question.csv
python3 code/analyze_ds.py
```

## Data

Inputs and logged outputs are in the hosted data folder: `experiments/12_deepseek_selection_generation/{data,raw,output}/`;
the pool tables are `data/derived/sql_pool_operation_transfer/` and `data/derived/sql_decomposition/`.
