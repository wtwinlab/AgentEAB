# 11 MAC-SQL Refiner and ComEM selector stages

Produces the MAC-SQL and ComEM inputs of Table VII, Fig. 6 and supplementary Table S24 (paper objects
`tab_replacement-cost`, `fig_stage-act`): MAC-SQL with Qwen3-32B and Qwen2.5-72B-AWQ on all BIRD and
Spider development questions, ComEM with Qwen3-32B and Qwen2.5-7B selectors on eight matching sets, and
the stage decomposition (how often the Refiner or the selector changes the result, its gain when it
does, and its calls, tokens and GPU-seconds).

## Code

| File | Role |
|---|---|
| `apply_patch_macsql.py`, `apply_patch_comem.py`, `patches/` | local-server adaptation of the MAC-SQL and ComEM checkouts (OpenAI-compatible endpoint, read-only databases, no cost table for local models) |
| `run_formal_ext.sh`, `run_phaseB.sh` | serve the models (vLLM), start one logging proxy per shard, run MAC-SQL shards and ComEM; need `EXT_WORKDIR` (checkouts, venvs, `run/` inputs) and `HF_HUB`/`HF_HOME` (weights) |
| `build_comem_dev.py`, `run_comem_formal.py` | ComEM test anchors, entity-disjoint development split, and the ComEM run |
| `extract_macsql_formal.py`, `analyze_formal_ext.py` | per-question MAC-SQL records (Decomposer query, Refiner query) and the stage estimates |

## Run

```
export EXT_WORKDIR=<work dir> HF_HOME=<cache dir>
bash code/run_formal_ext.sh && bash code/run_phaseB.sh      # collection
python3 code/analyze_formal_ext.py
```

The stage decomposition (Eq. (4)), re-execution by result equality, cost table and GPU-seconds are computed
by `share_eval/stage_replacement/` (see `share_eval/README.md`):

```
python3 share_eval/stage_replacement/reexec.py && python3 share_eval/stage_replacement/throughput.py
python3 share_eval/stage_replacement/decompose.py && python3 share_eval/stage_replacement/costs_comparable.py
```

## Data

Inputs and logged outputs are in the hosted data folder: `experiments/11_macsql_comem_stages/{data,raw,output}/`.
