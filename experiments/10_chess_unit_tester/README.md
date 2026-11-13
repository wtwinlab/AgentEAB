# 10 CHESS Unit Tester and its rule replacement

Produces the CHESS inputs of Table VII (Unit Tester row), Fig. 6 and supplementary Tables S25, S26 and
S28 (paper objects `tab_replacement-cost`, `fig_stage-act`): CHESS (IR + CG + UT) with Qwen3-32B on the
147 sampled BIRD development questions, every LLM call logged by a proxy, and the replacement of the
Unit Tester by the pipeline's self-consistency vote and by execution votes over the same candidates.

## Code

| File | Role |
|---|---|
| `run_formal.sh` | serves three Qwen3-32B replicas (vLLM), starts one logging proxy per replica (`proxy.py`), runs CHESS on three database shards; needs `CHESS_WORKDIR` (CHESS checkout with `patches/chess_local.patch` applied, its venv, `run/` with `tasks_147.json`, `shards.json`, `configs/CHESS_IR_CG_UT_local.yaml`) and `HF_HUB` (model weights) |
| `proxy.py` | OpenAI-compatible logging proxy: every generation request with its usage and timing |
| `extract_b1.py`, `b1_selectors.py` | per question: the candidate list before the Unit Tester, the Unit Tester's pick, the rule picks |
| `analyze_formal.py`, `result_agreement.py` | levels, contrasts and per-database values (bootstrap over databases) |
| `run_cli.py`, `chess_b1_adapter.py`, `chess_formal_spec.json` | the same estimates through the stage-replacement CLI in `../common/code/` |
| `patches/` | local-server adaptation of CHESS (OpenAI-compatible endpoint, local embedding model; replace `<HF_HUB>` with your weights directory) and the CHESS config |

## Run

```
export CHESS_WORKDIR=<work dir> HF_HUB=<huggingface hub dir>
bash code/run_formal.sh                       # collection (three GPUs)
python3 code/extract_b1.py --base $CHESS_WORKDIR --out output/offline
python3 code/analyze_formal.py
```

## Data

Inputs and logged outputs are in the hosted data folder: `experiments/10_chess_unit_tester/{data,raw,output}/`
(per-call proxy logs, per-question records). The BIRD databases are those of `data/external/`.
