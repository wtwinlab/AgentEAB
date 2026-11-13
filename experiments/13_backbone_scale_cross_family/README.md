# 13 Larger Qwen2.5 backbones and cross-family debates

Produces Table X, supplementary Tables S40–S42 and the online check of Section IV (paper
objects `tab_scale`, `tab_s-cross-family`, `tab_s-online`): Du-MAD, MAD-M² and BCCS with and without peers
on the 198 sampled BIRD questions, with Qwen2.5-14B/32B/72B and 16 models of other families, and BCCS
with execution voting inside the protocol (Qwen2.5-7B).

## Code

| File | Role |
|---|---|
| `sb_override.py`, `sb_run.py`, `sb_stage.sh` | run the extension collector of `experiments/02_controlled_sharing/code/` with these backbones (prompts, adapters, condition transforms and questions unchanged); output in `data/records/backbone_scale_cross_family/` (`SB_OUT`) |
| `sb_serve.sh` | start or remove one backbone's vLLM server (`HF_HOME`, `HF_HUB`) |
| `sb_g0.py`, `sb_g1_gate.py`, `sb_harmony.py` | per-backbone interface checks (thinking off, log-probabilities, chat template) and the answer-channel slicing for gpt-oss |
| `sb_bccs_execid.py` | BCCS with execution-result identity in its consensus test and final vote |
| `sb_fast_eval.py`, `sb_eval_equiv.py`, `sb_eval_equiv_sample.py` | evaluation of the delivered queries and its equivalence with the collector's evaluator |
| `sb_analyze.py`, `sb_q4.py`, `sb_cluster_ci.py`, `sb_online_checks.py` | estimands: interaction effect per model, scale and cross-family contrasts, question- and database-cluster intervals, online checks |

## Run

```
bash code/sb_serve.sh up q72                        # one backbone
SB_BACKBONES=qwen25-72b-awq bash code/sb_stage.sh q72 ext_collect --run --benchmark sql_bird_v2 --workers 8
python3 -I code/sb_q4.py
```

Model servers are local (`127.0.0.1:<port>`; ports per alias in `sb_override.py`, overridable by
`SB_PORT_*`). `RUN_ID` in `sb_override.py` is part of each run's seed and is kept as recorded.

## Data

Logged runs: `data/records/backbone_scale_cross_family/` in the hosted data folder; outputs:
`experiments/13_backbone_scale_cross_family/output/`.
