# 14 No-peer collection for Du-MAD and BCCS

The no-peer collection for Du-MAD and BCCS used by the paper. Its records are the no-peer inputs of every
paper object that reports these protocols without peers (`fig_access-matrix`, `fig_readout-cells`,
`tab_factorial`, `tab_matrix`, `tab_peer-controls`, `tab_s-controls`, `tab_s-reader-sensitivity`, `tab_scale`,
`tab_s-cross-family`, `tab_s-online`, `tab_sensitivity-census`, `text_matching-return-rule`).

The no-peer condition removes the peer replies and the clauses that refer to them from every prompt and
keeps rounds, stopping and return rules. Here each peer reply is replaced at prompt construction by a
fence-free placeholder, the condition transform of `experiments/02_controlled_sharing/code/arm_transform.py`
removes every peer block with its preamble, and every sent request is checked to contain no placeholder
and is logged (`messages_after_arm`). Each run uses the seed of the corresponding no-peer run, so the
questions, seeds, decoding and models are those of the with-peers runs.

## Code

| File | Role |
|---|---|
| `cn_override.py`, `cn_run.py`, `cn_stage.sh` | BIRD debates (extension collector), Llama-3-8B, Qwen2.5-7B and the larger backbones of `../13_backbone_scale_cross_family/`; output `data/sql/` (`CN_OUT`) |
| `cm_override.py`, `cm_run.py`, `cm_stage.sh` | QA and matching (controlled collector: GSM8K, StrategyQA, MMLU-Pro, WDC, Settlements); output `data/main/` (`CM_OUT`) |
| `cg_run.py`, `cg_stage.sh` | GPQA in the extension collector |
| `cn_peer_text_check.py`, `cn_gate.py` | check on the sent prompts: no 40-character window of an earlier peer reply, no placeholder, no peer-block marker |
| `cn_build_records.py` | builds the record directories the analyses read: the shards of the collections named below together with this collection's Du-MAD and BCCS no-peer records for the listed cells (condition `NOPEER`) |

## Run

```
CN_PORT_L8=<port> CN_PORT_Q7=<port> SB_BACKBONES=llama3-8b,qwen2.5-7b SB_PROTOS=du_mad,bccs \
  bash code/cn_stage.sh l8q7 ext_collect --run --benchmark sql_bird_v2 --workers 8
python3 code/cn_peer_text_check.py data/sql/raw > output/peer_text_check.json
python3 code/cn_build_records.py ext            # likewise sb, c5
```

`CN_RUN_ID` and `CN_ORIG_RUN_ID` are run identifiers recorded in each record; the item seed is derived
from `CN_ORIG_RUN_ID`.

## Data

Records of this collection: `experiments/14_nopeer_collection/data/{sql,main}/` in the hosted data folder;
the original shards it merges with: `data/records/{benchmark_extension,backbone_scale_cross_family,controlled_5x5_collection}/`.
