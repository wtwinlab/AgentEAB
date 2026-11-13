# 17 Debate budgets and independent drafts

The BIRD collection behind Section VIII-E and supplementary Section S8-D (paper object `fig_debate-budget`).
On the 198 BIRD questions of the debate comparisons (`code/questions.json`), Du-MAD and MAD-M² with
Llama-3-8B or Qwen2.5-7B agents run at four budgets, and each protocol's initial prompt produces 20
independent drafts per question:

| Condition | Agents × rounds | Calls per run | Repeats |
|---|---|---|---|
| `FULL` | 3 × 2 | 6 | 2 |
| `R3_FULL` | 3 × 3 | 9 | 2 |
| `A5_FULL` | 5 × 2 | 10 | 2 |
| `R4_FULL` | 3 × 4 | 12 | 2 |
| `DRAFTS20` | 20 × 1, the protocol's initial prompt and decoding (Du-MAD greedy, MAD-M² temperature 1) | 20 | 1 |
| `T07_DRAFTS20` | Du-MAD only: as `DRAFTS20` at temperature 0.7, each call with its own server seed | 20 | 1 |

A condition changes only the numbers of agents and rounds (and, for `T07_DRAFTS20`, the decoding).
Prompts, the 1,024-token completion budget, the protocols' later-round templates and return rules are the
collector's own: Du-MAD shows the peers' previous-round replies and returns the majority of its last round;
MAD-M² shows its debate prompt over the previous round with objective pruning and returns the answer of its
consensus rule. Both models are served by vLLM 0.30.0 with an 8,192-token context. A run whose request
the server rejects is recorded as failed and is not repeated under another setting; in this collection
these are 117 Du-MAD runs at 9 and 12 calls, whose prompts carry the accumulated history.

## Code

| File | Role |
|---|---|
| `bm_override.py` | runtime override of the extension collector (`experiments/02_controlled_sharing/code/ext_collect.py`): run identifier, output directory, questions, protocols, servers, and the conditions above |
| `bm_run.py`, `bm_stage.sh`, `bm_lane.sh` | one collector stage; one lane per question shard (3×2 debate, drafts, 3 and 4 rounds, 5 agents); a stage resumes from the records on disk |
| `bm_serve.sh` | vLLM servers for Llama-3-8B-Instruct and Qwen2.5-7B-Instruct |
| `bm_score.py` | read-only execution of every extracted query, verdicts against the reference result, last-round execution vote; writes `rows.jsonl`, `drafts.jsonl`, `exclusions.json` |
| `bm_tables.py` | the input tables `debate_runs.csv` and `drafts.csv` of `paper_objects/fig_debate-budget/` |
| `questions.json` | the 198 BIRD question identifiers |

## Run

```
bash code/bm_serve.sh up l8 <gpu> <port>; bash code/bm_serve.sh up q7 <gpu> <port>   # per lane
BM_PORT_L8=<port> BM_PORT_Q7=<port> bash code/bm_lane.sh <g> <n>                     # lane g of n
PYTHONPATH=data/records/protocol_all_matching:code:experiments/02_controlled_sharing/code \
  python3 code/bm_score.py --workers 16
python3 code/bm_tables.py
```

Commands run from the repository root, except `bm_lane.sh`, which changes to its own directory.
`BM_OUT` sets the output directory (default `experiments/17_budget_matched_debates/data`).
`BM_RUN_ID` is the run identifier recorded in each record; the item seeds and the per-call seeds of
`T07_DRAFTS20` derive from it, so regenerating the hosted records uses the identifier they carry.

## Data

Records of this collection: `experiments/17_budget_matched_debates/data/` in the hosted data folder
(`raw/<lane>/` call records, `output/` scored rows and drafts). The BIRD development databases are
`data/records/dataeng_raw/sql/dev_20240627/`.
