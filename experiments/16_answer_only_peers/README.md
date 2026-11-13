# 16 Answer-only peers for Du-MAD and MAD-M²

The peer-content collection on BIRD used by the paper: Du-MAD and MAD-M², each with Llama-3-8B or
Qwen2.5-7B agents, on the 198 sampled BIRD development questions with two repeats, in three conditions
collected together: full peer responses, each peer's extracted SQL only, and no peers. Its runs are the
inputs of paper object `tab_answer-content` (Table VI, Section VI-D, Table S15).

In the answers-only condition every peer block shown in an exchange round carries only that peer's SQL,
rendered in the protocol's own answer form (Du-MAD `\boxed{q}` inside its usual block, MAD-M²
`Answer: q`), and no peer reasoning. The peer's SQL is the protocol's own extraction, else the lenient
reading (last fenced block, else last line-initial SELECT/WITH), else empty. MAD-M² still prunes peers on
their real perplexities, so the same peers are shown as with full responses, and Du-MAD keeps each
agent's own history. The template text, decoding, rounds and return rules are those of the full-response
condition. In Du-MAD's no-peer condition each peer reply is replaced at prompt construction by a
placeholder that the no-peer transform removes with its preamble, and every sent request is checked to
contain no placeholder; MAD-M²'s no-peer prompt holds no peer text.

Settings are those of the extension collector of `experiments/02_controlled_sharing/code/` for the 7–8B
BIRD debates: Du-MAD 3 agents × 2 rounds at temperature 0; MAD-M² with objective pruning, 3 agents × 2
rounds at temperature 1.0 with log-probabilities; 1,024 output tokens; no retries; 300 s timeout. Models
are served by vLLM 0.30.0 (`meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, served
names `llama3-8b-instruct` / `qwen2.5-7b-instruct`, 8,192-token context).

## Code

| File | Role |
|---|---|
| `ao_override.py`, `ao_run.py`, `ao_stage.sh` | run the extension collector with the four conditions (`FULL`, `ANSWER_ONLY`, `NOPEERC` for Du-MAD, `NOPEER` for MAD-M²); output in `data/` (`AO_OUT`) |
| `ao_lane.sh` | one lane: a question shard, all conditions, two repeats, resumed up to three times |
| `ao_serve.sh` | start or remove one local vLLM server (`HF_HOME`) |
| `units198.json` | the 198 sampled BIRD questions |
| `ao_score.py` | read-only scoring of the logged runs into the per-run table of `tab_answer-content` (lenient extractor, execution vote with first-agent ties, the protocol's final answer, per-agent correctness) |

## Run

```
bash code/ao_serve.sh up l8 <gpu> <port_l8>; bash code/ao_serve.sh up q7 <gpu> <port_q7>
AO_PORT_L8=<port_l8> AO_PORT_Q7=<port_q7> bash code/ao_lane.sh 0 4       # likewise lanes 1-3
python3 -I code/ao_score.py                                              # from the repository root
```

The collector runs once `data/GATE.json` holds `{"state": "open"}`. `AO_RUN_ID` is part of each item's
seed and is kept as recorded. `ao_score.py` executes queries read-only against the BIRD development
databases and writes `output/runs.csv`, identical to `paper_objects/tab_answer-content/inputs/runs.csv`
when run on the hosted records.

## Data

Records of this collection: `experiments/16_answer_only_peers/data/raw/` in the hosted data folder
(4,752 runs, all with status ok); execution caches: `experiments/16_answer_only_peers/output/eval_cache/`.
