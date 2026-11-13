# 15 Stronger peers and the LLM selector controls

Produces Section V-C and supplementary Tables S10–S12 (paper object `text_peer-source`), and the
selector controls of Table S5 and Section S4-B (paper object `tab_strength`):

- a stronger peer source (Qwen3-32B) for the QA recipients Qwen3-32B, gpt-oss-120b and Gemma-4-26B, and
  Qwen2.5-7B as a weaker recipient of the same peers (`w3_*`, `build_requests.py`, `run_recipient.py`,
  `score.py`, `chain.sh`);
- one Qwen3-32B candidate selector over the four-model SQL pool and its order and reasoning controls
  (`w6_*`).

## Code

| File | Role |
|---|---|
| `common.py` | OpenAI-compatible client, record schema, parsing; endpoints from `INFERENCE_HOST` and the ports in the file |
| `w3_peers.py`, `w3_recipients.py`, `w3_score.py`, `w3_plot.py` | generate the stronger peer replies, the recipients' replies with each peer content, and the sharing effects |
| `build_requests.py`, `run_recipient.py`, `score.py`, `chain.sh` | Qwen2.5-7B as recipient of the same peers (`SPWR_WORKDIR` on the GPU host) |
| `w6_selector.py`, `w6_arms.py`, `w6_score.py`, `w6_arms_score.py`, `w6_position.py` | selector over the fixed pool; fixed and permuted candidate order, reasoning on and off; scoring and position diagnostic |

## Run

```
python3 code/w3_peers.py && python3 code/w3_recipients.py && python3 code/w3_score.py
python3 code/w6_selector.py --main && python3 code/w6_score.py
python3 code/build_requests.py && bash code/chain.sh && python3 code/score.py
```

## Data

Inputs and logged outputs are in the hosted data folder: `experiments/15_stronger_peers/{data,raw,output}/`;
the QA units are `data/records/framework_raw/e1_units.jsonl` and `experiments/08_cross_family/data/`.
