# Stronger peers

Labels: `sec:peer-source` (Section V-C) and Supplementary `sec:s-spend-peers` (Section S5-D) with
`tab:s-peer-effects` (Table S10), `tab:s-peer-change` (Table S11) and `tab:s-weaker` (Table S12).

## What is printed

The peer responses of the fixed-peer QA comparison come from Qwen3-32B instead of Qwen2.5-7B or
Llama-3-8B, on the same 480 tasks and three positions. The section prints how much the
answers-only effect of the recipients (Qwen3-32B, gpt-oss-120b, Gemma-4-26B-A4B) changes (2.36–3.75
pp), that correct-to-wrong changes fall in all six recipient–format combinations, and the
accuracies of a Qwen2.5-7B recipient with Qwen3-32B peers. The supplementary tables give the nine
sharing effects with Qwen3-32B peers, their change from Qwen2.5-7B peers split into recovered and
lost components, and the Qwen2.5-7B recipient by benchmark; its text adds the peers' own accuracy
by source and benchmark, the effects against the no-peer response collected with the Llama-3-8B
peer set, the three answers-minus-full changes, and the Qwen2.5-7B recipient's gap to its peers,
effects, transitions, tokens and truncation.

## Inputs

- `inputs/peer_source_outcomes.csv` (34,560 rows): recipient, peer source (`llama3-8b`,
  `qwen25-7b`, `qwen3-32b`), condition, task, position, benchmark,
  correct (0/1), first decode only. The Qwen3-32B source has no no-peer rows: its baseline is the
  no-peer response collected with the Qwen2.5-7B peer set, and the sensitivity uses the one
  collected with the Llama-3-8B set.
- `inputs/peer_source_peers.csv` (4,320 rows): each peer's own answer (source, benchmark, task,
  position, correct).
- `inputs/weaker_recipient_units.csv` (1,440 rows): one position of the Qwen2.5-7B recipient with
  Qwen3-32B peers: outcomes without peers, with answers and with full responses; the Qwen3-32B
  peer outcomes of the task's three positions; the outcomes with its own Qwen2.5-7B peers; input
  and output tokens and length truncation per condition; agreement of the re-scored no-peer
  outcome with its stored grade.

Grading is `consensus_delivery.adapters.mad_m2.extract_answer` (in `code/`) with the benchmarks' scorers
(`code/consensus_delivery/adapters/official_scorers.py`); an unreadable answer counts as wrong. The tables
were derived once from the saved responses.

## Command

```sh
python3 code/compute.py --out output
```

Writes `output/values.json`, `output/effects.csv` (all effects by recipient, source and benchmark)
and `output/changes.csv` (source changes).

## Estimates

- A difference is the mean over positions, each task weighted by its number of positions; changes
  between sources are paired differences of the per-position differences.
- Intervals resample tasks (10,000 draws, percentiles 2.5 and 97.5). The three larger recipients
  draw every estimate from one generator (seed 20261101) in the order recipient, source,
  benchmark, contrast, followed by the source changes; the Qwen2.5-7B recipient uses one generator
  with seed 20261105. Holm at 0.05 covers the nine effects with Qwen3-32B peers, and separately the
  nine changes from Qwen2.5-7B peers.
- Recovered is the share of all positions that are wrong without peers and correct with them; lost
  is the share that are correct without peers and wrong with them. The table's ΔRecovered and ΔLost
  subtract the two-decimal components of the two sources; `values.json` also holds the unrounded
  differences (`*_exact`).

## Check

`python3 ../check_objects.py --recompute --only text_peer-source` compares every listed value with
Section V-C or the supplement. Both ends of printed
ranges such as "8.54--15.49" are checked as separate values.
