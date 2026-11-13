# QA with fixed peer responses: sharing effects

Labels: `tab:family-sharing` (Table III) and its paragraphs in `sec:matching-access` (Section V-B),
and Supplementary `tab:s-first-decode` (Table S6, `sec:s-first-decode`, Section S4-C) and
`tab:s-qa-transitions` (Table S7, `sec:s-qa-transitions`, Section S5-B).

## What is printed

Qwen3-32B, gpt-oss-120b and Gemma-4-26B-A4B answer 480 QA tasks (120 each from GSM8K, MMLU-Pro,
GPQA and StrategyQA) with no peers, with the answers of two Llama-3-8B or Qwen2.5-7B peers, or
with their full responses. The table gives the paired differences full minus none, answers minus
none and answers minus full, with 95% intervals, and the no-peer accuracy. The text adds the Holm
decisions over the nine differences, the accuracy of the 2,880 distinct peer answers, the 12
recipient–benchmark comparisons of peer and recipient accuracy, and the share of task and peer-set
combinations in which the recipient is correct while the peer majority is wrong. The supplementary
tables repeat the nine differences on first decodes only and split the answers-only difference into
recovered errors and lost successes.

## Inputs

- `inputs/qa_recipient_units.csv` (29,480 rows): one scored recipient response. Columns: recipient,
  response id, task, benchmark, peer set (`panel`), fixed initial responses (`prefix_id`),
  position (`agent`), decode, condition, correct (0/1), readable (0/1). An unreadable answer counts
  as wrong.
- `inputs/qa_peer_slots.csv` (7,058 rows): one peer answer as shown in an answers-only prompt
  (peer set, task, position of the peer, position of the recipient, decode). `correct` charges the
  literal block that stands for an unreadable peer answer as wrong; `correct_as_rendered` grades
  that block as the StrategyQA answer "no", which is how the answers-only prompt reads it.

Both tables were derived once from the saved responses of the fixed-peer QA collection
(`experiments/08_cross_family/code/`) through the answer extractor and benchmark scorers in
`code/consensus_delivery/adapters/` (`mad_m2.extract_answer`, `official_scorers.py`).

## Command

```sh
python3 code/compute.py --out output
```

Writes `output/values.json`, `output/contrasts.csv` (the nine differences with intervals,
bootstrap p and Holm decision), `output/first_decode.csv` and `output/transitions.csv`.

## Estimates

- Conditions are paired inside one cell (peer set, fixed initial responses, position, decode). A
  difference is the mean over tasks of the per-task mean paired difference.
- Intervals: 10,000 resamples of the 480 tasks, percentile 2.5 and 97.5, a fresh generator with
  seed 20261028 for each estimate; Holm at 0.05 over the nine differences. The first-decode table
  draws all 18 estimates (all decodes, then first decodes, per recipient) from one generator with
  seed 20261105 and applies Holm within each set of nine.
- Transitions: answers only against no peers, each task weighted equally. `q` is the paired no-peer
  accuracy, `c` the share of no-peer errors corrected, `r` the share of no-peer successes lost;
  recovered and lost are their contributions in pp. The rows with both peers wrong use
  `correct_as_rendered` to count correct peers.
- Peer accuracy counts each (peer set, task, position) once (2,880 answers) and uses `correct`. The
  12 comparisons set the mean peer accuracy of each benchmark against the recipient's no-peer
  accuracy on the same tasks; the overlap share uses the majority of the recipient's no-peer
  responses and of the three peer answers in each of the 960 task and peer-set combinations.

## Check

`python3 ../check_objects.py --recompute --only tab_family-sharing` runs the command again and
compares every value in `object.json` with the number printed in Section V-B or the supplement.
Both ends of printed ranges such as "25--32%" are checked as separate values. Statements check the
Holm decisions the text states: both answers-only directions survive the nine-comparison correction,
and first generations change no decision.
