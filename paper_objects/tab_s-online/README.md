# tab_s-online: BCCS run online with execution voting, Qwen2.5-7B

Labels covered: Supplementary Table `tab:s-online` (Table S42) and the online-check paragraph of
`sec:s-spend-scale` (Section S8-C); `sec:s-online-bound` (Section S2-C); the replay-accuracy sentence of
the validation paragraph in Section IV (`sec:evaluation-checks`): online execution voting inside BCCS
differs from the replay by −0.25 pp with peers and +0.25 pp without, while the worst-case bounds allow gaps
of 79 and 53 pp.

## What is printed

BCCS with Qwen2.5-7B agents, with execution voting inside the protocol, on 198 BIRD questions, with and
without peers, two repeats each.
- `tab:s-online`: the interaction effect (change from the initial answers to the last round with peers minus
  the same change without peers, paired by question) with ties to the first agent, to the last agent, at
  random, and with ties to the first agent when the runs whose stop test fires on a single executing answer are
  excluded; 95% intervals over questions. The supplementary text adds the database-cluster interval for ties
  to the first agent and the number of databases in which the per-database difference is positive.
- `sec:s-online-bound`: per condition, the online last-round accuracy minus the replayed last-round accuracy
  of the logged BCCS debates, with its 95% interval over questions, and the same difference on the no-peer
  questions whose logged runs diverge only recoverably; the gap the bound of Lemma 1 allows (the share of
  logged runs whose replay diverges from deployment, 79.0 and 53.0 pp); the run-level outcome disagreement
  between the online and the replayed runs with the same question and repeat index (11.4% with peers, 9.8%
  without) next to the repeat-to-repeat disagreement inside each collection (7.6–13.6%). The online outcomes
  lie inside the deployment bounds of Lemma 1 in both conditions (`online_inside_bounds_*` in
  `values.json`, with the bounds themselves).

Questions 518 and 701 are excluded because their reference query exceeds the time limit.

## Inputs

- `inputs/online_bccs_runs.csv`, 792 rows. One row is one online run (`condition`, `question`, `database`,
  `repeat`): correctness of the vote over the initial answers and over the last round under the lenient extractor
  and the execution vote, for ties to the first agent, to the last agent and at random, and
  `single_answer_stop` (1 if the run's stop test fired on a single executing answer).
- `inputs/replay_bccs_qwen7b.csv`, 792 rows. One row is one logged Qwen2.5-7B BCCS run (`condition`,
  `question`, `database`, `repeat`): `divergence` (none, recoverable, unrecoverable: whether and how the
  replayed vote changes BCCS's stop decision), `replay_last_correct` (replayed last-round outcome under the same
  reading), and `bound_low` / `bound_high` (the run's deployment bounds of Lemma 1).

The extractor and vote are those of the paper (lenient extraction, execution vote, ties as stated); the
divergence classes and the bounds follow Lemma 1; the single-answer stop flag is read from each online run's
recorded stop trace. Both tables were derived once from the logged runs. The no-peer rows are the no-peer
runs in these inputs; their prompts contain no peer content.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` and `output/online_vs_replay.csv` (per question and condition: online accuracy,
replayed accuracy, divergence class). Runs in a few seconds.

## Estimation

Bootstrap with 10,000 draws and seed 20261106 over questions; for the interaction effect also over the 11
databases (all questions of a drawn database enter). Percentile intervals. The online run and the
replay are compared per question in distribution: the online outcome matches the replayed one on average
(Lemma 1).

## Check

`python3 ../check_objects.py --recompute --only tab_s-online` repeats the computation, requires the same
`values.json`, and compares each value in `object.json` with the number printed in the paper at the
printed precision. Three statements check that the online accuracy lies inside the bounds in both
conditions and that the per-database difference is positive in four of the 11 databases.
