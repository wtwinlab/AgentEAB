# tab_s-cross-family: cross-family BIRD debates

Labels covered: supplementary `tab:s-cross-family` (Table S41) with the cross-family Holm values in
`sec:s-spend-scale` (Section S8-C), and the cross-family paragraph of Section VIII-D (`sec:scale`).

## What is printed

For each of 16 models, on 198 BIRD questions:
- later rounds: last-round accuracy minus initial-answer accuracy, with peers (pp);
- peer interaction effect: that change with peers minus the same change without peers, paired by question
  (pp);
- the 95% interval over questions of both.

Each per-model value is the per-question mean of Du-MAD and MAD-M²; MiniMax-M2.5 runs Du-MAD only, so its row
is a Du-MAD value. Section VIII-D gives the range of the peer interaction effect over the 16 models
(-4.29 to +2.15 pp), names the two models whose negative effects are detected under both pointwise
intervals (Llama-3-8B and Qwen2.5-14B), states that no positive effect is detected, and gives MiniMax-M2.5's
round-to-round change with peers (6.31 pp) and without (10.61 pp). The supplementary text gives the
Holm-adjusted p of the nine models of the cross-family family. All values use the lenient extractor, the
execution vote and ties to the first agent (questions 518 and 701 are excluded because their reference query
exceeds the time limit).

## Inputs

`inputs/debate_runs.csv`, 36,581 rows, identical to the file of `tab_scale`. One row is one debate run:
`model`, `protocol` (`du_mad`, `madm2_objective`, `bccs`), `condition` (`with_peers`, `no_peers`),
`question`, `database`, `repeat`, and `initial_correct` / `last_correct` (1 if the execution vote over the
initial answers, or over the last-round answers, returns a query whose result matches the reference). This
object reads the Du-MAD and MAD-M² rows.

The extractor and the vote are those of the paper (lenient extraction, execution signature with failing
queries dropped, first-agent ties); the per-run table was derived once from the logged runs.
The no-peer rows of Du-MAD are those in `inputs/debate_runs.csv`; their no-peer prompts contain no peer content.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` (every checked value), `output/cross_family.csv` (each model with both
intervals, p and Holm-adjusted p) and `output/levels.csv` (accuracy of the initial answers and of the last
round, with and without peers, per model and protocol). Runs in a few seconds.

## Estimation

Bootstrap with 10,000 draws and seed 20261106, once over questions and once over the 11 databases (all
questions of a drawn database enter); percentile intervals. The p-value is twice the smaller share of draws
on either side of zero, at least 1/10,000. Holm correction runs over the nine models marked in
`tab:s-cross-family`; the Qwen2.5 series, gpt-oss and Llama-3-8B are tested in their own contrasts.
`output/levels.csv` lists the accuracy of the initial answers and of the last round, with and without peers,
of Du-MAD and MAD-M².

## Check

`python3 ../check_objects.py --recompute --only tab_s-cross-family` repeats the computation, requires the
same `values.json`, compares each value in `object.json` with the number printed in the paper at the printed
precision, and checks the three statements listed under `statements`.
