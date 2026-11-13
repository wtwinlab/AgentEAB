# tab_scale: peer interaction effect across Qwen2.5 backbone sizes on BIRD

Labels covered: `tab:scale` (Table X) and the paragraph that precedes it in Section VIII-D (`sec:scale`), and
supplementary `tab:s-scale-contrasts` (Table S40) with the equivalence cell described in `sec:s-spend-scale`
(Section S8-C).

## What is printed

- `tab:scale`: the peer interaction effect of Du-MAD, MAD-M², their mean, and BCCS with Qwen2.5-7B, 14B, 32B
  and 72B agents (pp). The peer interaction effect is the change from the initial answers to the last round
  with peers minus the same change without peers, per question, repeats averaged, paired by question.
- Section VIII-D: the scale contrast (72B minus 7B on the Du-MAD and MAD-M² mean, +0.51 pp) with the upper end
  of its 95% interval over questions (+3.41 pp), and the 72B minus 14B contrast (+3.79 pp).
- `sec:s-spend-scale`: MAD-M² at 72B with its 90% interval against a ±2 pp equivalence margin, and its two
  repeats.
- `tab:s-scale-contrasts`: every contrast of the per-question Du-MAD and MAD-M² mean, larger minus smaller
  model, with 95% intervals over questions and over the 11 BIRD databases, the two-sided bootstrap p, and the
  Holm-adjusted p over the five Qwen2.5 contrasts; the gpt-oss and Llama contrasts sit outside that family.

All values use the lenient extractor, the execution vote and ties to the first agent on 198 BIRD questions
(questions 518 and 701 are excluded because their reference query exceeds the time limit).

## Inputs

`inputs/debate_runs.csv`, 36,581 rows. One row is one debate run: `model`, `protocol` (`du_mad`,
`madm2_objective`, `bccs`), `condition` (`with_peers`, `no_peers`), `question`, `database`, `repeat`, and
`initial_correct` / `last_correct` (1 if the execution vote over the initial answers, or over the last-round
answers, returns a query whose result matches the reference). The file covers the 16 models of the scale and
cross-family debates, two repeats per condition, on the 200 sampled BIRD questions.

The extractor and the vote are those of the paper (lenient extraction, execution signature with failing
queries dropped, first-agent ties); the per-run table was derived once from the logged runs.
The no-peer rows of Du-MAD and BCCS are those in `inputs/debate_runs.csv`; their no-peer prompts contain
no peer content.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` (every checked value), `output/scale_table.csv` (each protocol and size with both
intervals) and `output/scale_contrasts.csv`. Runs in a few seconds.

## Estimation

Bootstrap with 10,000 draws and seed 20261106, once over questions and once over the 11 databases (all
questions of a drawn database enter); percentile intervals. The p-value is twice the smaller share of draws
on either side of zero, at least 1/10,000. Holm correction runs over the five Qwen2.5 contrasts. The 7B point
uses fp16 weights and the other Qwen2.5 points AWQ.

## Check

`python3 ../check_objects.py --recompute --only tab_scale` repeats the computation, requires the same
`values.json`, and compares each value in `object.json` with the number printed in the paper at the
printed precision.
