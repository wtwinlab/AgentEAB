# Allocating a fixed budget: debate against voting over independent drafts

Labels: `sec:debate-budget` with Fig. `fig:debate-budget` (Section VIII-E) and supplementary `sec:s-budget`
with `tab:s-budget-cost`, `tab:s-budget-contrasts` and `tab:s-draft-diversity`.

## What is printed

Two comparisons on BIRD with Du-MAD and MAD-M² and Llama-3-8B or Qwen2.5-7B agents (four cells), 198
questions and two debate repeats.

- **Fixed responses.** A vote over the six initial answers of the two repeats against the six-call debate
  (3 agents, 2 rounds), averaged over the four cells: against the protocols' final answers, under lenient
  extraction and execution voting on both sides, and under protocol extraction and exact-match voting on
  both sides.
- **Budget sweep.** Debates with 3 agents over 2, 3 or 4 rounds and with 5 agents over 2 rounds (6, 9, 12
  and 10 calls), against a vote over the first k of 20 independent initial drafts that use the protocol's
  initial prompt and decoding (Du-MAD greedy, MAD-M² temperature 1); for Du-MAD also 20 drafts at
  temperature 0.7. Both sides use lenient extraction and execution voting with first-agent ties. The
  section prints the gain of extra rounds or agents over the six-call debate, call-matched (k = calls) and
  token-matched (k = the smallest draft count whose mean tokens reach the debate's mean) differences, the
  vote gain from one to 20 drafts, and the failed runs. The supplementary tables give each configuration's
  calls, tokens per run and failed runs; the MAD-M² call- and token-matched differences with both
  intervals; and draft diversity with the vote gain.
- **Fig. `fig:debate-budget`** is drawn in LaTeX. It plots MAD-M² accuracy at 6, 9, 10 and 12 calls, for the
  debate's last-round vote and for the vote over as many drafts, per backbone. The figure places each
  accuracy, rounded to one decimal, at y = 1.2 × (accuracy − 20) and x = 5 × (calls − 6); the check
  compares the 16 y coordinates in the source with that transform of the computed accuracies.

## Inputs

| file | rows | one row |
|---|---|---|
| `fixed_response_drafts.csv` | 4,752 | one initial answer of a with-peers run (cell, question, draft 0–5: repeat 1 agents, then repeat 2): execution-result label under the lenient extractor and string label under the protocol's extractor (integers within the question, empty when there is none), and the correctness of each query |
| `fixed_response_runs.csv` | 1,584 | one with-peers run (cell, question, database, repeat): the protocol's final answer and the last-round execution vote under the lenient extractor (0/1) |
| `debate_runs.csv` | 6,336 | one run of the budget collection (protocol, backbone, configuration agents×rounds, question, database, repeat): status, calls, prompt and completion tokens, last-round execution vote and final answer (0/1); 117 Du-MAD runs carry status `failed` |
| `drafts.csv` | 23,760 | one independent initial draft (protocol, backbone, draft set `protocol` or `t0.7`, question, draft 0–19): execution-result label within the question (empty when the query does not execute), correctness, prompt and completion tokens |

The fixed-response tables were derived once from the BIRD debate records of
`experiments/02_controlled_sharing/` (the collection behind Table `tab:factorial`) with the cached
execution results and verdicts; their final answers and votes equal the per-run scores of
`tab_factorial`. The budget tables come from the collection of `experiments/17_budget_matched_debates/`
(`bm_score.py`, then `bm_tables.py`). A query is correct when its normalised result rows equal those of
the reference query; an unparsed draft counts as wrong.

## Command

```sh
python3 code/compute.py --out output
```

NumPy only; about 1 s. Writes `output/values.json`, `output/plotted_values.csv` (the 16 plotted points),
`output/vote_curves.csv` (vote accuracy over the first k drafts, k = 1..20, and its mean over 20 draft
orderings, with mean tokens) and `output/contrasts.csv` (every contrast with both intervals).

## Estimates

- A vote returns the most frequent execution result (or string) among the drafts and ties go to the
  first tied draft. Repeats are averaged within a question; cell means are averaged per question.
- Fixed responses: percentile intervals over questions and over the 11 databases (question-weighted),
  10,000 draws, seed 20261109, each from its own generator.
- Budget sweep: one generator (seed 20261109) draws 10,000 multinomial weights over questions and then
  10,000 over databases; every contrast uses these weights. A failed run is left out and a question stays
  when one repeat succeeds (155 questions for the Du-MAD four-round mean); the sensitivity counts a
  failed run as wrong.
- Token matching uses mean prompt-plus-completion tokens per run; when 20 drafts do not reach the
  debate's tokens (the larger Du-MAD debates), the comparison uses 20 drafts.
- Detected: both 95% intervals exclude 0.

## Scope

The context limit (8,192 tokens), the decoding temperatures and the numbers of agents and rounds are
settings of the collection and are not computed here. No model call and no database access.

## Check

`python3 ../check_objects.py --recompute --only fig_debate-budget` repeats the computation, requires the
same `values.json`, compares every value in `object.json` with the paper source at the printed precision
(both ends of printed ranges such as "3.91--5.56" separately), and checks the detection and count
statements.
