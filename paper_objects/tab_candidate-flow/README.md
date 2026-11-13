# Candidate flow

Labels: `sec:candidate-flow` (Section VI-A) with the identity `eq:candidate-flow` (Eq. 3, Section III-C), and
supplementary `sec:s-candidate-flow` with `tab:s-candidate-flow`, and the first paragraph of
`sec:s-output-interactions` (the extractor--vote interaction under three tie rules and its
corrections and regressions). Table `tab:s-ties-expanded` belongs to another object.

## What is printed

On 198 sampled BIRD development questions with two repeats, the with-peer debates of Du-MAD, MAD-M²
and BCCS with Llama-3-8B or Qwen2.5-7B agents (six configurations) are followed from the initial
pool to the final answer. Section VI-A prints the six-configuration means: the share of questions whose
initial pool holds a correct query ($C_0$, 34.76%) and whose last-round pool holds one ($C_T$,
32.87%); the shares on which revision loses all correct candidates (4.17%) or acquires one (2.27%);
the share on which the protocol extractor delivers a correct query ($C_D$, 26.43%); the baseline vote
($Y$, 21.13%); and lenient extraction with execution voting ($Y_x$, 29.12%, about 8 pp above $Y$).
The supplementary table gives the same quantities for each protocol and backbone, and its text gives
BCCS's actual returned-answer accuracies (9.60% and 23.99%). The supplementary paragraph on joint
output changes gives the extractor--vote interaction of Table `tab:factorial` under first-, last- and
random-agent ties, for Du-MAD and MAD-M² and including BCCS, and the runs that the joint change and the
change to last-agent ties correct and regress.

## Inputs

`inputs/candidate_flow_runs.csv`, 2,376 rows: one with-peer run (question, database, protocol,
backbone, repeat), 396 runs per configuration.

| Column | Meaning |
|---|---|
| `C0` | 1 if some agent's initial answer, read with the lenient extractor, is correct |
| `CT` | 1 if some agent's last-round answer, read with the lenient extractor, is correct |
| `CD` | 1 if some last-round answer read by the protocol extractor is correct |
| `final` | 1 if the protocol's final answer is correct |
| `{protocol,lenient}_{exact,execution}_{first,last,random}` | correctness of the last-round vote under each extractor, vote and tie rule |

The lenient extractor takes the protocol's answer when it finds one, else the last fenced SQL block,
else the last line that starts a query. The exact-match vote groups identical query strings; the
execution vote drops failing queries and groups the others by their result. A tie goes to the first
agent, the last agent, or a seeded random agent. A query is correct when its result equals the
reference result. BIRD questions 518 and 701, whose reference query exceeds the evaluation time limit,
are not among the 198. The vote columns are the per-run scores of `../tab_factorial/`; the table was
derived once from the saved debate responses and their stored execution results.

## Command

```sh
python3 code/compute.py --out output
```

Writes `output/values.json` and `output/candidate_flow.csv` (the supplementary table, two decimals).
Runs in about two seconds.

## Definitions

- $Y$ is the vote with the protocol extractor, exact-match voting and first-agent ties; $Y_x$ uses the
  lenient extractor and execution voting. For BCCS, $Y$ is the vote recomputed on the terminal
  responses; its actual return also depends on stopping and is reported separately.
- Loss: the initial pool holds a correct answer and the last-round pool holds none. Recovery: the
  reverse. Both are unconditional shares of runs.
- Per run, $Y\le C_D\le C_T$, and $1-Y=(1-C_0)+(C_0-C_T)+(C_T-C_D)+(C_D-Y)$ holds per question in every
  configuration; both are checked as statements.
- The extractor--vote interaction at a tie rule is $(Q_{11}-Q_{01})-(Q_{10}-Q_{00})$ with every
  reading at that tie rule. Corrections are runs wrong under the baseline configuration and correct
  under the changed rule; regressions are the reverse.

## Estimates

Repeats are averaged within a question; the four- and six-configuration values are equal-weight means
over configurations per question. Intervals of the interaction are 95% percentile bootstrap intervals
over the 198 questions (10,000 draws, seed 20261106) and over the 11 BIRD databases (10,000 draws,
seed 20261106); the interaction is detected when both intervals exclude 0. The question and
database intervals are in `values.json` (`interaction_*_q_lo`, `_q_hi`, `_db_lo`, `_db_hi`).

## Check

`python3 ../check_objects.py --recompute --only tab_candidate-flow` repeats the computation, requires
the same `values.json`, compares each value in `object.json` with the number printed in the paper source
at the printed precision, and checks the statements: the flow identity and ordering, the two gaps
larger than the net revision loss, two repeats, two regressions, and which interactions are detected.
