# Generation minus voting by generator and voting rule

Labels: `fig:sql-family` (Fig. 7) with the first paragraphs of `sec:sql-feedback` (Section VIII-A), and the
supplementary table `tab:s-family-sql` (Table S33, Section S8-A), which lists the plotted contrasts with
their Holm decisions.

## What is printed

- The figure and the table: generation by Qwen3-32B, gpt-oss-120b and Gemma-4-26B minus the
  Unfiltered, Valid and Nonempty votes of the four-candidate pool, on BIRD and Spider, with 95%
  intervals over databases (18 contrasts). A star in the table marks a contrast that passes Holm
  correction over the 18 contrasts jointly.
- In the text: on Spider, Gemma's 3.44-pp gain over Nonempty is the largest point estimate; execution
  feedback raises Qwen2.5-7B generation accuracy by 3.66 pp over the same candidate queries without it,
  on 1,532 paired BIRD questions.

## Inputs

| File | Rows | One row is |
|---|---|---|
| `inputs/crossfamily_generation_selection_per_question.csv` | 23,112 | one (benchmark, generator, voting rule, question): database, generation correct, voted candidate correct, their difference |
| `inputs/bird_qwen25_7b_feedback_per_question.csv` | 1,534 | one BIRD question: database, whether Qwen2.5-7B has a generation in all four input conditions (no peers, candidate queries only, candidate queries with execution feedback, full responses), correctness with candidate queries only and with execution feedback |

The first table was derived once from the cross-family generation responses (collected with
`experiments/08_cross_family/code/`) and the logged candidate records of the four-model BIRD and
Spider pools, scoring generation and each voting rule on the same questions.

The second table was derived once from the logged Qwen2.5-7B generation responses, read with the
extractor of `experiments/06_cost_resources/code/analyze_sql.py` and scored with
`experiments/06_cost_resources/code/sql_common.py` (`execution_match`): a read-only connection to the BIRD development database,
a 5-second limit per statement, and result sets compared as sets of tuples. The step executed
4,393 statements (each distinct generated query once, and its reference query whenever the generated query executes); `compute.py`
opens no database.

## Command

```
python3 code/compute.py --out output                     # values.json and plotted_values.csv
python3 code/plot.py              # output/sql_family.pdf (matplotlib)
```

`output/plotted_values.csv` holds the values Fig. 7 of the paper draws.

## Estimators

- Cross-family contrasts: each database weighted equally, the mean of per-database mean
  differences. The 95% interval resamples the 11 BIRD or 20 Spider databases, 10,000 draws of
  numpy `default_rng(20261028)` per contrast; the bootstrap p-value is twice the smaller tail share
  at zero, floored at 1/10,000. Holm at 0.05 is applied over the 18 contrasts jointly (keys ending
  `_holm_joint`), and over the two families of 15 and 3 contrasts (`_holm_fifteen`,
  `_holm_three`); the Gemma-on-Spider contrasts against Valid and Nonempty pass the family of three
  but not the joint family.
- Execution feedback: paired by question over the 1,532 BIRD questions with all four input
  conditions, each question weighted equally. The interval in `values.json` (not printed in the
  paper) resamples databases with 4,000 draws of Python `random.Random(20261013)`.
- Generation accuracy: each database weighted equally.

## What the check compares

`check_objects.py` finds each printed value in the paper and compares it with
`output/values.json` at the printed precision. Every printed value of the object reproduces.

## Scope

One generation per question for each generator, on the four-candidate pool; intervals are over
databases given that pool.
