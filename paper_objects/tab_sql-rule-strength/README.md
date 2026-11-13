# Execution filters on fixed SQL pools

Labels: `tab:sql-rule-strength` (Table VIII) and the prose of `sec:sql-final-selection` (Section VII-B);
supplementary `tab:s-q25-rules` (Table S27) with the prose of `sec:s-sql-rules` (S7-C), and `tab:s-empty`
(Table S29), `tab:s-testsuite` (Table S30) with the prose of `sec:s-exec` (S7-D). The execution-cost table `tab:s-exec-cost` is a
separate object.

## What is printed

- Qwen2.5-7B and Qwen2.5-32B generation minus each selection rule (Unfiltered, Valid, Nonempty)
  on the BIRD and Spider pools of four and six candidates, with 95% intervals over databases, and
  Nonempty minus Valid on Spider (Table `tab:sql-rule-strength`, Table `tab:s-q25-rules`).
- The voting-rule facts of Section VII-B: how often Unfiltered returns an unexecutable query,
  how often the pool already holds a correct query where generation succeeds and Unfiltered fails,
  Nonempty against Valid on nonempty-answer and empty-answer questions, the empty-answer rate,
  and the prevalence at which Nonempty and Valid break even.
- Execution-vote winners that return an empty result, in the pools and in the initial answers of
  the Llama-3-8B and Qwen2.5-7B BIRD debates (Table `tab:s-empty`).
- Spider accuracy on the single development database and on the test-suite databases, and the
  differences between systems under both scorings (Table `tab:s-testsuite`).
- The Qwen2.5-72B recipient on the four-candidate pool against Valid and against Qwen2.5-32B.

## Inputs

| File | Rows | One row is |
|---|---|---|
| `inputs/qwen25_pools_per_question.csv` | 5,133 | one (benchmark, pool, question): database, reference stratum (nonempty, empty, unknown), whether any candidate is correct, correctness of the Unfiltered, Valid and Nonempty winners, whether the Unfiltered winner fails to execute, whether each winner returns no rows |
| `inputs/qwen25_pools_per_execution.csv` | 30,798 | one generation run: benchmark, pool, recipient (Qwen2.5-7B or Qwen2.5-32B), database, question, run index, generation correct |
| `inputs/qwen25_72b_generation_per_question.csv` | 2,568 | one question: Qwen2.5-72B generation correct on the four-candidate pool (one run) |
| `inputs/debate_initial_vote_winners.csv` | 5,544 | one debate (question, protocol, backbone, with or without peers, repeat): whether any executable initial answer exists, whether the execution-vote winner returns no rows, whether it is correct |
| `inputs/spider_test_suite_per_question.csv` | 1,034 | one Spider question: number of test-suite databases, and for every system its correctness on the single development database and on all test-suite databases; Qwen2.5 generation as the number of correct runs out of six (two recipients, three runs each); per pool the number of result-equal vote groups with more than one SQL text and how many of them split on the test suite |

The BIRD pools cover the 1,533 (four candidates) and 1,532 (six candidates) questions with all three
generation runs; Spider covers 1,034 questions. The debate table holds 198 BIRD questions, two repeats each, with and without peers, for six
protocol and backbone settings and the MoA references (792 rows each); BIRD questions 518 and
701, whose reference query exceeds the 5-second limit, are excluded.

The selection rules (Unfiltered, Valid, Nonempty) are applied to the candidates of the four- and six-model
pools and their logged executions; the per-question and per-execution pool tables were derived once from those
records. The debate table was derived once from the logged debate runs with the lenient extractor and the
execution vote over the initial answers. The test-suite table was derived once from stored executions of every
query, run read-only on the 25 to 60 test-suite databases of its question. No step here opens a database.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` in a few seconds. Keys ending in `_lo` and `_hi` are interval ends;
keys ending in `_detected` are 1 when both the question and the database 95% intervals exclude 0.

## Estimators

- Pool contrasts: each generation run of a question is one observation, and the deterministic
  selection outcome of the question is repeated for each run, so questions carry equal weight.
  Intervals resample the BIRD (11) or Spider (20) databases: 5,000 draws of Python
  `random.Random(20261026)` over the databases in sorted order, the 2.5th and 97.5th order
  statistics of the resampled means.
- Break-even prevalence: with `g` the Nonempty gain on nonempty-answer questions and `l` its loss
  on empty-answer questions, Nonempty minus Valid at empty-answer prevalence `p` is `(1-p)g - pl`,
  zero at `p = g/(g+l)`.
- Empty winners: one decision per (benchmark, pool, question) in the pools. In the debates, shares
  of questions average the runs of a question first; shares of wrong winners pool the runs.
- Test suites: each question weighted equally; a query is correct on the suite only if its result
  set equals the reference's on every test-suite database of the question. Detection flags use
  10,000 draws of numpy `default_rng(20261106)` over questions and over the 20 databases.
- Qwen2.5-72B: per question, its single run against the deterministic Valid outcome and against
  the mean of the three Qwen2.5-32B runs.

## What the check compares

`check_objects.py` finds each printed value in the paper and compares it with
`output/values.json` at the printed precision. Both ends of printed ranges such as
"7.18--7.48" are checked as separate values.
`unreproduced` lists the printed values that do not round from the computed value, with the reason. One
value is listed: "19.0%" of BIRD questions on which Unfiltered returns an unexecutable query with six
candidates. 292 of the 1,532 compared questions give 19.06%, which rounds to 19.1; 292 of all 1,534 pool
questions give 19.04%, the printed 19.0.

## Scope

The intervals are over databases given the fixed candidate pools; the pool contrasts hold for
these Qwen2.5 recipients and pools. The test-suite reading compares result sets as sets of
tuples, the single-database notion of execution accuracy used throughout the paper, on every
test-suite database.
