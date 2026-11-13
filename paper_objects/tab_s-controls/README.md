# tab_s-controls: controls on the four Du-MAD and MAD-M² configurations

Labels covered: Supplementary Table `tab:s-controls` (Table S17) and the text of `sec:s-controls`
(Section S6-E); the control paragraphs of Section VI-C (`sec:peer-controls`: unified answer format,
placebo peers, temperature 0.7).

## What is printed

Four configurations: Du-MAD and MAD-M², each with Llama-3-8B or Qwen2.5-7B agents, two repeats per question and
condition. All values use the lenient extractor, the execution vote (queries that fail to execute cast no vote)
and ties to the first agent, paired per question, with 95% intervals over questions and over the 11 BIRD
databases. A contrast is detected when both intervals exclude 0.

- Interaction effect on the 1,333 remaining BIRD dev questions and on all 1,531, per protocol, per configuration, and
  by difficulty (924 simple, 463 moderate, 144 challenging questions).
- Last-round difference (last-round accuracy with peers minus without) on the 1,531 questions, per protocol
  and per configuration; the initial-answer difference; on the 198 sampled questions the four-configuration last-round
  difference and Du-MAD's interaction effect.
- Placebo peers (replies to other questions of the same database): real minus placebo and placebo minus no
  peers, on the change from the initial answers to the last round, four-configuration mean and per configuration.
- Du-MAD at temperature 0.7: interaction effect, the same on the questions whose initial drafts disagree,
  and the change in the share of runs whose initial queries return two or more distinct execution results.
- Unified SQL answer format: change in final-answer accuracy under the protocols' own extractors (four configurations and
  per configuration), change in the interaction effect, and change in the share of initial replies the protocol's
  extractor finds.
- In the text: minimum detectable effects at 80% power; leave-one-database-out ranges on 1,531 and 198
  questions and the number of cases whose intervals exclude 0; the placebo round-1 prompt length ratio; the
  agreement of initial replies and initial-answer correctness between the two conditions of a paired run.

## Inputs

- `inputs/rows.jsonl`, 38,592 rows, one per debate run: `source`, `unit` (BIRD question), `protocol`,
  `backbone`, `arm` (condition), `repeat`, `db`, `difficulty`; correctness of the execution vote over the
  initial answers (`r0`) and the last round (`last`); the protocol's own final answer (`final`); `r0_diverse`
  (the initial queries return two or more distinct execution results); `r0_strict_share` (share of initial
  replies in which the protocol's extractor finds a query); `r1_prompt_tokens` and `n_calls`. Conditions:
  `FULL` (with peers), `NOPEERC` (Du-MAD without peers; its later-round prompts contain no peer text), `NOPEER` (MAD-M² without peers), `PLACEBO`, `T07_FULL` / `T07_NOPEERC` (Du-MAD at temperature
  0.7), `FMT_FULL` / `FMT_NOPEERC` / `FMT_NOPEER` (unified answer format). Du-MAD rows with condition
  `NOPEER`, `T07_NOPEER` or `FMT_NOPEER` are not read by any value here.
- `inputs/question_lists.json`, `inputs/question_lists_e4c.json`, `inputs/eval_meta.json`: question sets and
  the three excluded questions.
- `inputs/per_run_scores.jsonl`, 5,544 rows: per-run scores of the 198 sampled questions, key
  `lenient|exec|{r0,last}|first`; the later-round prompts of its Du-MAD no-peer runs contain no peer text.
- `inputs/round0_pairs.jsonl`, 30,672 rows, one per Du-MAD or MAD-M² run: sha256 prefixes of the three
  initial prompts and the three initial replies, derived once from the logged debate runs.
- `inputs/bird_dev_questions.csv`, 1,534 rows: question, database and difficulty (from BIRD's `dev.json`).

## Command

    python3 code/compute.py --out output

`code/estimators.py` holds the estimators: on the 1,531 questions and for the 198-question controls, 10,000
bootstrap draws (seed 20261106) over questions and, separately, over the 11 databases, repeats averaged
within a question, multi-configuration values on the questions present in every configuration; the 198-question values
and their leave-one-database-out and MDE values use the question and database bootstrap of `tab_factorial`. The minimum
detectable effect is (1.96 + 0.84) times the bootstrap standard error.

## Outputs

`output/values.json` (every checked value) and `output/controls.json` (each contrast with both intervals,
per-database leave-one-out results, and initial-answer agreement per configuration).

## What the check compares

`object.json` lists 239 printed values, both ends of printed ranges included; each must occur in its paper
file and the computed value must round to it. Every printed value of the object reproduces.
