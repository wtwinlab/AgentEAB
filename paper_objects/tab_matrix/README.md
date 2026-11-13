# BIRD debate populations: the 198 sampled and the 1,531 development questions

**Labels.** The population paragraph of Section IV (`sec:setup`) and the BIRD debate rows of Table
`tab:matrix` (Table II).

## What is printed

- The sampled debate population: 198 of 200 questions drawn from 11 BIRD databases are scored (two
  reference queries exceed the 5-second limit); 68 simple, 68 moderate, and 62 challenging questions,
  over-representing challenging questions about 3.3 times relative to the development set.
- The full-development population: 1,531 of the 1,534 development questions are scored; the SQL candidate
  pools cover all 1,534.
- Table II repeats the 198 sampled and the 1,531 full-development questions.

## Inputs

- `inputs/rows.jsonl`, 38,592 rows. One row is one debate run: `source`, `unit` (BIRD question),
  `protocol`, `backbone`, `arm` (condition), `repeat`, `db`, `difficulty`, and the correctness of the
  execution vote over the initial answers (`r0`) and over the last-round answers (`last`).
- `inputs/question_lists.json`, `inputs/question_lists_e4c.json`, `inputs/eval_meta.json`: the 198 sampled
  questions and the two drawn questions whose reference query exceeds the time limit, the 1,333 remaining
  questions in three blocks, and the three development questions whose reference query fails
  (`bird|518`, `bird|701`, `bird|646`).
- `inputs/per_run_scores.jsonl`, 5,544 rows: the per-run scores of the 198 sampled questions, one row per
  run and every extractor / vote / tie combination.
- `inputs/bird_dev_questions.csv`, 1,534 rows: question, database and difficulty of every BIRD development
  question, written from BIRD's `dev.json`.

## Command

    python3 code/compute.py --out output

`code/estimators.py` holds the estimators. The scored questions are those present in every Du-MAD and
MAD-M² configuration; the over-representation is the challenging share of the sample divided by the
challenging share of the development set.

## Outputs

`output/values.json` (every checked value) and `output/fulldev_effects.json` (the four Du-MAD and MAD-M²
configurations on the 1,531 questions with both 95% intervals; the printed peer effects on these questions
are checked in `tab_peer-controls`).

## Check

`object.json` lists 11 printed values. `check_objects.py` confirms that each printed form occurs in its
paper file and that the computed value rounds to it.
