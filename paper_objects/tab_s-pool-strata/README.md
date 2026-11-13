# Generation gains by correct-candidate availability across candidate pools

Labels: `sec:pool-availability` (Section VIII-B) and supplementary `sec:s-pool-strata` with
`tab:s-pool-strata`.

## What is printed

Nine BIRD generator–pool comparisons of generation against the Nonempty vote: DeepSeek-V4-Pro, Qwen3-32B,
Gemma-4-26B and gpt-oss-120b on the four-model pool; Qwen3-32B on the Qwen3/GLM pool; Qwen2.5-7B and
Qwen2.5-32B on the four-model pool and on six candidates (the four-model pool plus Mistral-7B and
Qwen2.5-72B). Each generation-minus-Nonempty effect is split into the contributions of questions whose
pool holds a correct candidate and of questions whose pool holds none. Section VIII-B prints the DeepSeek
split (8.16 of 8.87 pp, +0.72 pp where a correct candidate exists, −1.51 pp against the DeepSeek
selector there) and the ranges over the nine comparisons; the supplement prints the table, the question
counts, and the DeepSeek gain without the two questions that lack a completed reference result.

## Inputs

`inputs/pool_strata_questions.csv`, 13,800 rows; one row is one BIRD question in one comparison:

| Column | Meaning |
|---|---|
| `recipient`, `pool` | generator and candidate pool (`four_model`, `qwen3_glm`, `six_candidates`) |
| `question_id`, `db_id` | the question and its database |
| `has_reference` | 0 for the two questions (518, 701) whose reference query does not finish within 5 s |
| `correct_candidate` | 1 if some candidate returns the reference result |
| `nonempty` | correctness of the Nonempty vote (0/1) |
| `generation_correct`, `generation_runs` | correct generations out of all generations (three for Qwen2.5, one otherwise) |
| `selector` | correctness of the LLM selector given the same inputs (DeepSeek-V4-Pro and Qwen3-32B rows) |

The rows were taken once from the per-question inputs of `tab_strength` (four-model and Qwen3/GLM pools,
DeepSeek-V4-Pro), `tab_sql-rule-strength` (Qwen2.5 pools and generations) and `fig_sql-family` (Gemma-4-26B
and gpt-oss-120b against Nonempty); questions without all three Qwen2.5 generations are not in those
inputs, so the Qwen2.5 comparisons keep 1,533 (four-model) and 1,532 (six candidates) questions. The two
questions without a reference result count as lacking a correct candidate, with every method wrong.

## Command

```sh
python3 code/compute.py --out output
```

NumPy only; a few seconds. Writes `output/values.json` and `output/strata.csv` (each comparison's total
and two contributions with both intervals).

## Estimates

- Each database is weighted equally. A contribution averages each question's paired difference times the
  stratum indicator, so the two contributions add up to the total; they are not conditional accuracies.
- Intervals: database-cluster bootstrap (10,000 draws, seed 20261105) and question bootstrap of the same
  database-weighted mean (10,000 draws, seed 20261109); each estimate draws from a fresh generator.
  Detected: both 95% intervals exclude 0.

## Check

`python3 ../check_objects.py --recompute --only tab_s-pool-strata` repeats the computation, requires the
same `values.json`, compares every value in `object.json` with the paper source at the printed precision,
and checks the detection and count statements.
