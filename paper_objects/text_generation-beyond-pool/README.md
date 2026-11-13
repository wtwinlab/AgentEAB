# text_generation-beyond-pool: where generation's gain over voting comes from

Labels covered: the availability paragraph of supplementary `sec:s-spend-pool` (Section S8-A), which
splits DeepSeek-V4-Pro generation's gain over Nonempty voting by whether the pool holds a correct query.

## What is printed

On the four-model BIRD pool (1,534 questions, 11 databases), DeepSeek-V4-Pro generation beats Nonempty
voting by 8.87 pp. The supplement splits this gain by whether the pool holds a correct query:

- 557 questions whose pool holds no correct query contribute 8.16 pp; DeepSeek-V4-Pro generation answers
  118 of them (21.2%), Qwen3-32B generation 59 (10.6%), and voting and both selectors none;
- 977 questions whose pool holds a correct query contribute +0.72 pp relative to Nonempty (not detected);
  on them generation is 1.51 pp below the DeepSeek-V4-Pro selector with full inputs (detected);
- on Spider, generation newly answers 28 questions without a correct candidate (+1.91 pp), and losses on
  questions with one offset them, so the total is not detected.

## Inputs

`inputs/pool_questions.csv`, 2,568 rows (1,534 BIRD and 1,034 Spider questions). One row is one question
of the four-model pool (Qwen2.5-7B, Llama-3-8B, Qwen2.5-14B, Qwen2.5-32B candidates):

| Column | Meaning |
|---|---|
| `benchmark`, `question_id`, `db_id` | the question and its database |
| `Oracle` | 1 if at least one candidate returns the reference result |
| `n_correct_cands` | number of correct candidates (0 to 4) |
| `Valid`, `Nonempty` | 0/1 correctness of the execution votes |
| `Q3_S_al`, `Q3_G` | Qwen3-32B selector with full inputs (the generation prompt), and Qwen3-32B generation |
| `DS_S_al`, `DS_G` | the same for DeepSeek-V4-Pro |
| `*_G_copy_any`, `*_G_copy_correct` | 1 if the generated query equals a candidate (any / a correct one) after whitespace and case normalisation |

The correctness columns agree row by row with the inputs of `../tab_strength/` (pool `original`), so the
totals equal Table `tab:strength`. BIRD questions 518 and 701 have no reference result; they
count as Oracle = 0 with every method wrong, as in Table `tab:strength`.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` and `output/contributions.csv` (per contrast: total, the two stratum parts, their
intervals, questions gained and lost). Runs in a few seconds. No query is executed: a correct generated
query and a correct candidate return the same rows as the reference, so the split needs only the stored
correctness of each query.

## Estimation

Each database is weighted equally, as in Table `tab:strength`. The two stratum parts add up to the total:
each question's paired difference is multiplied by the stratum indicator and averaged with the same weights.
Intervals are 95% database-cluster bootstrap intervals (10,000 draws, seed 20261105); a part is detected
when its interval excludes 0.

## Check

`python3 ../check_objects.py --recompute --only text_generation-beyond-pool` repeats the computation,
requires the same `values.json`, compares each value in `object.json` with the number printed in the
supplement at the printed precision, and checks the detection statements listed under `statements`.
