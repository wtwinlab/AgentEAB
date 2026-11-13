# Selection and generation calls on a fixed SQL candidate pool

Paper object `tab_strength`. It covers:

- Table `tab:strength` (Table IX, Section VIII-A) and the paragraph that follows it in `sec:sql-feedback`;
- the numbers of Section VIII-C (candidate source, pool size, and acquisition cost): the cross-pool change
  of the generation--LLM-selection gap, the acquisition tokens of both pools, the selector's cost per
  decision relative to generation, and six against two Qwen3-32B samples per question;
- Supplementary tables `tab:s-did` (Table S32), `tab:s-aligned` (S34), `tab:s-repeat` (S35),
  `tab:s-repeat-within` (S36), `tab:s-availability` (S37), `tab:s-quality-cost` (S38) and
  `tab:s-selector-controls` (S5), with the numbers in the prose of `sec:s-spend-pool` (S8-A) and the
  selector-control paragraph of `sec:s-decoding` (S4-B).

The cross-family table `tab:s-family-sql` and Fig. `fig:sql-family` are in `../fig_sql-family/`. The split of
DeepSeek-V4-Pro generation's gain by whether the pool holds a correct query, as printed in the supplement, is
in `../text_generation-beyond-pool/`.

## What is printed

All comparisons hold the candidate pool and the prompts fixed and change one operation:

- Unfiltered, Valid and Nonempty execution votes (no LLM call);
- the Qwen3-32B selector given an execution summary, without reasoning (condition B) or with reasoning (C);
  condition A is the same selector with the candidates in a fixed order;
- the LLM selector with full inputs, i.e. the generation prompt (Qwen3-32B or DeepSeek-V4-Pro);
- generation: the recipient writes its own query (Qwen3-32B or DeepSeek-V4-Pro).

Accuracy weights each database equally. Intervals are 95% database-cluster bootstrap intervals (10,000 draws):
BIRD has 11 databases, Spider 20. Differences are paired by question. Holm correction runs within the families
the paper names: five contrasts per model and benchmark for `tab:strength`, six per pool for `tab:s-aligned`,
six per block for `tab:s-repeat-within`, eight for `tab:s-selector-controls`.

Populations: within-pool comparisons use all 1,534 BIRD and 1,034 Spider questions; cross-pool comparisons,
`tab:s-availability` and `tab:s-quality-cost` use the 1,532 BIRD questions with scored candidates in both pools.
"Decode 1" and "decode 2" are the two generation repeats of the same prompts; "mean" averages them per
question.

## Inputs

| File | Rows | One row is |
|---|---|---|
| `inputs/pool_operations_per_question.csv` | 5,136 | one question in one pool (four-model or Qwen3/GLM) and benchmark: 0/1 correctness of Oracle (a correct candidate is in the pool), Unfiltered (`Original`), Valid, Nonempty, B, C, the selector given the generation prompt (`S_al`, decode 1) and Qwen3-32B generation (decode 1), with `db_id` |
| `inputs/pool_decodes_per_question.csv` | 5,136 | the same questions with both decodes of the selector given the generation prompt and of generation, and their per-question mean |
| `inputs/deepseek_per_question.csv` | 2,568 | one question of the four-model pool: DeepSeek-V4-Pro selector (generation prompt) and generation correctness, one decode, with tokens |
| `inputs/same_source_samples_per_question.csv` | 15,408 | one question and one number k = 1..6 of Qwen3-32B samples: Oracle, Unfiltered, Valid, Nonempty and generation correctness on the first k samples |
| `inputs/selector_controls_per_question.csv` | 2,568 | one question of the four-model pool: correctness of selector conditions A, B, C, the votes and Qwen3-32B generation |
| `inputs/selector_controls_calls.csv` | 7,704 | one selector call (condition, question): candidates shown, chosen position, tokens, latency |
| `inputs/call_tokens.csv` | 25,680 | one LLM call (pool, benchmark, operation, decode, question): input, output and reasoning tokens |
| `inputs/candidate_acquisition.csv` | 4 | one pool and benchmark: total candidate calls and their input and output tokens |

The two `*_calls`/`call_tokens` tables are extracted once from the stored model records; the correctness of every candidate
and every generated query comes from executing it against the benchmark's development database, as defined in
the scoring code of the source analyses. `candidate_acquisition.csv` holds totals over all candidate calls of
each pool.

## Command

```
python3 code/compute.py --out output
```

Writes `output/values.json` (flat `{key: number}`, accuracies and differences in percent or pp, tokens per
question, USD per 1,000 questions, acquisition tokens in millions). Keys ending in `_holm` are Holm decisions
(1 = passes) and match the stars printed in `tab:s-aligned` and `tab:s-repeat-within` and the Holm statements in
the text; they are not compared by the check. Runs in under a second.

## What the check compares

`python3 ../check_objects.py --recompute --only tab_strength` runs `compute.py` again, compares the result with
`output/values.json`, and compares every entry of `object.json` `values` with the number printed in
Section VIII or the supplement, at the printed number of decimals.

Both ends of printed ranges (`1.8--1.9`, `1.3--1.5`, `1.56--4.59`, `0.08--4.06`) are checked as separate
values. Every printed value of the object reproduces, including the 40.3 s mean latency of condition C on
BIRD (unrounded 40.349 s).

## Scope the paper states

- One decode per question for DeepSeek-V4-Pro; Qwen3-32B values in `tab:strength` are the first decode.
- On Spider, accuracy uses single-instance execution on the development database.
- Prices are $0.08 and $0.28 per million input and output tokens; costs exclude candidate acquisition and
  execution.
- The two pools differ in model family, sampling and within-pool correlation together.
