# What peers share: field comparisons in matching, and QA debate traces

Labels: `sec:matching-content` (Section V-C, the matching numbers), `sec:qa-candidate-traces`
(Section VI-E), Supplementary `sec:s-content` (Section S5-C) with `tab:s-confusion` (Table S8) and
`tab:s-content-choice` (Table S9), `sec:s-qa-traces` (Section S6-I) with `tab:s-traces` (Table S23),
and the shared-content and QA-trace rows of `tab:matrix` (Table II).

## What is printed

- Matching: Qwen3-32B, gpt-oss-120b and Gemma-4-26B-A4B judge candidate pairs with no peers, with
  a fixed Qwen2.5-7B peer decision, with that decision and schema-defined field comparisons, or
  with the peer's full explanation. The text prints the Amazon–Google test F1 drop from the
  decision to the decision with field comparisons (0.98–9.94 pp over the three recipients), Qwen3's paired changes against no peers and
  how many of them follow the shown decision, Qwen3's confusion counts, test F1 for every input on
  DBLP–ACM (800 test pairs) and Amazon–Google (1,000), the losses development selection avoids on
  Amazon–Google, and the DBLP–ACM gaps of the selected input to the best tested input.
- QA debate traces: coverage of the 10,900 planned runs of eight protocol variants in nine
  collections, and for runs that start with a correct candidate whether it is lost before final
  selection (Loss) or still present when a wrong answer is returned (Miss); Section VI-E repeats
  the run counts (7,044 runs start with a correct candidate, 1,470 of them fail; 2,224 runs start
  with only wrong candidates). Table II gives the 1,800 shared-content test pairs and the 10,295
  traces.

## Inputs

- `inputs/amazon_google_pairs.csv` (18,000 rows) and `inputs/dblp_acm_pairs.csv` (15,600 rows):
  one recipient × split (development, test) × input × candidate pair, with the gold label, the
  recipient's decision (empty when unreadable; counted as a non-match), the shown peer decision
  (Amazon–Google) and prompt tokens. Decisions are read by `read_decision` in
  `experiments/07_content_choice/code/e2_score.py` and `experiments/08_cross_family/code/mt2_score.py`.
- `inputs/qa_trace_runs.csv` (10,900 rows): one planned QA debate run with its protocol variant,
  whether its trace is missing, whether a correct candidate is present initially, in the final pool
  and at any stage (`true`, `false`, `unknown`), final success and endpoint state (success, wrong,
  recorded no output, unscorable), derived once from the logged matching traces.

The three tables were derived once from the logged runs.

## Command

```sh
python3 code/compute.py --out output
```

Writes `output/values.json`, `output/matching_levels.csv` (development and test F1, counts and the
selected input) and `output/traces.csv` (per-variant counts).

## Definitions

- Development pair F1 selects one input per recipient; ties go to fewer mean prompt tokens, then
  to the input name.
- Loss = runs that lose every correct candidate / (runs starting with a correct candidate minus
  those whose final availability is undetermined); the reading bounds use all runs starting
  correct as denominator, without and with the undetermined runs in the numerator.
- Miss = (wrong + no-output endpoints) / runs with a correct candidate in the final pool and a
  scorable answer; the second form counts unscorable answers as failures.

## Check

`python3 ../check_objects.py --recompute --only text_matching-content` compares the listed values
with Sections IV, V-C and VI-E and the supplement. Three F1
cells of Tables S8 and S9 are computed from the pair counts (for example 68.63 = 2·279/(558+34+221));
every printed value of the object reproduces.
Both ends of the range "7.01--10.92%" are checked; the word "one" is under `not_checked` and is computed
as 1.
