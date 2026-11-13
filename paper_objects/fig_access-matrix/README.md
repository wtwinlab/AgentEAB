# Sharing effect on the protocol's final answer

**Object.** Fig. `fig:access-matrix` (Fig. 3, Section V-A, `figures/access_matrix.tex`), the numbers
of the paragraphs that discuss it (`sec:q1-matrix`), the range repeated in Supplementary Section S5
(`sec:s-sharing`) and row R23 of Table S1 (`tab:s-contrasts`), and Supplementary Section S5-A
(`sec:s-ford`: FORD's stopping rule and exhausted ties).

**What is shown.** A table of shaded cells, one row per benchmark and one column per configuration:
for FORD and MoA (mixed-model rosters) and for Du-MAD, MAD-M², and BCCS with
Llama-3-8B (L8) or Qwen2.5-7B (Q7) agents, on GSM8K, StrategyQA, MMLU-Pro, GPQA, WDC, Settlements, and
BIRD: the protocol's final answer with peers minus without, in pp (pair F1 for matching, accuracy
elsewhere); unanswered tasks count as wrong. A dagger marks a sign that depends on how unreadable
replies are counted; `0.0§` marks Du-MAD/L8 on BIRD, which delivers no answer in either condition;
BCCS on WDC and Settlements has no estimate.

**Inputs** (`inputs/`):

| file | rows | one row |
|---|---|---|
| `rq_effects.csv` | 322 | one paired effect (benchmark, protocol, backbone, contrast, dimension, scoring convention) with its interval and delivery counts |
| `t3_contrasts.csv` | 138 | one matching effect re-scored with corrected labels, with its interval |
| `u4_residue.csv` | 46 | one matching cell: range of the effect over all readings of replies the extractor cannot resolve |
| `ford_stop_firing.csv` | 63 | one (benchmark, condition, repeat): runs and runs on which FORD's stopping rule fires |
| `ford_tie_identity.json` | 6 | one (matching benchmark, condition): FORD runs, runs on which the rule fires, tied runs |
| `matching_return_rule.csv` | 40 | one (protocol, benchmark, shared content): sharing effect under the protocol's return rule and under the all-round majority with each tie rule |
| `ford_wdc_runs.csv` | 4,000 | one FORD run on a WDC Products pair, with or without peers (`pair`, `repeat`): reference label, final label, `tied` (the trace holds as many "match" as "non-match" predictions), calls, output tokens |
| `ford_gsm8k_runs.csv` | 400 | one FORD run on a GSM8K task, with or without peers: calls, input and output tokens |

The tables were derived once from the logged runs by the collection's scoring code. The no-peer runs
of Du-MAD and BCCS behind `rq_effects.csv` and `matching_return_rule.csv` are the no-peer runs whose
prompts contain no peer content.

The two FORD run tables check the numbers of `sec:s-ford` and Section V-A: runs are paired by pair
and repeat and split by whether the no-peer run ends in a tie; the 333 tied runs account for 7.25 of
the 7.35-pp accuracy gain, the 1,667 untied runs gain 0.50 pp of F1, and mean calls fall from 6.00 to
2.23 in the tied runs and from 2.01 to 2.00 elsewhere. On GSM8K, sharing saves about 2.7 calls and
389 output tokens per task.

**Commands.**

    python3 code/compute.py --out output     # values.json, plotted_values.csv
    python3 code/plot.py --out output         # access_matrix.tex (standard library)

**Outputs.** `output/values.json`, `output/plotted_values.csv` (56 cells, 52 with an estimate),
`output/access_matrix.tex`. `plot.py` writes the LaTeX table that the paper inputs as Fig. 3: each cell
prints the effect to one decimal on a box shaded at 4/3 of its magnitude.

**What the check compares.** The printed values in `object.json` against `values.json`: the 52 cells
of Fig. 3 at their printed precision, and the prose values of Section V-A and the supplement.
