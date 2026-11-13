# How often alternative output rules change the conclusion

**Object.** Table `tab:sensitivity-census` (Table S31, Supplementary S7-G, `sec:s-census`) and its
prose: alternative output rules change the conclusion in 24 of 98 eligible cases, and all 27 cases
whose extractor finds every answer stay stable.

**What is printed.** For each family of cases, the number of cases, how many flip their sign (Sign),
how many flip whether their 95% interval excludes zero (Detect.), and how many do neither (Stable).
On BIRD the baseline is the protocol's final answer or its own extractor with exact-match voting, and
the alternative uses the lenient extractor with execution voting; on matching and QA both sides are
re-read leniently. The return-rule rows replace the protocol's return rule by the all-round majority
with ties to the first agent, to the last agent, or kept; the last row compares the sharing effect on
pair F1 with that on cluster F1.

**Inputs** (`inputs/`):

| file | rows | one row |
|---|---|---|
| `census_cases.json` | 223 | one case: family, cell, baseline (`naive`) and alternative (`controlled`) estimate with 95% intervals, whether the baseline estimate is defined, and (matching and QA) whether the protocol's extractor finds every answer |
| `grouping_sharing_effects.csv` | 990 | one sharing–grouping combination on the test split: sharing effect on pair F1 and on cluster F1 (pp) |

The BIRD cases use the same per-run scores as `tab_factorial`; the no-peer records of Du-MAD and BCCS
are those whose no-peer prompts contain no peer content. The Du-MAD/Llama-3-8B configuration, whose
extractor delivers nothing, has an undefined baseline estimate in the first two families and is not
eligible.

**Command.**

    python3 code/compute.py --out output

**Outputs.** `output/values.json`: per family, cases, sign flips, detection flips, and stable cases;
the totals over the four output-rule families; the grouping counts.

**What the check compares.** Every cell of the table and the three prose numbers against
`values.json`.
