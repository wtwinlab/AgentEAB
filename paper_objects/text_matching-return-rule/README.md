# Return rules on stored matching runs (Section VII-D, `sec:qa-vote-replay`)

Labels: `sec:qa-vote-replay` (Section VII-D), the sentence on matching, and row R20 of
supplementary `tab:s-contrasts` (Table S1).

## What is printed

On stored matching runs of Du-MAD, MAD-M², and BCCS (WDC and Settlements), replacing the
protocol's return rule by an all-round majority moves the sharing effects by up to 1.98 pp of pair
F1, depending on the tie convention, and the Du-MAD/Settlements effect changes sign (-0.14 pp under
the protocol's rule, +0.10 pp under the majority with ties to the last agent or kept).

## Inputs

| file | rows | one row is |
|---|---|---|
| `inputs/return_rule_runs.csv` | 72,000 | one run of a protocol on one matching unit (2 datasets x 3 protocols x 3 conditions x 1,000 units x 2 backbones x 2 repeats): the protocol's final answer and the all-round majority over every call's answer with ties to the first agent, to the last agent, or kept at the protocol's answer |
| `inputs/return_rule_units.csv` | 2,000 | a matching unit (candidate pair): reference label and the reference entities of its two records |

Conditions: FULL (full peer responses), NOPEER (no peers), ANSONLY (peer answers only). The
no-peer runs of BCCS are those used throughout the paper; Du-MAD and MAD-M² use the same runs as the
other matching results. The per-run table was derived once from the stored matching records by applying
the protocols' extractors and the all-round majority.

## Command

```sh
python3 code/compute.py --out output
```

Outputs: `values.json` and `sharing_effects.csv` (12 sharing effects under the four return rules
and the move of each majority variant).

## Definitions

- Sharing effect: pair F1 (match class) with full sharing minus pair F1 with no peers (access) or
  with answers only (content), pooled over all runs of the units present in both conditions; a run
  without an answer predicts non-match.
- Move: the sharing effect under a majority variant minus the effect under the protocol's own
  return rule; 1.98 pp is the largest absolute move over the 12 effects and three tie conventions.

## Scope

The census rows "Return rule→majority" of `tab:sensitivity-census` cover these 12 matching effects
together with the FORD and QA rows of the same analysis and are packaged with that table.

## Check

`python3 ../check_objects.py --recompute --only text_matching-return-rule` recomputes the values with `compute.py`,
compares its `values.json` with the shipped one and compares every value in `object.json` with the
number printed in the paper.
