# Using pair judgments to form groups (Fig. merge-validation)

Labels: `fig:merge-validation` (Fig. 5) and `sec:downstream` (Section VI-F), the matching sentence
of the abstract, and Supplementary `sec:s-grouping` (Section S6-H) with `tab:s-reversals` (Table S21)
and `tab:s-relations` (Table S22).

## What is printed

- Fig. merge-validation (a,b): F1 of each model alone (Llama-3-8B, Mistral-7B, Qwen2.5-7B to 72B)
  and of the six-model vote on WDC and Settlements, at three levels: the supplied candidate pairs,
  all direct record links over all record pairs, and the relations implied by the final groups
  (connected components).
- Fig. merge-validation (c): the change from no peers in pair F1 and cluster F1 when Qwen2.5-7B or
  Qwen2.5-32B receives the peers' decisions (D) or full responses (F), full-population Settlements.
- Section VI-F: connected components minus center clustering in test cluster F1 over 48
  comparisons (16 configurations per dataset); WDC's candidate pairs hold 500 of its 2,970 true
  matches, which caps direct-link F1 at 0.2882, and Settlements' candidates cover 99.35%; the
  Qwen2.5-32B decision-sharing example (pair F1 +0.03 pp, cluster F1 -3.05 pp; 23 of 4,719 judgments
  change, all to match; 147 added group relations, 133 of them incorrect indirect relations; nine
  incorrect changed links join previously separate components, each below the cluster-F1 improvement
  condition); 34 of 990 sharing-grouping combinations where pair F1 and cluster F1 move in opposite
  directions. The abstract repeats the 23 judgments and 133 relations.
- Supplementary `sec:s-grouping`: direction counts of the 990 combinations, by dataset and by
  grouping family (`tab:s-reversals`), the added and removed relations of the example
  (`tab:s-relations`), its nine merges and the largest one; each changed link added alone to the
  no-peer groups (14 join two groups, the nine incorrect ones add 130 false relations, the largest
  single addition is 20 relations); the same contrast in the second judgment collection on the same
  records (ten changed judgments, 24 added relations with 22 false indirect ones, pair F1 +0.05 pp,
  cluster F1 -0.55 pp); and the 13 of 66 test configurations whose sharing effect on cluster F1
  changes sign across the 15 grouping settings.

## Inputs

| file | rows | one row is |
|---|---|---|
| `inputs/settlements_records.csv` | 1,494 | a Settlements test record and its reference entity |
| `inputs/settlements_pairs.csv` | 4,719 | a Settlements candidate pair: reference label; each model's decision (most frequent over its repeats, none on a tie); the decision of Qwen2.5-7B and Qwen2.5-32B with no peers, with the peers' decisions, and with full responses |
| `inputs/wdc_pairs.csv` | 4,500 | a WDC test pair: the two records, their reference entities, reference label, each model's decision (most frequent over its repeats) |
| `inputs/settlements_second_collection_q32.csv` | 4,719 | a Settlements candidate pair in the second judgment collection: the Qwen2.5-32B decision with no peers and with the peers' decisions |
| `inputs/config_levels.csv` | 144 | a matching configuration (dataset x recipient x sharing condition, with random-peer draws) on the entity-disjoint test or development side: pair F1 and cluster F1 under each of the 15 grouping settings |

The two pair tables were derived once from the per-pair model replies, read through the matching code in `experiments/05_matching_downstream/code/gs_common.py` (Settlements
records, entities, candidate blocking) and `experiments/06_cost_resources/code/em_common.py` (WDC).
`config_levels.csv` comes from the grouping rules and metric code of
`experiments/09_grouping_transfer/code/` applied to the logged pair judgments, and
`settlements_second_collection_q32.csv` holds the two Qwen2.5-32B judgment sets of that collection
that the second-collection example compares.

## Command

```sh
python3 code/compute.py --out output
python3 code/plot.py      # draws merge_validation.pdf/.png from output/
```

`compute.py` writes `values.json`, `plot_data.json` (the plotted values),
`sharing_grouping_combinations.csv` (the 990 combinations with their pair and cluster changes) and
`settlements_q32_merges.csv` (the nine group merges of the example).

## Definitions

- A decision counts as a match only if it reads "match"; no decision counts as a non-match. The
  vote takes the most frequent decision of the six models and gives no decision on a tie.
- Cluster F1 scores all relations implied by the final groups against the reference relations on
  the same records, 2T/(P+G).
- A changed link added alone to the no-peer connected components joins groups U and V or falls inside
  one group; a joining link adds |U||V| relations, t of them true, and raises cluster F1 exactly when
  t/(|U||V|) exceeds half the no-peer cluster F1.
- A sharing-grouping combination is one sharing configuration (66: three datasets, two recipients,
  11 sharing conditions including random-peer draws) with one grouping setting held fixed; its
  pair and cluster changes are taken against the no-peer configuration of the same recipient.
  A change counts as zero only within floating-point noise; the near-zero column treats changes
  below 0.10 pp as zero.
- The center-versus-components comparison uses one configuration per recipient and sharing
  condition (the first random-peer draw), 16 per dataset; pair F1, calls and tokens are equal for
  the two rules because they group the same judgments.

## Scope

The full-population Settlements example and the 990 combinations come from different judgment
collections and are not pooled: the example uses the judgments of `experiments/05_matching_downstream`,
and the 990 combinations and the second-collection example use those of `experiments/09_grouping_transfer`, which judges the same 1,494
Settlements test records and 4,719 candidate pairs again, together with WDC and MusicBrainz. Counts over the 990 combinations are correlated (shared records,
judgments and baselines) and describe this matrix only. Reference entities identify the false
relations after the fact.

## Check

`python3 ../check_objects.py --recompute --only fig_merge-validation` recomputes the values with `compute.py`, compares
its `values.json` with the shipped one and compares every value in `object.json` with the number
printed in the paper. The plotted F1 values of panels (a,b) are listed with their printed
three-decimal form and no source file. Statements cover the counts printed as words (nine merges, four
components, five entities, ten changed judgments, nine of 16 negative comparisons) and the improvement
condition of the incorrect joining links.
