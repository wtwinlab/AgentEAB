# Grouping rules on fixed judgments (Section VII-C, `sec:grouping-rules`)

Labels: `sec:grouping-rules` (Section VII-C) and Supplementary `sec:s-grouping-menu` (S7-E).

## What is printed

- The menu: 15 grouping settings in six families (center clustering, merge-center clustering,
  connected components, disjoint stars, greedy correlation clustering at six thresholds, Markov
  clustering at five inflation values); 24 configurations per domain.
- Selecting the setting by development cluster F1 and scoring it on disjoint test entities: the
  mean gap to the best of the 15 settings on test (the regret) is 0.023, 0.028 and 0.069 pp on WDC,
  Settlements and MusicBrainz (maxima 0.244, 0.185, 0.205 pp); always using connected components
  loses 4.23 pp on Settlements and always using center clustering 7.88 pp on MusicBrainz.
- Which settings are selected, the development entity and record counts, and the logged time of
  all 15 settings per configuration (0.027-0.171 s).
- Development size on Settlements: scoring development cluster F1 on 21, 84, 126, 210 or all 419
  development entities gives mean test gaps of 4.009, 2.624, 1.244, 0.315 and 0.028 pp (fixed
  center clustering: 1.782 pp); scored on sampled entities, the false pairs of connected
  components fall from 200.2 at full size to 51.4 at half size and 1.5 at one tenth, while recall
  stays near 0.96.

## Inputs

| file | rows | one row is |
|---|---|---|
| `inputs/rule_cells.csv` | 1,080 | dataset x configuration x grouping setting: development and test cluster F1, test precision, recall and group counts |
| `inputs/selection.csv` | 72 | dataset x configuration: the selected setting, its test cluster F1, the best setting on test, the regret |
| `inputs/grouping.json` | 3 datasets | per configuration and setting on the development side: cluster F1 and logged seconds of the grouping algorithm; development entity and record counts |
| `inputs/settlements_dev_groups.csv` | 1,560 | a Settlements development record: its reference entity and its group id under each of the 24 configurations x 15 settings, computed on the complete development graph |

The first three files are the outputs of `experiments/09_grouping_transfer/code/eb1_grouping.py`.
`settlements_dev_groups.csv` applies the same rule code to the Settlements development judgments; every
column's development cluster F1 equals `rule_cells.csv`.

## Command

```sh
python3 code/compute.py --out output     # about 3 s
```

Outputs: `values.json` and `development_size.csv` (mean test gap per number of development
entities).

## Definitions

- Regret of a configuration: best test cluster F1 over the 15 settings minus the test cluster F1 of
  the chosen setting, in pp; means are over the 24 configurations of a dataset.
- Development selection takes the highest development cluster F1; ties go to the setting listed first
  in the menu order above (parameters increasing).
- Development size: for each of 200 seeded samples of Settlements development entities (one
  sample shared by the 24 configurations), development cluster F1 is scored only over the records
  of the sampled entities while the groups stay those of the complete development graph; the
  selected setting is scored on the unchanged test side. The precision and recall diagnostic uses
  the first 50 samples of each size. The seeds are those in `code/compute.py`.
- Seconds are the logged times of the grouping algorithms alone, summed over the 15 settings, on
  the development side of each configuration.

## Scope

Samples are nested in one finite development side and configurations share records and
judgments, so the development-size summaries are descriptive.

## Check

`python3 ../check_objects.py --recompute --only text_grouping-rules` recomputes the values with `compute.py`, compares
its `values.json` with the shipped one and compares every value in `object.json` with the number
printed in the paper.
