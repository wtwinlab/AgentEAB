# Reproducing the paper

All commands run from the repository root unless stated otherwise. Timings are wall-clock on an Apple
Silicon laptop (CPython 3.14.5, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1).

## 1. Environment

```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

The computations need numpy, pandas and scipy; the figure scripts (`code/plot.py`) also need matplotlib.

## 2. Everything at once (about one minute)

```
python3 verify.py --recompute
```

The last line reads `N of N checks passed; 3143 of 3143 printed values reproduce`. With `--json FILE` it
also writes every check and every printed value with its computed value to FILE.

## 3. One paper object

```
cd paper_objects/<dir>
python3 code/compute.py --out output     # writes output/values.json and the object's tables
cd ../.. && python3 paper_objects/check_objects.py --recompute --only <dir>
```

| Directory | Paper objects | Time |
|---|---|---|
| `tab_matrix` | Tab. II, BIRD populations of Section IV | 4 s |
| `fig_access-matrix` | Fig. 3, Section V-A, Sections S5, S5-A, row R23 of Tab. S1 | < 1 s |
| `tab_family-sharing` | Tab. III, Tab. S6, S7 | < 1 s |
| `text_matching-content` | Section V-C (matching), Section VI-E, Tab. S8, S9, S23 | < 1 s |
| `text_peer-source` | Section V-C (stronger peers), Tab. S10–S12 | 2 s |
| `tab_candidate-flow` | Section VI-A, Eq. 3, Tab. S13, Section S6-C | < 1 s |
| `tab_factorial` | Tab. IV, debate row of Tab. VII, Sections VI-B, VII-A, Tab. S1, S20, Sections S2-B, S4-A | 2 s |
| `fig_readout-cells` | Fig. 4 | < 1 s |
| `tab_peer-controls` | Tab. V, Sections VI-B to VI-D (peer interaction and per-agent transitions), Tab. S14, S16, Section S4-A | 7 s |
| `tab_answer-content` | Tab. VI, Section VI-D (answer-only peers), Tab. S15, Section S6-B | < 1 s |
| `tab_s-controls` | Tab. S17, controls of Section VI-C | 3 s |
| `tab_s-reader-sensitivity` | Tab. S18, Sections S2, S2-D, answer-recovery audits of Section IV | 6 s |
| `tab_s-failures` | Tab. S19 | < 1 s |
| `fig_merge-validation` | Fig. 5, Section VI-F, abstract, Tab. S21, S22, Section S6-H | < 1 s |
| `fig_stage-act` | Fig. 6, Section VII-A, Tab. S24 | < 1 s |
| `tab_replacement-cost` | Tab. VII (pipeline rows), Section VII-A, Tab. S25, S26, S28 | < 1 s |
| `tab_sql-rule-strength` | Tab. VIII, Section VII-B, Tab. S27, S29, S30 | 4 s |
| `text_grouping-rules` | Section VII-C, Section S7-E | 2 s |
| `text_qa-vote-replay` | Section VII-D (QA), Section S7-F | < 1 s |
| `text_matching-return-rule` | Section VII-D (matching), row R20 of Tab. S1 | < 1 s |
| `tab_sensitivity-census` | Tab. S31, Section S7-G | < 1 s |
| `fig_sql-family` | Fig. 7, Tab. S33 | < 1 s |
| `tab_strength` | Tab. IX, Sections VIII-A, VIII-C, Tab. S5, S32, S34–S38 | < 1 s |
| `text_generation-beyond-pool` | availability paragraph of Section S8-A | < 1 s |
| `tab_s-pool-strata` | Section VIII-B, Tab. S39 | 2 s |
| `tab_scale` | Tab. X, Section VIII-D, Tab. S40 | 2 s |
| `tab_s-cross-family` | Section VIII-D (other model families), Tab. S41 | 3 s |
| `tab_s-online` | Tab. S42, Sections S2-C, S8-C, replay-accuracy check of Section IV | < 1 s |
| `fig_debate-budget` | Fig. 8, Section VIII-E, Tab. S43–S45, Section S8-D | 1 s |

## 4. Figures

```
cd paper_objects/<fig_dir> && python3 code/plot.py
```

draws the figure from `output/`; the directory's README names the paper figure it corresponds to. Figure binaries carry the plotting library's metadata and are not byte-identical across
library versions; the plotted values are.

## 5. Collecting model outputs

The collection code is in `code/` and `experiments/*/code/` (`code/README.md`). It needs inference
endpoints you provide and the upstream protocol implementations named in `code/README.md`. No step above
runs it: the paper objects start from the scored tables in their `inputs/`.
