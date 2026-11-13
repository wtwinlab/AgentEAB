# Paper objects and their reproduction

Paper: *SHARE-Eval: A Stage-Level Benchmarking Framework for Multi-Agent Debate and Consensus in Data Workflows* (supplement `paper/supplementary.pdf`). Table and figure numbers are those of the
paper and the supplement.

Every paper object has one directory under `paper_objects/` with `inputs/` (per-question, per-run or
per-pair scored tables), `code/compute.py`, the expected `output/` and `object.json`, which lists each
printed value with where the paper prints it (`label`, `where`, and `tex_file`, the manuscript
section file) and the output key that reproduces it. Run from the
repository root:

```
python3 verify.py --recompute                                  # all objects and repository checks
python3 paper_objects/check_objects.py --recompute --only tab_strength
cd paper_objects/<dir> && python3 code/compute.py --out output # one object
```

A printed value passes when the computed value rounds to it;
a value the paper states as a bound ("no more than 2 pp") passes when the computed value does not exceed
it. Values printed inside a figure are checked against the plotted values. Each directory's `README.md` names its inputs, what one row
is, the estimator, the interval method and the scope the paper states.

Reading used by every BIRD debate object, as in the paper: lenient extractor and execution vote with ties
to the first agent unless a tie rule is stated; 198 sampled BIRD development questions or 1,531 BIRD
development questions; 95% paired bootstrap intervals over questions and over the 11 BIRD databases; an
effect is detected when both question-level and database-cluster 95% intervals exclude 0. The no-peer
runs of every object are those in its `inputs/`; their prompts contain no peer content.

## Main paper

| # | Label | Section | Directory | Printed values checked |
|---|---|---|---|---|
| Tab. I | `tab:design-dimensions` | II-B | — | design facts of the five protocols as run; no measured value |
| Fig. 1 | `fig:framework` | III | — | worked example; no measured value |
| — | `eq:candidate-flow` (Eq. 3) | III-C | `tab_candidate-flow` | statements: the identity holds exactly in every configuration and every run satisfies its ordering |
| — | `sec:evaluation-checks` | IV | `tab_s-reader-sensitivity`, `tab_s-online` | answer-recovery audits; online minus replay accuracy |
| Tab. II | `tab:matrix` | IV | `tab_matrix` | 11 (with the BIRD populations of Section IV); its trace and pair counts are checked in `text_matching-content` |
| Fig. 3 | `fig:access-matrix` | V-A | `fig_access-matrix` | 97 (figure cells, Section V-A, Supplement S5, S5-A, row R23 of Tab. S1) |
| Tab. III | `tab:family-sharing` | V-B | `tab_family-sharing` | 93 and 3 statements (with Tab. S6, S7) |
| — | `sec:matching-content` | V-C | `text_matching-content` | 196 (with Section VI-E, Tab. S8, S9, S23) |
| — | `sec:peer-source` | V-C | `text_peer-source` | 144 and 1 statement (with Tab. S10–S12) |
| — | `sec:candidate-flow` | VI-A | `tab_candidate-flow` | 71 and 13 statements (with Tab. S13 and Section S6-C) |
| Tab. IV | `tab:factorial` | VI-B | `tab_factorial` | 200 and 4 statements (with the debate row of Tab. VII, Section VII-A, Tab. S1, S20, Sections S2-B, S4-A) |
| Fig. 4 | `fig:readout-cells` | VI-B | `fig_readout-cells` | 126 plotted values |
| Tab. V | `tab:peer-controls` | VI-C | `tab_peer-controls` | 120 and 25 statements (with the peer-interaction paragraph of VI-B, the per-agent paragraph of VI-D, Tab. S14, S16, Section S4-A) |
| Tab. VI | `tab:answer-content` | VI-D | `tab_answer-content` | 53 and 7 statements (with Tab. S15, Section S6-B) |
| Fig. 5 | `fig:merge-validation` | VI-F | `fig_merge-validation` | 135 and 8 statements (with the abstract, Tab. S21, S22, Section S6-H) |
| Fig. 6 | `fig:stage-act` | VII-A | `fig_stage-act` | 89 (with Tab. S24) |
| Tab. VII | `tab:replacement-cost` (pipeline rows) | VII-A | `tab_replacement-cost` | 170 (with Tab. S25, S26, S28, Sections S7-B, S7-D); the debate row is in `tab_factorial` |
| Tab. VIII | `tab:sql-rule-strength` | VII-B | `tab_sql-rule-strength` | 228 (with Tab. S27, S29, S30) |
| — | `sec:grouping-rules` | VII-C | `text_grouping-rules` | 51 (with Section S7-E) |
| — | `sec:qa-vote-replay` (QA) | VII-D | `text_qa-vote-replay` | 9 (with Section S7-F) |
| — | `sec:qa-vote-replay` (matching) | VII-D | `text_matching-return-rule` | 10 (with row R20 of Tab. S1) |
| Fig. 7 | `fig:sql-family` | VIII-A | `fig_sql-family` | 59 (with Tab. S33) |
| Tab. IX | `tab:strength` | VIII-A | `tab_strength` | 272 (with Section VIII-C, Tab. S5, S32, S34–S38) |
| — | `sec:pool-availability` | VIII-B | `tab_s-pool-strata` | 41 and 7 statements (with Tab. S39) |
| Tab. X | `tab:scale` | VIII-D | `tab_scale` | 73 (with Tab. S40) |
| — | `sec:scale` (other model families) | VIII-D | `tab_s-cross-family` | 104 and 3 statements (with Tab. S41) |
| Fig. 8 | `fig:debate-budget` | VIII-E | `fig_debate-budget` | 173 and 22 statements (with Tab. S43–S45, Section S8-D) |
| Tab. XI | `tab:prior-evaluations` | IX | — | positioning against cited work; no measured value |

## Supplement

| # | Label | Directory |
|---|---|---|
| Tab. S1 | `tab:s-contrasts` | `tab_factorial` (class counts and rows R1, R3–R11); row R20 in `text_matching-return-rule`; row R23 in `fig_access-matrix` |
| Tab. S2 | `tab:s-suite-records` | execution information used by the suite; no measured value |
| Tab. S3 | `tab:s-coverage` | experiment matrix, as Tab. II |
| Tab. S4 | `tab:s-deviations` | configurations as run; no measured value |
| Tab. S5 | `tab:s-selector-controls` | `tab_strength` |
| Tab. S6 | `tab:s-first-decode` | `tab_family-sharing` |
| Tab. S7 | `tab:s-qa-transitions` | `tab_family-sharing` |
| Tab. S8 | `tab:s-confusion` | `text_matching-content` |
| Tab. S9 | `tab:s-content-choice` | `text_matching-content` |
| Tab. S10 | `tab:s-peer-effects` | `text_peer-source` |
| Tab. S11 | `tab:s-peer-change` | `text_peer-source` |
| Tab. S12 | `tab:s-weaker` | `text_peer-source` |
| Tab. S13 | `tab:s-candidate-flow` | `tab_candidate-flow` |
| Tab. S14 | `tab:s-peer-transitions` | `tab_peer-controls` |
| Tab. S15 | `tab:s-answer-transitions` | `tab_answer-content` |
| Tab. S16 | `tab:s-ties-expanded` | `tab_peer-controls` |
| Tab. S17 | `tab:s-controls` | `tab_s-controls` (239 values, with the controls of Section VI-C) |
| Tab. S18 | `tab:s-reader-sensitivity` | `tab_s-reader-sensitivity` (254 values, with the extractor audits of Sections S2 and S2-D) |
| Tab. S19 | `tab:s-failures` | `tab_s-failures` (35 values) |
| Tab. S20 | `tab:s-levels` | `tab_factorial` |
| Tab. S21 | `tab:s-reversals` | `fig_merge-validation` |
| Tab. S22 | `tab:s-relations` | `fig_merge-validation` |
| Tab. S23 | `tab:s-traces` | `text_matching-content` |
| Tab. S24 | `tab:s-decomposition` | `fig_stage-act` |
| Tab. S25 | `tab:s-chess` | `tab_replacement-cost` |
| Tab. S26 | `tab:s-chess-resources` | `tab_replacement-cost` |
| Tab. S27 | `tab:s-q25-rules` | `tab_sql-rule-strength` |
| Tab. S28 | `tab:s-exec-cost` | `tab_replacement-cost` |
| Tab. S29 | `tab:s-empty` | `tab_sql-rule-strength` |
| Tab. S30 | `tab:s-testsuite` | `tab_sql-rule-strength` |
| Tab. S31 | `tab:sensitivity-census` | `tab_sensitivity-census` (37 values, with Section S7-G) |
| Tab. S32 | `tab:s-did` | `tab_strength` |
| Tab. S33 | `tab:s-family-sql` | `fig_sql-family` |
| Tab. S34 | `tab:s-aligned` | `tab_strength` |
| Tab. S35 | `tab:s-repeat` | `tab_strength` |
| Tab. S36 | `tab:s-repeat-within` | `tab_strength` |
| Tab. S37 | `tab:s-availability` | `tab_strength` |
| Tab. S38 | `tab:s-quality-cost` | `tab_strength` |
| Tab. S39 | `tab:s-pool-strata` | `tab_s-pool-strata` |
| Tab. S40 | `tab:s-scale-contrasts` | `tab_scale` |
| Tab. S41 | `tab:s-cross-family` | `tab_s-cross-family` |
| Tab. S42 | `tab:s-online` | `tab_s-online` (41 values and 3 statements, with Sections S2-C and S8-C) |
| Tab. S43 | `tab:s-budget-cost` | `fig_debate-budget` |
| Tab. S44 | `tab:s-budget-contrasts` | `fig_debate-budget` |
| Tab. S45 | `tab:s-draft-diversity` | `fig_debate-budget` |

Supplementary text without a table: the proof of Lemma 1 (`sec:s-lemma-proof`) carries no measured value;
`sec:s-ford` is in `fig_access-matrix`; `sec:s-spend-peers` in `text_peer-source`; `sec:s-grouping-menu`
in `text_grouping-rules`; `sec:s-return` in `text_qa-vote-replay`; the availability paragraph of
`sec:s-spend-pool` in `text_generation-beyond-pool` (12 values and 4 statements); the replay counts of
`sec:s-inference` in `tab_factorial` and `tab_peer-controls`.

Rows R1, R3–R11, R20 and R23 of Tab. S1 are checked in the objects named above; the other rows restate
effects that the objects of their own sections check.

## Collection code behind each paper object

| Experiment code | Paper objects it feeds |
|---|---|
| `code/consensus_delivery/`, `experiments/02_controlled_sharing/` | `fig_access-matrix`, `tab_family-sharing`, `tab_candidate-flow`, `tab_factorial`, `tab_matrix`, `tab_peer-controls`, `fig_readout-cells`, `tab_s-controls`, `tab_s-reader-sensitivity`, `tab_s-failures`, `text_qa-vote-replay`, `text_matching-return-rule`, `tab_sensitivity-census`, `fig_debate-budget` |
| `experiments/14_nopeer_collection/` (no-peer runs of Du-MAD and BCCS) | `fig_access-matrix`, `fig_readout-cells`, `tab_factorial`, `tab_matrix`, `tab_peer-controls`, `tab_s-controls`, `tab_s-reader-sensitivity`, `tab_s-online`, `tab_scale`, `tab_s-cross-family`, `tab_sensitivity-census`, `text_matching-return-rule` |
| `experiments/04_sql_selection_generation/` | `tab_sql-rule-strength`, `tab_strength`, `fig_sql-family`, `tab_s-pool-strata` |
| `experiments/05_matching_downstream/`, `experiments/06_cost_resources/` | `fig_merge-validation`, `fig_sql-family` |
| `experiments/07_content_choice/` | `text_matching-content` |
| `experiments/08_cross_family/` | `tab_family-sharing`, `text_matching-content`, `fig_sql-family`, `text_peer-source`, `tab_strength`, `tab_replacement-cost` |
| `experiments/09_grouping_transfer/` | `text_grouping-rules`, `fig_merge-validation` |
| `experiments/10_chess_unit_tester/` | `fig_stage-act`, `tab_replacement-cost` |
| `experiments/11_macsql_comem_stages/` | `fig_stage-act`, `tab_replacement-cost` |
| `experiments/12_deepseek_selection_generation/` | `tab_strength`, `text_generation-beyond-pool`, `tab_s-pool-strata`, `tab_replacement-cost` |
| `experiments/13_backbone_scale_cross_family/` | `tab_scale`, `tab_s-cross-family`, `tab_s-online` |
| `experiments/15_stronger_peers/` | `text_peer-source`, `tab_strength` |
| `experiments/16_answer_only_peers/` | `tab_answer-content`, `fig_debate-budget` (the three-agent, two-round debates) |
| `experiments/17_budget_matched_debates/` | `fig_debate-budget` |
| `share_eval/replay/`, `share_eval/replay_validity/`, `share_eval/stage_replacement/` | the re-reading, Lemma 1 and stage-replacement steps behind the objects above (`share_eval/README.md`) |

## Figures

Figures 4–7 are drawn by the `code/plot.py` of the corresponding object from its `output/` (matplotlib);
Fig. 3 is a LaTeX table that `code/plot.py` writes from `output/`; Fig. 8 is drawn in LaTeX from the
plotted values its object checks:

| Figure | File | Generator |
|---|---|---|
| Fig. 3 | `access_matrix.tex` | `paper_objects/fig_access-matrix/code/plot.py` |
| Fig. 4 | `readout_cells.pdf` | `paper_objects/fig_readout-cells/code/plot.py` |
| Fig. 5 | `merge_validation.pdf` | `paper_objects/fig_merge-validation/code/plot.py` |
| Fig. 6 | `stage_act.pdf` | `paper_objects/fig_stage-act/code/plot.py` |
| Fig. 7 | `sql_family.pdf` | `paper_objects/fig_sql-family/code/plot.py` |
| Fig. 8 | drawn in the paper's source | plotted values in `paper_objects/fig_debate-budget/output/plotted_values.csv` |

## SQL execution

No paper-object computation executes SQL; the scored tables in `inputs/` carry execution outcomes. The
per-question scores of Qwen2.5-7B generation with and without execution feedback (`fig_sql-family`) were
obtained by executing the generated and reference queries read-only against the BIRD development databases
(4,393 statements). The Spider test-suite outcomes in `tab_sql-rule-strength` come from the stored
execution results on the test-suite databases.
