# Peer interaction per BIRD protocol-backbone pair under two output rules and three tie rules

**Labels.** Fig. `fig:readout-cells` (Fig. 4, Section VI-B, `sec:factorial`; `figures/readout_cells.pdf`).

**What is plotted.** For Du-MAD, MAD-M², and BCCS with Llama-3-8B (L8) or Qwen2.5-7B (Q7) agents and
the mean of the six pairs: the peer interaction effect, (last round − initial answers) with peers minus
the same difference without peers, in pp with 95% intervals over the 198 questions. The left panel uses
the baseline output rules (protocol extractor, exact-match vote), the right panel the jointly changed
rules (lenient extractor, execution vote); each pair is shown under first-agent, last-agent, and random
ties: 42 points with their intervals. The figure prints no numbers in the text; `object.json` registers
the 126 plotted estimates and interval bounds.

**Inputs.** `inputs/per_run_scores.jsonl`, 5,544 rows; one row is one run (question, protocol,
backbone, condition, repeat) with the correctness of the vote under every extractor, vote, round, and tie
rule. It is the same file as in `tab_factorial`; the no-peer records of Du-MAD and BCCS are those
whose no-peer prompts contain no peer content.

**Commands.**

    python3 code/compute.py --out output     # values.json, plotted_values.csv (NumPy)
    python code/plot.py --out output          # readout_cells.pdf/.png (needs Matplotlib)

**Outputs.** `output/values.json` (estimate and interval of every point under all four combinations of
extractor and vote), `output/plotted_values.csv` (84 points in plotting order; the figure draws the 42 of
its two panels), `output/readout_cells.pdf`.

**Estimation.** Repeats are averaged within a question; the mean of six averages the pairs per
question. Intervals are percentile bootstrap intervals over questions (10,000 draws, seed 20261106).

**Scope.** The four-configuration means of Du-MAD and MAD-M² printed in Section VI-B under each tie
rule belong to Table S16 (`tab:s-ties-expanded`) and are reproduced with that table.

**Check.** The 126 plotted values in `object.json` against `values.json`; `plot.py` draws the paper's
Fig. 4 from `output/plotted_values.csv`.
