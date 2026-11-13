# Extraction and voting on fixed BIRD debate responses (198 questions)

**Labels.** Table `tab:factorial` (Table IV, Section VI-B, `sec:factorial`) and the paragraphs that
discuss it; the BCCS sentences of Section VI-C (`sec:peer-controls`: baseline vote above the returned
answer, changed continuation decisions, deployment bound of the peer effect); the initial-answer vote in
Section VII-A (`sec:replacement-cost`) and the debate row of Table `tab:replacement-cost` (Table VII);
Supplementary Section S2-B (`sec:s-contrasts`) with rows R1 and R3–R11 of Table `tab:s-contrasts`
(Table S1), the BCCS dependency checks and contrast classes of Section S4-A (`sec:s-inference`), and
Table `tab:s-levels` (Table S20).

**What is printed.** On 198 sampled BIRD development questions with two repeats, Du-MAD, MAD-M², and BCCS
debates with Llama-3-8B or Qwen2.5-7B agents (six configurations; MoA where stated) are re-read under two
extractors (the protocol extractor, a lenient extractor) and two votes (exact-match vote, execution vote
that groups queries by their execution result and drops failing ones), at the initial answers and at the
last round, with and without peers. Table IV gives the gain of the lenient extractor, of the execution
vote, and of both over the baseline configuration (protocol extractor, exact-match vote, last round, with
peers), and the joint gain minus the sum of the two single gains, with 95% intervals over questions and
over databases. The first column averages the four Du-MAD and MAD-M² configurations, whose re-reads equal
the deployed outcome; the second averages all six, where BCCS values are replay values with a deployment
bound. The prose adds the run-level corrections of the joint change (141 runs), the six-configuration
deployment bound of the joint gain, the three-way extractor, vote, and round interaction (4.10 pp), the BCCS
replay bounds, and the vote over each run's initial answers against the debate's baseline answer, with its
call and token savings. Section S4-A states that recomputing BCCS's reads from the logged prefix
reproduces 3,553 answer verdicts, 2,688 stopping decisions, and 1,969 peer choices, and classifies the
24 contrasts under Lemma 1 (12 exact, four on average, three bounded, five for Du-MAD and MAD-M² alone).
All values are pp (levels in %).

**Inputs** (`inputs/`):

| file | rows | one row |
|---|---|---|
| `per_run_scores.jsonl` | 5,544 | one run (question, protocol, backbone, condition, repeat): correctness of the final answer and of the vote under every extractor, vote, round, and tie rule |
| `bccs_divergence_runs.jsonl` | 1,584 | one BCCS run: for each reading, whether and where the deployed run would diverge from the logs, and the outcome interval |
| `bird_units_db.csv` | 1,534 | one BIRD development question and its database |
| `debate_cost_runs.csv` | 3,200 | one BIRD run with peers: calls and prompt/completion tokens of the initial round and of later rounds |
| `vote_ties.csv` | 168 | one (configuration, reading, round, condition): share of runs whose vote is tied |
| `matching_return_rule.csv` | 40 | one (protocol, benchmark, shared content): sharing effect under the protocol's return rule and under the all-round majority with each tie rule, and the tied shares |
| `replacement_contrasts.csv` | 24 | one replacement contrast and its class under Lemma 1 |
| `bccs_read_checks.csv` | 1,584 | one BCCS run: answer verdicts, stopping decisions (the no-consensus flag), and peer choices recomputed from the logged prefix with the protocol's own functions, and how many equal the logged values |

The no-peer runs of Du-MAD and BCCS in `per_run_scores.jsonl` and `bccs_divergence_runs.jsonl` have
prompts that contain no peer content. The per-run scores define the extractors and the votes as in the
paper: the lenient extractor takes the protocol's answer when it finds one, else the last fenced SQL block,
else the last line that starts a query; a tie goes to the first agent, the last agent, or a seeded random
agent. `bird_units_db.csv` and `debate_cost_runs.csv` were extracted once from the BIRD development file and
the per-run cost log; `bccs_read_checks.csv` was derived once from the logged BCCS debates (the runs behind
the 198-question comparisons) by recomputing each read with the stop rule, conflict estimate, and peer
selection taken from the protocol's source.

**Command.**

    python3 code/compute.py --out output

Standard library, NumPy, and SciPy; about 3 s.

**Outputs.** `output/values.json`: every value the check compares, plus the per-configuration estimates
and bounds of all contrasts.

**Estimation.** Repeats are averaged within a question; the six- and four-configuration values are
equal-weight means over configurations per question. Question intervals are percentile bootstrap intervals
over questions (10,000 draws, seed 20261106, one index matrix for all contrasts); database intervals
resample the 11 BIRD databases (10,000 sequential draws, seed 20261106). For BCCS, whose stopping rule reads
the extracted answers, the deployment bound replaces each diverging run's outcome by 0 and by 1 unless early
stopping determines it; Du-MAD and MAD-M² replay exactly (Lemma 1). Shares of diverging runs come with
Clopper–Pearson intervals over questions. Run counts of corrections and regressions compare the lenient
extractor with the execution vote against the baseline configuration in each of the 1,584 Du-MAD and MAD-M²
runs with peers.

**Scope.** The peer interaction effects under the three tie rules printed in Section VI-B belong to
Table S16 (`tab:s-ties-expanded`); the 1,531-question comparisons belong to Table V (`tab:peer-controls`).

**Check.** Each entry of `object.json` names a printed number, the paper file in which it appears, and the
key of `values.json` that must round to it; Table VII prints Δ as stage minus rule. The statements check
that no recomputed BCCS read differs from the logged one and the three contrast classes printed as words.
