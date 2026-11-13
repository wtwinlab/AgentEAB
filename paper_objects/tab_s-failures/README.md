# Why a final answer on BIRD is wrong

**Object.** Supplementary Table `tab:s-failures` (Table S19, Section S6-G, `sec:s-failures`).

**What is printed.** For each protocol and backbone on the 198 BIRD questions with peers (396 runs per
configuration), the share of runs whose final answer is correct and the shares of three exhaustive failure
classes: generation (no reply holds a correct query under the lenient extractor), format (a reply holds a
correct query that the protocol's own extractor misses), and selection (the protocol's extractor finds a
correct query, but the vote, stop rule, aggregator, or return step does not return it). The replies
considered are every agent's initial and last-round reply.

**Inputs.** `inputs/failure_classes_runs.csv`, 3,168 rows; one row is one BIRD run with peers
(question, protocol, backbone, repeat), with the outcome of the protocol's final answer and its
class. The table was derived once from the debate logs and the stored execution verdicts by the
class rule stated above. Correctness is execution match with the gold query on the BIRD database.

`inputs/failure_classes_candidate_sets.csv`, 16 rows; one row is one protocol–backbone configuration with peers
under one candidate set (`initial_and_last_round`: every agent's initial and last-round reply;
`every_round`: every agent's reply in every round), with the run count and the counts of correct runs
and of the three classes. Its initial-and-last-round counts equal the per-run table
(`every_round_cells_agree_with_runs`).

**Command.**

    python3 code/compute.py --out output

**Outputs.** `output/values.json`: per configuration, runs and the four shares (%), and the ranges
over configurations.

**What the check compares.** Every cell of the table, the run count, the 74.5% of Du-MAD/Llama-3-8B
runs without a correct query, and the statement of `sec:s-failures` that using every round moves no class by more than 2 pp,
checked as a bound (the largest move is 1.52 pp, BCCS/Llama-3-8B selection).
