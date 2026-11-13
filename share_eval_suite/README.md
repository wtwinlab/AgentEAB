# SHARE-Eval suite

The paper is reproduced with `verify.py` (paper objects, from their shipped inputs) and with `share_eval/`
(the scripts that wrote those inputs from the logged runs). This package is a refactor of `share_eval/`
for reuse by others: it gives the same results and is organised like the benchmarking suite in the
top-level README (`docs/share_eval_suite.svg`): adapters run workflows on tasks and write execution
records; stage evaluation reads the records for candidate availability and answer changes; alternative
comparisons contrast peer inputs, stage replacements, added computation, interactions and dependency
checks; reports give stage metrics, final quality, cost and confidence intervals. A new protocol or task
is added as an entry of the adapter registries (`adapters/protocols.py`, `adapters/tasks.py`), and a new
alternative for a stage as a comparison in `comparisons/` that uses the estimators of `stats.py` and the
reports of `reports/`.

The records, stage evaluation, comparisons and reports are implemented in this package; shared
estimators are in `stats.py` (percentile bootstraps over questions, databases and dataset strata,
Clopper–Pearson intervals, Holm's procedure). They use the same estimators, seeds and draw orders as
the paper objects, so they return exactly the values the paper objects ship. `steps/` runs the analyses of
`share_eval/` on the logged runs with this code: each step writes the same files as the `share_eval/`
script of the same name (see "Same results as share_eval/"). The protocol registry loads each protocol's
collection code (`code/consensus_delivery/`, `experiments/`) on request.

## Layout

| Suite box | Module | Contents |
|---|---|---|
| Protocol adapters | `adapters/protocols.py` | registry of the debate and consensus methods (Du-MAD, MAD-M², BCCS, MoA, FORD and the other adapters in `code/consensus_delivery/adapters/`) and the pipelines (CHESS, MAC-SQL, ComEM): stages, final decision, implementation, `load()` |
| Task adapters | `adapters/tasks.py` | question answering, text-to-SQL and entity matching: task inputs, answer extraction and scorers on in-memory values |
| Execution records | `records/schema.py`, `records/readers.py` | `ExecutionRecord` (one run on one task: per agent and round the responses and candidate answers, calls and tokens; the final decision and stopping; task scores); readers of the logged debates and of the scored per-run tables |
| Candidate availability | `stages/availability.py` | whether a correct answer is present after generation, revision, extraction and the final decision; gains split by whether the pool holds a correct candidate |
| Answer changes | `stages/changes.py` | transitions between rounds, readings and peer conditions: corrections, regressions, net change |
| Entity matching downstream | `stages/grouping.py` | pair judgments → grouping → indirect relations; pair, direct-link and cluster F1 |
| Peer inputs | `comparisons/peers.py` | with vs without peers under a fixed extractor, vote, round and tie rule; answer-only peers; controls |
| Stage replacements | `comparisons/replacement.py` | LLM stage vs a rule with no LLM call: change rate × conditional gain (Eq. 4), stage − rule, cost per correct answer |
| Added computation | `comparisons/computation.py` | call-budget-matched debates vs votes over independent drafts, larger backbones, stronger models, selection and generation on a fixed SQL pool |
| Interactions | `comparisons/interactions.py` | factorial main and interaction effects; joint change vs the sum of single changes |
| Dependency checks | `comparisons/dependency.py` | replay bounds from first-divergence events, the online check, the sensitivity census |
| Stage metrics, final quality | `reports/quality.py` | tables assembled from the comparisons |
| Cost | `reports/cost.py` | calls, prompt and completion tokens, GPU-seconds |
| Confidence intervals | `reports/intervals.py`, `stats.py` | percentile bootstraps over questions, databases and dataset strata |
| Logged runs → scored tables | `steps/` | the `share_eval/` steps on the logged runs (table below) |

| `share_eval/` script | Suite step | Writes |
|---|---|---|
| `replay/factorial_replay.py` | `steps/factorial_replay.py` | `per_run_scores.jsonl`, `factorial_replay.json`, `.csv`, `ties.csv` |
| `replay/db_cluster_bootstrap.py` | `steps/db_cluster_bootstrap.py` | `db_cluster_bootstrap.json`, `.md` |
| `replay/report_tables.py` | `steps/report_tables.py` | `tables.md` |
| `replay_validity/bccs_divergence.py` | `steps/bccs_divergence.py` | `bccs_divergence_runs.jsonl`, `bccs_divergence.json` |
| `replay_validity/bounds.py`, `bounds_bccs_only.py` | `steps/bounds.py`, `steps/bounds_bccs_only.py` | `bounds.json`, `bounds_bccs.json` |
| `replay_validity/grouping_bound.py` | `steps/grouping_bound.py` | `grouping_bound.json` |
| `replay_validity/online_validation.py` | `steps/online_validation.py` | `online_validation.json` |
| `replay_validity/sensitivity_census.py` | `steps/sensitivity_census.py` | `sensitivity_census.json`, `.csv` |
| `stage_replacement/reexec.py` | `steps/reexec.py` | `macsql_units_<model>_<benchmark>.csv`, `chess_units.csv`, `reexec_log.json` |
| `stage_replacement/throughput.py` | `steps/throughput.py` | `throughput_by_server.csv`, `stage_gpu_seconds.csv`, `throughput.json` |
| `stage_replacement/decompose.py` | `steps/decompose.py` | `decomposition.csv`, `.json`, `movers.csv`, `costs.csv`, `comem_per_dataset.csv`, `comem_pair_level.csv` |
| `stage_replacement/costs_comparable.py` | `steps/costs_comparable.py` | `costs_comparable.csv` |

Each step takes the options of its script; outputs go to `output/share_eval_suite/<part>/` by default.

Functions read the scored tables shipped in `paper_objects/<dir>/inputs/` by default; each takes another
inputs directory, table file or list of rows as an argument.

## Use

From the repository root:

```
python3 -m share_eval_suite list-protocols [--kind consensus|pipeline] [--paper]
python3 -m share_eval_suite list-tasks
python3 -m share_eval_suite replay [--suite] [factorial_replay|db_cluster_bootstrap|report_tables] [ARGS...]
python3 -m share_eval_suite stage-replacement [--suite] [reexec|throughput|decompose|costs_comparable] [ARGS...]
python3 -m share_eval_suite replay-validity [--suite] [bccs_divergence|bounds|...] [ARGS...]
python3 -m share_eval_suite check
```

`replay`, `stage-replacement` and `replay-validity` run the `share_eval/` scripts on the logged runs of
the hosted data folder (see the top-level README); `--suite` runs the suite's step of the same name
instead, with the same outputs. Without the logged runs they print the missing paths and the step's options.

```python
from share_eval_suite.comparisons import peers, replacement
from share_eval_suite.stages import availability

peers.peer_effect("protocol", "exact", tie="first", configurations="both")     # +3.03 pp, intervals
peers.peer_effect("lenient", "execution", tie="first", configurations="both")  # -1.96 pp
replacement.replacement_cost("refiner", "qwen3-32b", "bird")                   # +1.83 [1.00, 2.70] pp
availability.candidate_flow()["mean"]                                          # candidate flow, six configurations
```

## Check

`python3 -m share_eval_suite check` runs `share_eval_suite/tests/` (pytest). The tests reproduce the
paper's numbers from the shipped inputs and compare them with each paper object's
`output/values.json`, among them the peer interaction effects +3.03 and −1.96 pp (Section VI-C), the 23 changed pair
judgments and 133 false indirect relations (Section VI-F), the stage − rule differences of Table VII, the
candidate flow, the call-budget-matched debates and the replay bounds. No test calls a model, starts a
server or needs a GPU; without the hosted data, SQL is executed only on the small SQLite fixtures of
`reexec`, and one test of the logged-debate reader runs only when `data/records/` is present. `tests/test_standalone.py` checks that the stage evaluation,
comparisons and reports load no code outside this package.

## Same results as `share_eval/`

`tests/test_consistency_replay.py`, `test_consistency_validity.py` and `test_consistency_replacement.py` run
every `share_eval/` script and the suite step of the same name on the same inputs, into separate
directories, and compare every output file byte for byte. Fields that record when or how long a run took
are removed before the comparison:

| File | Fields not compared |
|---|---|
| `factorial_replay.json` | `generated_utc`, `wall_s`, `execution_cost.wall_s` |
| `SHA256SUMS` of the replay directory | the line of `factorial_replay.json` (its digest covers `generated_utc`; it is checked against the file in its own directory) |
| `exec_signature_cache.jsonl`, `verdict_cache.jsonl` | the per-query execution seconds (`s`, `seconds`) |
| `bccs_divergence.json` | `exec_cost.wall_s` |
| `reexec_log.json` | `started`, `finished`, the `seconds` of each MAC-SQL cell and of CHESS |

The steps whose inputs ship with the repository (`db_cluster_bootstrap`, `bounds`, `bounds_bccs_only`,
`grouping_bound`, `sensitivity_census`, `online_validation` without online runs, and `reexec` on small
SQLite fixtures) are compared in every run of `python3 -m share_eval_suite check`. The others read the logged runs of the hosted data folder, from six
of its archives: `share_eval_bird_debates`, `share_eval_bird_debates_backbones`, `share_eval_sql_caches`,
`exp10_chess_unit_tester`, `exp11_macsql_comem_stages` and `records_benchmark_inputs_and_sql_pools` (for
`dev.json` of BIRD and Spider). Extract them into one directory (the repository root, or any other
directory named by `SHARE_EVAL_DATA_ROOT`), obtain the BIRD and Spider databases as described in
`data/external/README.md` and check them against `data/external/CHECKSUMS.json`, link the BIRD databases
into the data directory, and run the tests:

```
export SHARE_EVAL_DATA_ROOT=<directory holding data/records and data/caches>
export BIRD_DB_DIR=<BIRD dev_databases> SPIDER_DB_DIR=<Spider database>
ln -s "$BIRD_DB_DIR" "$SHARE_EVAL_DATA_ROOT/data/records/dataeng_raw/sql/dev_20240627/dev_databases"
python3 -B -m pytest -q -p no:cacheprovider share_eval_suite/tests
```

The run takes about 40 minutes on a 288-core server; most of it is the replay of the larger backbones.

The replay and replay-validity steps read the BIRD reference queries (`dev.json`) and databases under the
data root, the directory named by `ARTIFACT_ROOT` (default: the repository root); the tests set it to
`SHARE_EVAL_DATA_ROOT`, and a step run directly on a data folder elsewhere needs `ARTIFACT_ROOT`.

`factorial_replay` executes the queries that have no stored result on the BIRD databases, read from
`<data root>/data/records/dataeng_raw/sql/dev_20240627/dev_databases/`. The scripts `bccs_divergence.py`
and `online_validation.py` execute such queries as well; the suite steps of the same name execute none:
without a stored result for every query they report how many are missing and stop with exit status 2, so
they are compared on complete stored results. The factorial replay is compared in two parts. First, each
implementation executes the queries without a stored result on its own; the queries executed, their order
and their results must agree, except the results of queries that read the current time (SQLite `'now'`,
e.g. an age computed as `julianday('now') - julianday(birth_date)`), which may differ between executions,
and of queries that time out in one execution and not in the other. Second, both run again on the
execution results the `share_eval/` run wrote, so that no query without a stored result is executed, and
every output file is compared as above; in both runs the recheck of the few suspect queries (listed in
the audit file, or whose stored result and verdict disagree on whether the query ran) executes them again.
`reexec` executes the MAC-SQL and CHESS outputs on the BIRD and Spider databases in each implementation
on its own, and the unit files and the log are compared byte for byte (the log without the fields in the
table).

The same tests also check that the re-run outputs equal the scored tables shipped in `paper_objects/`
(`per_run_scores.jsonl`, `bccs_divergence_runs.jsonl`, the stage-replacement unit files). In the MAC-SQL
unit files, the result-class digests `keep_class` and `refiner_class` of questions whose query reads
`'now'` are not compared with the shipped ones. The throughput, decomposition and cost outputs are also
compared with those of the hosted `stage_decomposition/` folder: the same text, except that a number may
differ by 1e-12 relative, as floating-point sums do across Python and numpy builds.
