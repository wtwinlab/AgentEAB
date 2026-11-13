# SHARE-Eval

SHARE-Eval is the paper's evaluation procedure for logged multi-agent runs (Section III). This directory
holds its code in three parts:

| Directory | What it does | Paper |
|---|---|---|
| `replay/` | re-reads logged debates under every combination of answer extractor (protocol's own or lenient), vote (exact-match vote or execution vote), round (initial answers or last round) and peer condition (with or without peers) | Sections III-C, VI-B; Table IV, Fig. 4 |
| `replay_validity/` | checks when a replay on the logs equals deployment (Lemma 1): first divergence of BCCS's own reads, bounds, the grouping bound, the online check, the sensitivity census | Sections III-D, VIII-D, S7-G; Table S31 |
| `stage_replacement/` | Eq. (4): for an extra stage replaced by a rule with no LLM call, how often the stage changes the result, its gain when it does, and its calls, tokens and GPU-seconds | Section VII-A; Fig. 6, Table VII |

All scripts run from the repository root with `python3 -I`, make no model call, and open the BIRD and
Spider databases read-only only when a query has no cached execution result. Seeds and bootstrap draws
are fixed in the code (10,000 draws; seed 20261106 for `replay/` and `replay_validity/`, 20261105 for
`stage_replacement/`).

## Inputs

The paper objects in `paper_objects/` start from the intermediate files these scripts write. The
logged runs the scripts read are not in this repository (see "Data not in this repository" in the
top-level README); the default paths follow that layout:

| Default path | Contents |
|---|---|
| `data/records/bird_debates/` | the BIRD debates among Llama-3-8B or Qwen2.5-7B agents (Du-MAD, MAD-M², BCCS, MoA; with and without peers; two repeats): `raw/sql_bird_v2__*.jsonl`, the evaluated set `analysis/sql_per_unit.jsonl` and the stored verdicts `analysis/sql_verdicts.json` |
| `data/records/bird_debates_backbones/` | the same layout for the larger Qwen2.5 and cross-family backbones and the online BCCS runs with execution voting |
| `data/caches/sql_execution/` | the execution result signature and verdict of every logged query (`r0_exec_signatures.jsonl`, `replay_exec_signature_cache.jsonl`, `replay_verdict_cache.jsonl`, `divergence_exec_signature_cache.jsonl`, `divergence_verdict_cache.jsonl`, `exec_fresh.jsonl`, `verdict_audit.json`); with them no query is executed |
| `data/records/dataeng_raw/sql/dev_20240627/dev.json` | BIRD development questions and reference queries; `ARTIFACT_ROOT` sets the directory that holds `data/records/` for the shared SQL helpers |
| `data/records/macsql_comem_stages/` | MAC-SQL and ComEM runs (`output/`, `data/llm4em/`, `raw/`, `logs/host/db_before.sha256`) |
| `data/records/chess_unit_tester/` | CHESS runs (`output/`, `data/tasks_147.json`, `raw/`, `logs/db_formal_after.sha256`) |
| `$BIRD_DB_DIR`, `$SPIDER_DB_DIR` | the BIRD and Spider databases (`data/external/README.md`) |

Every input and output path is also a command-line argument (`--help` lists them). Outputs go to
`output/share_eval/<part>/` by default. The SQL helpers are imported from
`experiments/02_controlled_sharing/code/` (`sql_tasks.py`, `ext_eval_sql.py`) and
`experiments/06_cost_resources/code/` (`sql_common.py`); the BCCS read functions come from
`code/consensus_delivery/adapters/bccs.py`.

## `replay/`

```
python3 -I share_eval/replay/factorial_replay.py        # -> output/share_eval/replay/
python3 -I share_eval/replay/db_cluster_bootstrap.py
python3 -I share_eval/replay/report_tables.py
```

- `factorial_replay.py` reads every run's initial answers and last-round replies, extracts a query with
  the protocol's own extractor and with the lenient extractor (the protocol's answer, else the last
  fenced SQL block, else the last line-initial SELECT/WITH), votes by exact match or by execution result
  (ties to the first, the last or a seeded random agent), and scores the winner against the reference
  result. A question's repeats are averaged before any contrast; questions 518 and 701 have no reference
  result and are excluded. Writes `per_run_scores.jsonl` (one row per run: correctness under every
  extractor × vote × round × tie rule), `factorial_replay.json` and `.csv` (levels, effects, interaction
  effects with question-level intervals), `ties.csv`, and the execution caches it adds.
- `db_cluster_bootstrap.py` adds the database-cluster intervals (resampling the 11 BIRD databases)
  next to the question-level ones: `db_cluster_bootstrap.json` and `.md`.
- `report_tables.py` renders `factorial_replay.json` as `tables.md`.

`per_run_scores.jsonl` is the input of `paper_objects/tab_factorial`, `fig_readout-cells`,
`tab_matrix` and `tab_s-controls` (the same file in each) and of `replay_validity/`.

## `replay_validity/`

```
python3 -I share_eval/replay_validity/bccs_divergence.py   # -> output/share_eval/replay_validity/
python3 -I share_eval/replay_validity/bounds.py
python3 -I share_eval/replay_validity/grouping_bound.py
python3 -I share_eval/replay_validity/online_validation.py
python3 -I share_eval/replay_validity/sensitivity_census.py
python3 -I share_eval/replay_validity/bounds_bccs_only.py --div <dir>/bccs_divergence_runs.jsonl --out <dir>/bounds_bccs.json
```

- `bccs_divergence.py` takes BCCS's stop test, conflict estimate and peer choice from the adapter source,
  checks that they reproduce every logged decision, and finds each run's first divergence under every
  alternative extractor and vote: none, recoverable or unrecoverable. Writes `bccs_divergence_runs.jsonl`
  (one row per BCCS run: divergence class and the run's deployment bounds) and `bccs_divergence.json`.
  Consumed by `paper_objects/tab_factorial` (rows R7 and R11 of Table S1) and `tab_s-online`.
- `bounds.py` turns the divergence classes and the replay scores into the effect interval of Section III-D for
  every replayed contrast: `bounds.json`. `bounds_bccs_only.py` does the same for BCCS debates without
  replay scores (Qwen2.5-14B).
- `grouping_bound.py` evaluates the grouping corollary on a synthetic bridge and on the Settlements
  comparison of Section VI-F, from `data/sharing_long.csv`: `grouping_bound.json`.
- `online_validation.py` compares the replay with the online BCCS runs with execution voting inside the
  protocol (Qwen2.5-7B), per question and in distribution: `online_validation.json`. Consumed by
  `paper_objects/tab_s-online`.
- `sensitivity_census.py` compares the reading a protocol reports with a controlled reading for every
  eligible cell (Table S31): `sensitivity_census.json` and `.csv`. Its matching and QA inputs are the
  per-cell tables `data/match_qa_audit.json` (protocol's extractor against the lenient extractor) and
  `data/return_rule_cells.csv` (protocol's return rule against the all-round majority). Consumed by
  `paper_objects/tab_sensitivity-census`.

## `stage_replacement/`

```
python3 -I share_eval/stage_replacement/reexec.py          # -> output/share_eval/stage_replacement/
python3 -I share_eval/stage_replacement/throughput.py
python3 -I share_eval/stage_replacement/decompose.py
python3 -I share_eval/stage_replacement/costs_comparable.py
```

- `reexec.py` decides for every question whether the stage changes the result (A_u): it executes the
  stage's output and the rule's output read-only and compares their result classes under the
  pipeline's own scorer. Writes `macsql_units_<model>_<benchmark>.csv`, `chess_units.csv`,
  `reexec_log.json`. The unit files are the inputs of `paper_objects/fig_stage-act` and
  `tab_replacement-cost`.
- `throughput.py` measures per-server throughput from the per-call proxy records and estimates each
  stage's GPU-seconds: `throughput_by_server.csv`, `stage_gpu_seconds.csv`, `throughput.json`.
- `decompose.py` computes Eq. (4) per stage and rule: n, the change rate P(A) and the gain E[D | A] with
  bootstrap intervals, gains and losses, and the identity check total = P(A) · E[D | A]. Writes
  `decomposition.csv` and `.json`, `movers.csv`, `costs.csv`, `comem_per_dataset.csv`,
  `comem_pair_level.csv`.
- `costs_comparable.py` puts the stage's incremental calls, tokens, seconds and GPU-seconds next to the
  quality change: `costs_comparable.csv`.

## Checks

```
python3 -I share_eval/check_consistency.py
```

reruns the steps whose inputs ship with the repository (`db_cluster_bootstrap.py` and `bounds.py` on
`paper_objects/tab_factorial/inputs/`, `grouping_bound.py` on `replay_validity/data/`) and compares each
output's SHA-256 with that of the output the paper uses. With the logged runs in place, rerunning
`factorial_replay.py` on the BIRD debates reproduces `paper_objects/tab_factorial/inputs/per_run_scores.jsonl`
byte for byte and `bccs_divergence.py` reproduces `bccs_divergence_runs.jsonl`; from the unit files in
`paper_objects/fig_stage-act/inputs/` and the MAC-SQL, ComEM and CHESS runs, `throughput.py`,
`decompose.py` and `costs_comparable.py` reproduce their outputs byte for byte.
