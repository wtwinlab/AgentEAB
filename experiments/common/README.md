# Shared analysis modules

Modules imported by several experiments in `experiments/10_*` to `experiments/15_*`:

| File | Role |
|---|---|
| `code/core.py`, `code/share_eval.py`, `code/adapters/` | stage-replacement estimates: paired cluster bootstrap, transitions, specification checks; adapters for CHESS, SQL pools, matching and QA |
| `code/build_inputs.py`, `code/score_sal.py`, `code/pool_score.py` | inputs and scoring of the fixed SQL candidate pool (selector given the generation prompt, generation) |
| `code/a_sql_small_cluster.py` | small-cluster bootstrap sensitivity over the 11 BIRD databases |

No model call. Paths are relative to the repository root; the derived tables they read are in
`data/derived/` of the hosted data folder.
