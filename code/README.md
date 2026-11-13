# Collection code

This directory and `experiments/*/code/` hold the code that produced the model outputs behind the
paper's scored tables. The paper objects in `paper_objects/` do not run it: they start from the scored
tables in their `inputs/`. It is included so a reader can see how each experiment called the models,
built the prompts of each condition, extracted answers and scored them.

## `code/consensus_delivery/`

- `adapters/`: one adapter per consensus protocol. The paper's five protocols are Du-MAD
  (`du_mad.py`, `du_mad_roster.py` for mixed-model rosters), MAD-M² (`mad_m2.py`, `mad_m2_roster.py`),
  BCCS (`bccs.py`), MoA (`moa.py`) and FORD (`ford.py`). Each keeps the released protocol's prompts,
  rounds, stopping and return rule, and records every call. The other adapters (`dmad.py`,
  `liang_mad.py`, `debate_or_vote.py`, `reconcile.py`, `delib_task_allocation.py`,
  `mallm_upstream.py`) cover protocols that were run but are not reported in the paper.
  `official_scorers.py` holds the benchmark scorers; `native_run.py` and `_runner.py` the shared run
  loop; `sources.py` the upstream source pointers.
- `providers.py`, `registry_live.py`, `credentials.py`: the OpenAI-compatible client, the model
  endpoint registry and credential lookup.
- `collect_matrix.py`, `collect/`: collection drivers for the protocol-by-model matrix.
- `contracts.py`, `formation.py`, `mechanisms.py`: per-protocol answer and stopping contracts.
- `tests/`: offline tests of the scorers, credential handling and run accounting (no model call).

## `experiments/*/code/`

| Directory | What the code collects or scores |
|---|---|
| `02_controlled_sharing/code/` | The controlled sharing study: the five protocols on QA, entity matching and BIRD/Spider, with peers and without peers. `arm_transform.py` builds the prompts of each condition (the no-peer condition removes the peer replies and the clauses that refer to them, and keeps rounds, stopping and return rules); `arm_transform_collection.py` is the transform module the collection runner imported, which `verify_sent_input.py` uses to recompute the stored request hashes. `collect_controlled.py`, `collect_sql_v2.py`, `collect_all_matching.py`, `unit_driver.py` and `ext_*.py` drive the collection; `answer_extract.py`, `eval_sql.py`, `eval_matching.py` extract and score answers. `code_snapshot/` holds the adapter versions the collection imported. |
| `04_sql_selection_generation/code/` | SQL candidate pools, selection and generation calls on BIRD and Spider, and the matching evidence variants. |
| `05_matching_downstream/code/` | Pair judgments on the Geographic Settlements data and the grouping of judged pairs into record groups. |
| `06_cost_resources/code/` | Entity-matching and SQL pools with per-call token and time accounting, used for the cost tables. |
| `07_content_choice/code/` | Amazon–Google peer-content conditions (no peers, peer decision, decision plus fields, full explanation). |
| `08_cross_family/code/` | DBLP–ACM content conditions, QA with fixed peers, cross-family SQL generation, and paired token counts. |
| `09_grouping_transfer/code/` | Grouping rules applied to the judged pairs and their configuration levels. |

## Running it

Every collection script calls models through OpenAI-compatible endpoints that you provide. Set
`INFERENCE_HOST` to the host serving the open-weight models (the ports per model are in
`code/consensus_delivery/registry_live.py` and the collection scripts), and for hosted models set the
provider key in the environment (for example `DEEPSEEK_API_KEY`); `credentials.py` reads keys from the
environment or from a `.env` file you create, and never writes them out. `deploy_pool.sh` and
`ext_serve.sh` start vLLM servers on a GPU host given by `REMOTE_SSH_TARGET`. Collection writes under
`workspace/` (controlled sharing runner) or `results/` (other drivers), relative to the repository root. The raw records are not in this repository (see the
top-level `README.md`).

## Upstream implementations

Some adapters read the released protocol code at run time, to check prompts against it or to call its
decision function. Clone each upstream repository at the commit below into `vendor/` at the repository
root before running them:

| Path | Repository | Commit | Read by |
|---|---|---|---|
| `vendor/mad` | https://github.com/composable-models/llm_multiagent_debate | `9846749350eb` | `adapters/du_mad.py` (hash of `gsm/gen_gsm.py`), `adapters/du_mad_roster.py` |
| `vendor/consensus_literature/moa` | https://github.com/togethercomputer/MoA | `1b5cab0f0905` | `adapters/moa.py` (reads `moa.py` on import) |
| `vendor/reconcile` | https://github.com/dinobby/ReConcile | `902ae913cbb3` | `adapters/reconcile.py` |
| `vendor/consensus_literature/mallm` | https://github.com/Multi-Agent-LLMs/mallm | `a57c4a77d37f` | `adapters/mallm_upstream.py` (with `patches/`) |
| `vendor/delib` | upstream of the task-allocation games (repository and commit not recorded here) | — | `adapters/delib_task_allocation.py` (not used by the paper) |

The same paths are read by the copies in `experiments/02_controlled_sharing/code/code_snapshot/`.
