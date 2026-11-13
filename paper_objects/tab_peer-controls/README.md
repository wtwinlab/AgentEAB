# tab_peer-controls: Du-MAD and MAD-M² peer effects under fixed output rules

Labels: `tab:peer-controls` (Table V) with `sec:peer-controls` (Section VI-C); the peer interaction
paragraph of `sec:factorial` (Section VI-B) that compares output rules and tie rules; the per-agent
paragraph of `sec:peer-flow` (Section VI-D); supplementary `sec:s-inference` (the paragraph on the two
Du-MAD no-peer collections), `sec:s-peer-flow` with `tab:s-peer-transitions` (Table S14), and
`tab:s-ties-expanded` (Table S16).

## What is printed

Du-MAD and MAD-M², each with Llama-3-8B or Qwen2.5-7B agents, on BIRD development questions, with peers
and without peers, two repeats per question and condition.

- Table V and Section VI-C, 1,531 questions, lenient extractor, execution vote, first-agent ties:
  accuracy of the initial answers and of the last round with / without peers per protocol and pooled;
  the last-round difference (with peers minus without) with question and database intervals and per
  backbone; the initial-accuracy difference over the four pairs; the peer interaction effect with both
  intervals; the Du-MAD peer interaction loss under first-, last- and random-agent ties, and whether the
  pooled effect is detected under each tie rule.
- Section VI-B, 198 sampled questions: the peer interaction effect over the four configurations under
  the protocol extractor with the exact-match vote and under the lenient extractor with the execution
  vote, for each tie rule, and which of these are detected.
- Section VI-D and Table S14, 1,531 questions: per-agent loss (correct initial answer, wrong last-round
  answer) and recovery (wrong to correct) as shares of all agent trajectories, their difference (net),
  and conditional loss and recovery (shares of initially correct and initially wrong answers), with
  peers minus without; the Du-MAD recovery rates with and without peers; the vote-level peer interaction
  effect on the same questions.
- Table S16: the peer interaction effect under the three tie rules, on the 198 questions for the two
  output rules and per protocol, and on the 1,531 questions per protocol and pooled.
- Section S4-A: the Du-MAD no-peer runs of the 198-question comparison and of the 1,531-question
  comparison are separate collections; the number of the 792 runs whose vote differs under at least one
  execution-vote reading, the four-configuration peer interaction effect from each (−1.96 and −1.83), and
  the effect on the 1,333 other questions (−1.30).

## Inputs

- `inputs/runs_198.csv`, 3,168 rows: one row per run of the four configurations on the 198 sampled
  questions (question, database, protocol, backbone, condition `FULL` / `NOPEER`, repeat) with the
  correctness of the vote under the protocol extractor with the exact-match vote, the protocol extractor
  with the execution vote and the lenient extractor with the execution vote, for the initial answers and
  the last round, under each tie rule. These are rows of the per-run scores of `tab_factorial`; the
  later-round prompts of the Du-MAD no-peer runs contain no peer text.
- `inputs/runs_1531.csv`, 24,496 rows: one row per run on the 1,531 questions (source block, question,
  database, protocol, backbone, condition, repeat) with the correctness of the execution vote under both
  extractors, both rounds and the three tie rules, and the correctness of every agent's initial and
  last-round answer under the lenient extractor (`agents_r0`, `agents_last`, one digit per agent). With
  peers is condition `FULL`; without peers is `NOPEERC` for Du-MAD (later-round prompts contain no peer
  text) and `NOPEER` for MAD-M². Source `orig` holds the 198 sampled questions, `rx` the 1,333 remaining
  questions and the Du-MAD no-peer runs on all 1,531.
- `inputs/question_sets.json`: the 198 sampled and the 1,333 remaining questions; `bird|518`,
  `bird|701` and `bird|646` are excluded because their reference query exceeds the time limit.

The tables were derived once from the logged runs and their cached execution verdicts (the per-run and
per-agent tables of `tab_factorial` and `tab_s-reader-sensitivity`). A random tie is broken by a
SHA-256 hash of seed 20261106 and the run key, as in the factorial replay
(`share_eval/replay/factorial_replay.py`).

## Command

    python3 code/compute.py --out output

About 7 s. Writes `output/values.json` (every checked value) and `output/effects.json` (each estimate
with both intervals and whether it is detected).

## Estimates

- Peer interaction effect per question: (last round − initial answers) with peers minus the same change
  without peers. Repeats are averaged within a question and condition; a multi-configuration value
  averages the configurations per question.
- 1,531 questions: each question pairs its with-peer runs with the no-peer runs of the same
  configuration; a multi-configuration value keeps the questions present in every configuration. The
  question interval resamples questions and the database interval resamples the 11 databases (10,000
  draws each, seed 20261106). Conditional rates are ratios of summed agent counts within a configuration,
  averaged over configurations; their intervals use the same draws as resampling counts.
- 198 questions: question interval over questions and database interval over the 11 databases, 10,000
  draws each, seed 20261106, as in Fig. 4.
- A contrast is detected when its question-level and database-cluster 95% intervals both exclude 0.

## Scope

BIRD development questions, Du-MAD and MAD-M², Llama-3-8B and Qwen2.5-7B agents, fixed logged replies.

## Check

`object.json` lists 119 printed values (both ends of printed intervals and ranges) and 25 statements on
which contrasts are detected. `python3 ../check_objects.py --recompute --only tab_peer-controls`
confirms each printed form in its paper file and that the computed value rounds to it.
