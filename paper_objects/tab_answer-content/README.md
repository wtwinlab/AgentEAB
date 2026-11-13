# tab_answer-content: full peer responses, extracted SQL only, and no peers

Labels: `tab:answer-content` (Table VI) and the answer-only paragraph of `sec:peer-flow`
(Section VI-D); supplementary `sec:s-peer-flow` with `tab:s-answer-transitions` (Table S15).

## What is printed

Du-MAD and MAD-M², each with Llama-3-8B or Qwen2.5-7B agents, on the 198 sampled BIRD questions with two
matched repeats, in three conditions: full peer responses, each peer's extracted SQL alone (rendered in
the protocol's answer form), and no peers. All conditions come from one collection
(`experiments/16_answer_only_peers/`).

- Table VI: the change from the initial answers to the last round, compared between conditions
  (answers − full, answers − no peers, full − no peers) over the four configurations, with question and
  database intervals.
- Section VI-D: the conditional loss of initially correct MAD-M²/Qwen2.5-7B answers with full responses
  and with answers only, and the MAD-M²/Qwen2.5-7B answers-minus-full contrast.
- Table S15: conditional loss and recovery per configuration and condition.
- Section S6-B: the number of runs, the answers-minus-full contrast per protocol, and the
  Du-MAD/Llama-3-8B returned accuracy under protocol extraction and its leniently read last-round vote,
  with full responses and with answers only.

## Inputs

`inputs/runs.csv`, 4,752 rows. One row is one debate run: question, database, protocol, backbone,
condition (`FULL`, `ANSWER_ONLY`, `NOPEER`; the Du-MAD no-peer runs contain no peer text), repeat; the
correctness of the execution vote over the initial answers (`r0`) and over the last round (`last`) with
the lenient extractor and first-agent ties; the correctness of the protocol's final answer under its own
extractor (`final`); and the correctness of each agent's initial and last-round answer under the lenient
extractor (`agents_r0`, `agents_last`, one digit per agent). Correctness is result equality with the
reference query. The table was derived once from the logged runs of the collection.

## Command

    python3 code/compute.py --out output

Under 1 s. Writes `output/values.json` and `output/contrasts.json` (every contrast with both intervals
and whether it is detected).

## Estimates

- Vote-level contrast: the round-to-round change (last − r0) of one condition minus that of another,
  paired per question; repeats averaged within a question, configurations averaged per question.
- Conditional loss: lost / (retained + lost) agent answers; conditional recovery: recovered /
  (recovered + stayed wrong); counts pooled over repeats and agents. The two rates have different
  denominators, so recovery minus loss is not the accuracy change.
- Intervals: 95% percentile bootstrap with 10,000 draws from one generator (seed 20261109), question
  weights first and then database-cluster weights over the 11 databases. A contrast is detected when
  both intervals exclude 0.

## Scope

198 sampled BIRD development questions; Du-MAD and MAD-M² with Llama-3-8B and Qwen2.5-7B agents.

## Check

`object.json` lists 53 printed values and 7 statements on which contrasts are detected.
`python3 ../check_objects.py --recompute --only tab_answer-content` confirms each printed form in its
paper file and that the computed value rounds to it.
