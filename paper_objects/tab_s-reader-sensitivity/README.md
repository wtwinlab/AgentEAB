# tab_s-reader-sensitivity: extractor errors and the 1,531-question effects; extractor audits

Labels covered: Supplementary Table `tab:s-reader-sensitivity` (Table S18) and the text of
`sec:s-reader-sensitivity` (Section S6-F); the audits of `sec:s-reader-audit` (Section S2-D) and their
summary at the start of Section S2 (`sec:s-framework`); the answer-recovery sentence of the validation
paragraph in Section IV (`sec:evaluation-checks`); the extractor sentences of Section VI-C
(`sec:peer-controls`).

## What is printed

The 1,531-question Du-MAD and MAD-M² effects (Llama-3-8B and Qwen2.5-7B agents; last-round difference and
interaction effect) under four readings, each with the execution vote and ties to the first agent:

- lenient extractor (the reading of Tables V and S17);
- protocol extractor: the protocol's own extractor alone, which drops every recovered query (a query the protocol's extractor
  misses and the lenient extractor finds);
- joint worst case, extractor's candidates: every ambiguous recovered reply (a reply with more than one
  candidate query and different execution outcomes among them) is reassigned, jointly across the agents of a
  run, to the candidate outcome least favourable to the estimate, and the vote is taken again;
- joint worst case, broad candidates: the same with every fenced block and every bare SELECT or WITH line as
  a candidate.

Intervals are 95% over questions and over the 11 BIRD databases; detected means both exclude 0. The text adds
the share of ambiguous recovered replies, the counts of vote-deciding recovered queries, the number of votes
the joint worst case changes, and the audit-rate sensitivity (1,000 random draws).

The audits: a blind human audit of 50 of the 17,350 BIRD replies in which only the lenient extractor finds SQL
(49 accepted as the intended final answer, Clopper–Pearson 95% interval); two agent judges (GPT-6 and Claude
Opus 5.5) on the same 50 items (agreement with the human annotator) and on 450 further items (438 accepted
by both, counting a query that carries the fence tag "boxed" as the same located query); a second blind
human audit of 50 recovered queries that decide an execution vote in the Du-MAD runs on the 1,531 questions,
stratified by backbone and condition (Wilson 95% intervals). The agent-judge results are agent review, not
human annotation.

## Inputs

- `inputs/rows.jsonl`, 38,592 rows, one per debate run (see `tab_s-controls`): `r0`, `last` (lenient extractor)
  and `r0_strict`, `last_strict` (protocol's own extractor), each the correctness of the execution vote. The
  Du-MAD no-peer runs are those of condition `NOPEERC` (their later-round prompts contain no peer text); MAD-M²
  uses `NOPEER`.
- `inputs/question_lists.json`, `inputs/question_lists_e4c.json`, `inputs/eval_meta.json`: question sets and
  the excluded questions.
- `inputs/candidates.jsonl`, 38,592 rows, one per run: for each agent reply of the initial and the last
  round, whether the protocol's extractor (`s`) and the lenient extractor (`l`) find a query, the execution
  signature and correctness of the query the lenient extractor takes (`cur`), and the execution signature and
  correctness of every candidate query (`cands`: the lenient extractor's own detection; `bcands`: the broad
  set). Derived once from the logged runs; candidate queries were executed read-only on the BIRD dev SQLite
  files. No query is executed here.
- `inputs/audit_first/`: `sample_key.json` (population per protocol and backbone, and the 50 sampled calls),
  `sample_public.json` (the 50 items as shown to the annotator: question, reply, recovered query),
  `human_labels.json` (the annotator's verdicts), `agent_judge_items.jsonl` (the 500 items shown to the agent
  judges), `agent_judge_key.json` (which items are the 50 human-labelled ones), and
  the two judges' verdicts `agent_judge_gpt6_extra_high.jsonl`, `agent_judge_claude_opus5.5_high.jsonl`.
- `inputs/audit_second/`: `sample_key.json` (population per backbone and condition, the 50 sampled replies
  with their run and vote), `sample_public.json` (the items as shown to the annotator), `human_labels.json`.

The extractor and the vote are the lenient extractor and the execution vote of the paper; `compute.py`
applies the four readings to `candidates.jsonl`.

## Command

    python3 code/compute.py --out output

The estimators are in `code/estimators.py` (10,000 bootstrap draws, seed 20261106, over questions and over
the 11 databases). The compute step re-derives every run's vote from `candidates.jsonl` and asserts that it
equals `rows.jsonl`. The audit-rate sensitivity draws from one random stream (seed 20261106) in the order of
the source analysis and asserts that every mean shift stays below 0.1 pp.

## Outputs

`output/values.json`, `output/reader_sensitivity.json` (every estimate with both intervals, votes changed,
ambiguity counts, audit-rate means and shifts) and `output/audits.json`.

## What the check compares

`object.json` lists 254 printed values, both ends of printed ranges included; each must occur in its paper
file and the computed value must round to it. The table cells' printed forms are read from the table source
itself. The bound "less than 0.1 pp" is listed under `not_checkable`; `compute.py` asserts it.
