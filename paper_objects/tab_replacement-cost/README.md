# Rules with no LLM call in place of a stage (Table replacement-cost, stage rows; Tables S-chess, S-chess-resources, S-exec-cost)

Labels: `tab:replacement-cost` (Table VII; all rows except "Debate to initial-answer vote", which is a
separate object), `sec:replacement-cost` and `sec:chess` (Section VII-A), and in the Supplementary
`tab:s-chess` (Table S25), `tab:s-chess-resources` (Table S26), `sec:s-chess` (S7-B), and the
execution-cost table `tab:s-exec-cost` (Table S28) with the GPU-time estimation paragraph of `sec:s-exec`
(S7-D; the empty-winner and test-suite tables of that section are separate objects).

## What is printed

Each stage is replaced by a rule with no LLM call on the same logged input: the MAC-SQL Refiner by the
Decomposer's query, the CHESS Unit Tester by the pipeline's self-consistency vote (and the Valid and
Nonempty execution votes), and the ComEM selector by the ranker's own decision.

- Quality change, stage minus rule (pp), with 95% intervals; a positive value is accuracy given up by the
  replacement. For CHESS also the branch counts, the sensitivity rows (flagged outcomes scored as
  failures, the 145 questions on which the Unit Tester is entered, database weights) and the agreement of
  each rule with the Unit Tester's execution result.
- LLM work the rule saves: the stage's calls and input/output tokens, and its share of the pipeline's
  estimated LLM GPU time.
- Additional correct questions per million incremental tokens of the BIRD Refiner: 46.5 (Qwen3-32B) and
  72.4 (Qwen2.5-72B).
- The CHESS paragraph: 37.1% of calls removed, 85, 85 and 88 correct answers of the rules against the
  Unit Tester's 83, agreement on 114--116 questions, and 0.02--0.26 s of SQLite per question.
- Table S-exec-cost: per question, the execution votes' SQLite seconds next to the calls, tokens and
  estimated GPU-seconds of the LLM stage each replaces (fixed four-model BIRD and Spider pools, CHESS, and
  the Llama-3-8B and Qwen2.5-7B BIRD debates with peers); the GPU-time paragraph gives the range of the
  single-stream decoding rates (Qwen3-32B 49.97--51.70, Qwen2.5-72B-AWQ 40.21--40.56 tokens/s).

## Inputs (`inputs/`)

| file | rows | one row |
|---|---|---|
| `macsql_units_{q3,q72}_{bird,spider}.csv` | 1,534 / 1,034 | one dev question, Qwen3-32B (`q3`) or Qwen2.5-72B (`q72`): `Y_stage` (Refiner output correct), `Y_alt` (Decomposer query correct), `db_id` |
| `macsql_calls_per_question.csv` | 8,787 | LLM calls and tokens of one MAC-SQL agent on one question |
| `comem_anchors.csv` | 6,400 | one query record (8 datasets x 400) per selector model: selector and ranker correct (all ten decisions), selector call and tokens |
| `chess_questions.csv` | 147 | one BIRD question: correctness of the Unit Tester's returned answer (`UT`), of the same with flagged outcomes scored as failures (`UT_fail0`), of the three voting rules and of the best candidate; whether the Unit Tester is entered; whether each rule's pick has the same SQL text and the same execution result as the Unit Tester's |
| `chess_executions.csv` | 2,940 | one read-only execution of one of the 20 candidates of a question, with its logged seconds |
| `chess_wall_time.csv` | 147 | wall time of one CHESS question from the pipeline's log |
| `llm_calls.csv` | 22,597 | one proxy-logged LLM call of MAC-SQL, CHESS or ComEM: vLLM server, model, tool, start and end time, latency, status, prompt and completion tokens, and whether the prompt is the Refiner's |
| `pool4_questions.csv` | 2,568 | one BIRD or Spider question of the fixed four-model pool: SQLite seconds of its candidates, correctness of Valid, Nonempty, the Qwen3-32B selector and generation, and the DeepSeek-V4-Pro selector and generation, with DeepSeek tokens |
| `pool4_selector_calls.csv`, `pool4_generation_calls.csv` | 2,568 each | tokens of one Qwen3-32B selector or generation call on that pool |
| `chess_server_occupancy.csv` | 3 | one Qwen3-32B server replica of the CHESS workflow: start and removal time (UTC) |
| `debate_decodes.csv` | 2,772 | one decode of a BIRD debate with peers (7 protocol and model settings x 198 questions x 2 decodes): distinct initial queries and their SQLite seconds, later-round calls and tokens, whether the execution vote over the initial answers and the protocol's final answer are correct |

The MAC-SQL unit tables come from the stage-decomposition analysis, which re-executed both outputs
read-only and reproduced every stored score. The other tables are one-row-per-unit extracts of the logged
records; the voting rules and scorers that define the correctness columns are those of the CHESS, MAC-SQL
and ComEM workflows as the paper describes them.

Values worth naming: the server occupancy of the CHESS workflow, about 5.2 GPU-hours on three Qwen3-32B
replicas including loading and idle time, is the sum of the three start-to-removal times. The Table S28 cells 0.258 s
(CHESS), 23.2--39.6 (debate execution vote), +1 to +10.4 calls and +1.8k to +21.0k tokens (debates with
peers) and the batched GPU-second range 1.5--26.9 are checked against the same computation.

## Command

```
python3 code/compute.py --out output      # a few seconds
```

## Outputs

- `output/values.json`: every printed number under the key named in `object.json`.
- `output/throughput_by_server.csv`: the single-stream prefill and decode rates fitted for each vLLM server
  and the busiest 60-second rates, from which the GPU-second estimates follow.

## What the check compares

`check_objects.py` confirms that each printed value occurs in the named paper file and that the computed
value rounds to it. Quality intervals are percentile bootstraps with 10,000 draws and seed 20261105: SQL
rows resample databases (11 BIRD or 20 Spider databases, question weights, or database weights where the
table says so); ComEM rows resample query records within each of the 8 datasets. Calls, tokens and seconds
are sums or per-question means of the logged values.

GPU-seconds are estimates: for each stage, prompt tokens over the prefill rate plus completion tokens over
the decode rate, with rates fitted by least squares to the calls that had no overlapping call on the same
server. The point uses the median single-stream fit of the model; the batched value uses the fastest rate seen for the
model. The share is the stage's estimate over the estimate for the whole pipeline.

Both ends of printed ranges (for example "0.02--0.26 s", "49.97--51.70 tokens/s") are checked as separate values.
Every printed value of the object reproduces.

## Scope

CHESS is evaluated on 147 BIRD questions, MAC-SQL on BIRD dev and Spider dev, ComEM on 400 query records in
each of 8 datasets, the fixed pools on BIRD dev (1,534) and Spider dev (1,034), and the debates on 198 BIRD
questions. GPU-seconds refer to one H200 host; they are estimates of GPU work, not wall time or cost.
The debates have no throughput fit, and DeepSeek-V4-Pro is hosted, so neither has a GPU-second value.
