# Where extra stages change the result (Fig. stage-act, Table S-decomposition)

Labels: `fig:stage-act` (Fig. 6), `sec:decomposition`, `sec:replacement-cost`, `sec:chess` (Section VII-A),
`tab:s-decomposition` (Table S24), `sec:s-stages` (Supplementary S7-A).

## What is printed

Three published pipelines each have an extra stage that can be replaced by a rule with no LLM call on the
same logged input: the MAC-SQL Refiner (rule: keep the Decomposer's query), the CHESS Unit Tester
(rule: the pipeline's self-consistency vote) and the ComEM selector (rule: the ranker's own decision).
For every task, D = (stage correct) - (rule correct), and the stage changes the result when the two outputs
fall in different scorer classes (SQL result sets; for ComEM, the vector of ten match decisions of a
query record). Every task on which D is not 0 is a changed task, so the stage's total effect is
(share of changed tasks) x (mean D on changed tasks).

- Fig. stage-act: share of all tasks on which the stage changes the result, split into gains, losses and
  both wrong, with the total effect; next to it the stage's share of its pipeline's LLM calls and tokens.
- Table S-decomposition: the counts behind the figure, the gain on changed tasks (where at least 30 tasks
  change), and the total effect with its 95% interval.
- Section VII-A quotes these numbers: the Refiner changes 8-10% of BIRD results and gains 23-33 pp on
  them; on Spider fewer than 2% change and the gains stay below 1 pp; the Unit Tester changes 15.65% of
  results with 4 gains and 6 losses; the Qwen3-32B ComEM selector improves accuracy by 5.19 pp.

## Inputs (`inputs/`)

| file | rows | one row |
|---|---|---|
| `macsql_units_{q3,q72}_{bird,spider}.csv` | 1,534 / 1,034 | one BIRD or Spider dev question for Qwen3-32B (`q3`) or Qwen2.5-72B (`q72`): `A_result` (outputs in different result classes), `Y_stage` (Refiner output correct), `Y_alt` (Decomposer query correct), `db_id` |
| `chess_units.csv` | 441 | one of 147 BIRD questions x one voting rule; the figure uses `alternative = SC_official` (self-consistency) |
| `comem_anchors.csv` | 6,400 | one query record (8 datasets x 400) for the Qwen3-32B (`q3`) or Qwen2.5-7B (`q7`) selector: `changed`, `selector_correct`, `ranker_correct` (all ten decisions correct), selector tokens |
| `macsql_calls_per_question.csv` | 8,787 | LLM calls and tokens of one MAC-SQL agent (Selector, Decomposer, Refiner) on one question |
| `chess_costs_per_question.csv` | 294 | LLM calls and tokens of one CHESS question, split into the shared prefix and the Unit Tester's increment |

The MAC-SQL and CHESS unit tables come from the stage-decomposition analysis, which re-executed both outputs
read-only on the BIRD and Spider SQLite files and reproduced every stored score. The ComEM and MAC-SQL call
tables are one-row-per-unit extracts of the logged ComEM records and MAC-SQL traces.

## Command

```
python3 code/compute.py --out output      # values.json, plotted_values.csv (seconds)
python code/plot.py                        # stage_act.pdf, stage_act.png from output/ (needs matplotlib)
```

## Outputs

- `output/values.json`: every printed number under the key named in `object.json`. The bounds of the text
  ("fewer than 2%" and "below 1 pp" on Spider) are checked as `"relation": "le"` entries.
- `output/plotted_values.csv`: the bars and markers of the figure; identical to the values the paper's
  figure is drawn from.
- `output/stage_act.pdf`: the figure.

## What the check compares

`check_objects.py` confirms that each printed value occurs in the named paper file and that the computed
value rounds to it. Intervals are percentile bootstraps with 10,000 draws and seed 20261105: the SQL rows
resample the 11 BIRD or 20 Spider databases with question weights; the ComEM rows resample query records
within each dataset. No interval is given for the gain on changed tasks when fewer than 30 tasks change
(both Spider rows and CHESS). Both ends of printed ranges (for example 27--31%) are checked as
separate values.

## Scope

The CHESS comparison covers 147 BIRD questions; MAC-SQL covers BIRD dev (1,534) and Spider dev (1,034);
ComEM covers 400 query records in each of 8 datasets. A stage's share of calls and tokens is over its own
pipeline's LLM calls; ComEM's selector is that pipeline's only LLM call.
