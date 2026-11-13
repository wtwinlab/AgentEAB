"""Final-quality and stage tables assembled from the comparisons."""
from __future__ import annotations

from typing import Dict, List, Tuple

from share_eval_suite.comparisons import replacement as R

# (row label, stage, model, benchmark) in the order of Table VII and Table S24
STAGE_ROWS = (
    ("MAC-SQL Refiner, Qwen3-32B, BIRD", "refiner", "qwen3-32b", "bird"),
    ("MAC-SQL Refiner, Qwen2.5-72B, BIRD", "refiner", "qwen2.5-72b", "bird"),
    ("MAC-SQL Refiner, Qwen3-32B, Spider", "refiner", "qwen3-32b", "spider"),
    ("MAC-SQL Refiner, Qwen2.5-72B, Spider", "refiner", "qwen2.5-72b", "spider"),
    ("CHESS Unit Tester vs self-consistency, Qwen3-32B, BIRD", "unit_tester", "qwen3-32b", "bird"),
    ("ComEM selector, Qwen3-32B", "selector", "qwen3-32b", None),
    ("ComEM selector, Qwen2.5-7B", "selector", "qwen2.5-7b", None),
)


def replacement_table(inputs=None) -> List[Tuple[str, R.ReplacementCost]]:
    """Table VII: per stage, tasks, calls, tokens, GPU-seconds share and stage minus rule."""
    return [(lab, R.replacement_cost(s, m, b, inputs)) for lab, s, m, b in STAGE_ROWS]


def decomposition_table(inputs=None) -> List[Tuple[str, R.StageDecomposition, Dict[str, float]]]:
    """Table S24: Eq. 4 per stage with the stage's share of calls and tokens."""
    return [(lab, R.stage_decomposition(s, m, b, inputs), R.stage_share(s, m, b, inputs)) for lab, s, m, b in STAGE_ROWS]


def unit_tester_answers(inputs=None) -> Dict[str, int]:
    """CHESS on 147 BIRD questions: correct answers of the Unit Tester's returned branch and of each
    vote, and the questions on which a vote agrees with the Unit Tester's execution result."""
    rows = R._read("tab_replacement-cost", "chess_questions.csv", inputs)
    out = dict(questions=len(rows), unit_tester_entered=sum(int(r["ut_entered"]) for r in rows),
               unit_tester_correct=sum(int(r["UT"]) for r in rows))
    for name, col in R.VOTES.items():
        out[f"{name}_correct"] = sum(int(r[col]) for r in rows)
        out[f"{name}_same_result"] = sum(int(r[f"same_result_{col}"]) for r in rows if r[f"same_result_{col}"] != "")
    return out
