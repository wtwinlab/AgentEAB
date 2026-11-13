#!/usr/bin/env python3
r"""The BIRD SQL adaptation extended to **all five** target protocols, plus the SQL answer kind.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

import answer_extract as AX
import arm_transform as AT
import sql_tasks as ST

BENCH = ST.BENCH                      # "sql_bird_v2"
KIND = ST.KIND                        # "sql"

#: a statement opener, used both by the natural-language tier and by the well-formedness gate
_STMT = re.compile(r"(?is)\b(SELECT|WITH)\b")
_WELL = re.compile(r"(?is)^\s*(SELECT|WITH)\b")
_FENCE_ANY = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)

_INSTALLED: Dict[str, Any] = {}


# ------------------------------------------------------------------ the SQL answer kind
def sql_nl(text: Any) -> Optional[str]:
    """The peer's own query, read out of prose. `None` when the peer stated no query.
    """
    t = str(text or "")
    if not t.strip():
        return None
    m = _FENCE_ANY.findall(t)
    if m:
        got = ST.unwrap_sql(m[-1])
        if got and _WELL.match(got):
            return got
    hit = _STMT.search(t)
    if not hit:
        return None
    seg = t[hit.start():]
    # a statement ends at its semicolon when it has one; otherwise at the first blank line, because
    # these replies routinely continue with an explanation paragraph
    semi = seg.find(";")
    if semi >= 0:
        seg = seg[:semi]
    else:
        brk = seg.find("\n\n")
        if brk > 0:
            seg = seg[:brk]
    got = ST.unwrap_sql(seg)
    return got if got and _WELL.match(got) else None


def sql_well_formed(s: Any) -> bool:
    """True when `s` IS a query rather than prose that contains one."""
    t = str(s or "").strip()
    if not t or "```" in t:
        return False
    return bool(_WELL.match(t))


def _register_answer_kind() -> Dict[str, Any]:
    AX._NL[KIND] = lambda t: sql_nl(t)
    AX._WELL_FORMED[KIND] = sql_well_formed
    AX.KIND_BY_BENCHMARK[BENCH] = KIND
    return {"nl_parser": "ext_sql_tasks.sql_nl", "well_formed": "ext_sql_tasks.sql_well_formed"}


# ------------------------------------------------------------------ Du-MAD
#: Du's sentence shape, with the task noun and the answer form substituted -- the same one-clause
#: change his adapter already labels for MMLU-Pro, GPQA, StrategyQA and SQuAD 2.0.
DU_ANSWER_CLAUSE = ("Your final answer should be a single SQL query, in the form "
                    "\\boxed{{answer}}, at the end of your response.")
DU_FRAMING = "Can you answer the following database question? {q} Explain your reasoning. "
DU_SUFFIX_FRAMING = ("\n\n Using the solutions from other agents as additional information, can "
                     "you provide your answer to the question? \n The original question is {q}. ")


def _install_du_mad() -> Dict[str, Any]:
    from consensus_delivery.adapters import du_mad
    du_mad.ANSWER_CLAUSE.setdefault(BENCH, DU_ANSWER_CLAUSE)
    du_mad.FRAMING.setdefault(BENCH, DU_FRAMING)
    du_mad.SUFFIX_FRAMING.setdefault(BENCH, DU_SUFFIX_FRAMING)
    if not getattr(du_mad, "_ext_sql_installed", False):
        orig = du_mad.extract_for

        def extract_for(benchmark, text, n_options: int = 10):
            if benchmark == BENCH:
                # Du-MAD's own contract on this task is its `\boxed{}` clause, so the box is read
                # first; the unwrapping is protocol-agnostic only.
                g = du_mad.boxed_group(text)
                got = ST.unwrap_sql(g) if g else ""
                return got or None
            return orig(benchmark, text, n_options)

        extract_for.__name__ = "extract_for"
        du_mad.extract_for = extract_for
        du_mad._ext_sql_installed = True
    return {"answer_clause": DU_ANSWER_CLAUSE, "extractor": "extract_for -> boxed_group+unwrap"}


# ------------------------------------------------------------------ FORD
#: FORD's released instruction shape, with the answer form changed and nothing else -- the same
#: labelled change its GSM8K variant (`INSTRUCTION_NUMERIC`) already makes.
FORD_INSTRUCTION_SQL = (
    "Do you think I am more reasonable? Please give your final answer starting with \"Answer: \" "
    "followed by the single SQL query, and explain very shortly starting with \"Explanation: \". "
    "You should give only one query.")


def ford_parse_sql(text: str) -> Tuple[str, str]:
    """`(prediction, explanation)` for SQL, in `get_prediction_explanation`'s shape.
    """
    t = (text or "").strip()
    if "Answer:" not in t:
        return "None", t
    after = t.split("Answer:", 1)[1]
    m = re.split(r"Explanation:\s*", after, maxsplit=1)
    pred_part = m[0]
    expl = m[1] if len(m) > 1 else ""
    got = ST.unwrap_sql(pred_part)
    if not got:
        got = sql_nl(pred_part) or ""
    return (got or "None"), (expl.strip() or t)


def _install_ford() -> Dict[str, Any]:
    from consensus_delivery.adapters import ford
    if not getattr(ford, "_ext_sql_installed", False):
        o_instr, o_parse = ford.instruction, ford.parse_prediction

        def instruction(benchmark: str, n_options: int = 4) -> str:
            if benchmark == BENCH:
                return FORD_INSTRUCTION_SQL
            return o_instr(benchmark, n_options)

        def parse_prediction(text: str, benchmark: str):
            if benchmark == BENCH:
                return ford_parse_sql(text)
            return o_parse(text, benchmark)

        instruction.__name__ = "instruction"
        parse_prediction.__name__ = "parse_prediction"
        ford.instruction = instruction
        ford.parse_prediction = parse_prediction
        ford.ADAPTED = set(ford.ADAPTED) | {BENCH}
        ford._ext_sql_installed = True
    return {"instruction": FORD_INSTRUCTION_SQL, "parser": "ext_sql_tasks.ford_parse_sql"}


# ------------------------------------------------------------------ MoA
#: MoA ships no released parser or answer contract (its demo answers one open-ended question with
#: no scorer), so it gets a one-line answer instruction, exactly as its four QA benchmarks do.
MOA_ANSWER_INSTRUCTION_SQL = " State the single SQL query at the end."


def _install_moa() -> Dict[str, Any]:
    from consensus_delivery.adapters import moa
    moa.ANSWER_KIND.setdefault(BENCH, KIND)
    moa.ANSWER_INSTRUCTION.setdefault(BENCH, MOA_ANSWER_INSTRUCTION_SQL)
    if not getattr(moa, "_ext_sql_installed", False):
        orig = moa.extract

        def extract(text: str, kind: str) -> str:
            if kind == KIND:
                # the study's own convention for this kind, called rather than reimplemented, and
                # deliberately no more permissive than the contract extractors it is compared with
                return ST.unwrap_sql(text) if sql_well_formed(ST.unwrap_sql(text)) \
                    else (sql_nl(text) or "")
            return orig(text, kind)

        extract.__name__ = "extract"
        moa.extract = extract
        moa._ext_sql_installed = True
    return {"answer_instruction": MOA_ANSWER_INSTRUCTION_SQL,
            "extractor": "unwrap_sql, else the natural-language tier"}


# ------------------------------------------------------------------ the ANSONLY render for FORD
def _install_render() -> Dict[str, Any]:
    r"""FORD's ANSONLY re-render has to produce SQL in FORD's own answer form.
    """
    prev = AT._RENDER["ford"]
    if getattr(prev, "_ext_sql", False):
        return {"ford_render": "already installed"}

    def render(lab: str) -> str:
        if sql_well_formed(lab):
            return "Answer: %s" % lab
        return prev(lab)

    render._ext_sql = True                                                # type: ignore[attr-defined]
    AT._RENDER["ford"] = render
    return {"ford_render": "Answer: <query> when the label is a query, released sentence otherwise"}


# ------------------------------------------------------------------ install / verify
def install() -> Dict[str, Any]:
    """Idempotent."""
    if _INSTALLED:
        return _INSTALLED
    base = ST.install()                               # bccs, dmad, mad_m2 -- unchanged
    base["signature_order"] = ST.verify_signature_order()
    out = {
        "benchmark": BENCH, "answer_kind": KIND,
        "from_sql_tasks": base,
        "added_here": {
            "answer_kind_tables": _register_answer_kind(),
            "du_mad": _install_du_mad(),
            "ford": _install_ford(),
            "moa": _install_moa(),
            "ansonly_render": _install_render(),
        },
        "adapter_files_modified": False,
        "execution_feedback_to_any_arm": False,
        "gold_sql_on_any_prompt": False,
        "prompt_rule": base["prompt_rule"],
        "why_not_dropped": ("du_mad / ford / moa had no SQL adaptation. The five-family comparison "
                            "is the design, so the adaptation was implemented and verified rather "
                            "than the protocols removed to fill the coverage table"),
    }
    _INSTALLED.update(out)
    return out


def verify() -> Dict[str, Any]:
    """Assert, against the live adapters, that every piece an SQL run needs is actually wired.
    """
    install()
    from consensus_delivery.adapters import bccs, du_mad, ford, mad_m2, moa
    bad = []
    q = "SELECT T1.a FROM t AS T1 WHERE T1.b = 1"

    def ck(cond, label):
        if not cond:
            bad.append(label)

    # answer-kind tables
    ck(AX.KIND_BY_BENCHMARK.get(BENCH) == KIND, "KIND_BY_BENCHMARK missing sql_bird_v2")
    ck(AX.nl_answer(f"```sql\n{q};\n```", KIND) == q, "nl tier cannot read a fenced query")
    ck(AX.nl_answer("I have no idea.", KIND) is None, "nl tier invents a query")
    ck(AX.is_well_formed(q, KIND) is True, "a query is not accepted as well formed")
    ck(AX.is_well_formed(f"Here it is: ```sql\n{q}```", KIND) is False,
       "prose containing a query passes the well-formedness gate")

    # each protocol's own contract extractor, called the way the condition transform calls it
    ck(bccs.extract_boxed("x \\boxed{%s} y" % q, KIND) == q, "bccs sql contract")
    ck(mad_m2.extract_answer("blah Answer: %s" % q, KIND) == q, "mad_m2 sql contract")
    ck(du_mad.extract_for(BENCH, "\\boxed{%s}" % q) == q, "du_mad sql contract")
    ck(du_mad.extract_for(BENCH, "no box here") is None, "du_mad sql contract invents an answer")
    ck(moa.extract("```sql\n%s\n```" % q, KIND) == q, "moa sql extraction")
    ck(ford.parse_prediction("Answer: %s\nExplanation: because.\n" % q, BENCH)[0] == q,
       "ford sql parse")
    ck(ford.parse_prediction("no answer at all", BENCH)[0] == "None", "ford sql parse sentinel")
    ck(ford.instruction(BENCH, 2) == FORD_INSTRUCTION_SQL, "ford sql instruction")

    # the released behaviour must be untouched for every other benchmark
    ck(ford.instruction("gpqa", 4).endswith('"Answer: (A|B|C|D) is more plausible." and explain '
                                            'very shortly starting with "Explanation: ". You '
                                            'should choose only one answer.'),
       "ford gpqa instruction changed by the sql install")
    ck(ford.parse_prediction("Answer: (C) is more plausible.\nExplanation: x.", "gpqa")[0] == "C",
       "ford gpqa parse changed by the sql install")
    ck(du_mad.extract_for("gpqa", "\\boxed{C}") == "C", "du_mad gpqa extraction changed")
    ck(moa.extract("the answer is (C).", "mcq") == "C", "moa mcq extraction changed")
    ck(AT._RENDER["ford"]("C") == "Answer: C is more plausible.", "ford render changed for mcq")
    ck(AT._RENDER["ford"](q) == "Answer: %s" % q, "ford render wrong for sql")

    # the question must carry no gold and no execution result, and no answer-placement instruction
    text = ST.sql_question("How many?", "CREATE TABLE t(a int)")
    ck("Return only the SQL query." not in text, "a format-dictating line is present")
    ck("boxed" not in text and "Answer:" not in text,
       "the question dictates where the answer goes, which is the protocol's contract")

    if bad:
        for b in bad:
            print("  FAIL " + b)
    print("ext_sql_tasks.verify: %s (%d checks)" % ("PASS" if not bad else "FAIL", 19))
    return {"ok": not bad, "failures": bad}


if __name__ == "__main__":
    import json
    import pathlib
    import sys
    _H = pathlib.Path(__file__).resolve().parent
    for _p in (_H.parents[2] / "code", _H):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    print(json.dumps(install(), indent=1)[:2000])
    raise SystemExit(0 if verify()["ok"] else 1)
