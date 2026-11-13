#!/usr/bin/env python3
"""Offline acceptance for the SQL extension. Zero model calls; the databases are read only.

    python3 ext_test_sql_offline.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE, ROOT / "experiments/06_cost_resources/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import answer_extract as AX                                              # noqa: E402
import arm_transform as AT                                              # noqa: E402
import ext_design as D                                                   # noqa: E402

FAIL: List[str] = []
N = {"n": 0}


def ck(cond: Any, label: str) -> bool:
    N["n"] += 1
    if not cond:
        FAIL.append(label)
    return bool(cond)


# ---------------------------------------------------------------- A. prompts for all five
class _Captured(Exception):
    def __init__(self, messages):
        super().__init__("captured")
        self.messages = messages


def first_prompt(protocol: str, unit: str) -> List[Dict[str, str]]:
    import ext_unit_driver as EUD
    from consensus_delivery import providers
    EUD.ensure_installed("sql")
    real = providers.chat

    def stub(**kw):
        raise _Captured(list(kw.get("messages") or []))

    providers.chat = stub
    for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa"):
        try:
            __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"]).chat = stub
        except Exception:
            pass
    try:
        EUD.run_unit("sql_bird_v2", protocol, unit, D.ALIAS[D.PANELS[protocol][0]], arm="FULL")
        raise AssertionError("the stub did not intercept the first call")
    except _Captured as c:
        return c.messages
    finally:
        providers.chat = real
        for mod in ("bccs", "du_mad", "mad_m2", "ford", "moa"):
            try:
                __import__(f"consensus_delivery.adapters.{mod}", fromlist=["chat"]).chat = real
            except Exception:
                pass


def test_prompts() -> Dict[str, Any]:
    import sql_common as SQ
    f = D.frozen("sql_bird_v2")
    unit = f["unit_ids"][0]
    qid = int(unit.split("|")[1])
    q = next(x for x in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())
             if int(x["question_id"]) == qid)
    out = {}
    for proto in D.PROTOS:
        msgs = first_prompt(proto, unit)
        text = "\n".join(m.get("content") or "" for m in msgs)
        out[proto] = {"messages": len(msgs), "chars": len(text)}
        ck(q["question"] in text, f"A/{proto}: the question is not in the prompt")
        ck("CREATE TABLE" in text, f"A/{proto}: the schema is not in the prompt")
        ck(q["SQL"] not in text, f"A/{proto}: the GOLD QUERY is in the prompt")
        ck("Return only the SQL query." not in text,
           f"A/{proto}: a format-dictating line is in the prompt")
        ck("row(s)" not in text and "result set is empty" not in text,
           f"A/{proto}: execution feedback text is in the prompt")
        if proto == "ford":
            import ext_sql_tasks as EST
            ck(EST.FORD_INSTRUCTION_SQL in text,
               "A/ford: the SQL answer-format instruction is not appended to the view")
        if proto == "moa":
            import ext_sql_tasks as EST
            ck(EST.MOA_ANSWER_INSTRUCTION_SQL.strip() in text,
               "A/moa: the SQL answer instruction is not appended")
        if proto == "du_mad":
            import ext_sql_tasks as EST
            ck(EST.DU_ANSWER_CLAUSE.format(answer="answer") in text,
               "A/du_mad: the SQL answer clause is not in the opening")
    return out


# ---------------------------------------------------------------- B. the conditions on SQL prompts
GOOD = "SELECT T1.name FROM schools AS T1 WHERE T1.county = 'Alameda'"
PROSE_REPLY = ("The question asks for school names in Alameda county, so we filter on county.\n\n"
               "```sql\n" + GOOD + ";\n```\n\nThat returns the names.")
BOXED_REPLY = "Reasoning about the join...\n\\boxed{" + GOOD + "}"
ANSWER_REPLY = "Step by step we need a filter.\nAnswer: " + GOOD
FORD_REPLY = ("Answer: " + GOOD + "\nExplanation: the county column is on the schools table.")


def _peer_prompt(proto: str, bodies: List[str]) -> List[Dict[str, str]]:
    """A prompt in the protocol's own peer grammar, carrying `bodies` as rendered peers."""
    if proto in ("bccs", "du_mad"):
        blocks = "".join("\n\n One agent solution: ```%s```" % b for b in bodies)
        return [{"role": "user",
                 "content": "These are the solutions to the problem from other agents: " + blocks
                            + "\n\n Using the solutions from other agents as additional "
                              "information, can you provide your answer to the question? "}]
    if proto.startswith("madm2"):
        blocks = "".join("\n\n\nOne agent's solution: %s" % b for b in bodies)
        return [{"role": "user",
                 "content": "These are the solutions to the problem from other agents:" + blocks
                            + "\n\nUsing the solutions from other agents as additional "
                              "information, give your updated answer."}]
    if proto == "moa":
        return [{"role": "system",
                 "content": AT.MOA_MARKER + " from various open-source models to the latest user "
                            "query. Your task is to synthesize these responses into a single, "
                            "high-quality response.\n\n" + AT.MOA_REFS_HEADER
                            + "".join("\n%d. %s" % (i + 1, b) for i, b in enumerate(bodies))}]
    import ext_sql_tasks as EST
    return [{"role": "user", "content": "the question"},
            {"role": "user", "content": bodies[0] + "\n" + EST.FORD_INSTRUCTION_SQL}]


def test_arms() -> Dict[str, Any]:
    import ext_arms as EA
    import ext_sql_tasks as EST
    EST.install()
    EA.install()
    kind = "sql"
    out = {}
    replies = {"bccs": BOXED_REPLY, "du_mad": BOXED_REPLY, "madm2_objective": ANSWER_REPLY,
               "moa": PROSE_REPLY, "ford": FORD_REPLY}
    for proto in D.PROTOS:
        bodies = [replies[proto]] * (2 if proto != "ford" else 1)
        msgs = _peer_prompt(proto, bodies)
        if proto == "moa":
            EA.reset()
            for b in bodies:
                EA.observe_reply(b)
        _n, tr = AT.transform(proto, "FULL", msgs, kind=kind, benchmark="sql_bird_v2")
        ck(tr["peer_blocks_found"] == len(bodies),
           f"B/{proto}: FULL found {tr['peer_blocks_found']} peer blocks, expected {len(bodies)}")

        if proto == "moa":
            EA.reset()
            for b in bodies:
                EA.observe_reply(b)
        new, tr = AT.transform(proto, "NOPEER", msgs, kind=kind, benchmark="sql_bird_v2")
        ck(tr["n_peers_rendered"] == 0, f"B/{proto}: NOPEER left peers rendered")
        txt = "\n".join(m.get("content") or "" for m in new)
        ck(GOOD not in txt, f"B/{proto}: NOPEER left the peer's query in the prompt")
        if proto == "ford":
            ck(EST.FORD_INSTRUCTION_SQL in txt,
               "B/ford: NOPEER dropped the SQL answer-format instruction")

        if proto == "moa":
            EA.reset()
            for b in bodies:
                EA.observe_reply(b)
        new, tr = AT.transform(proto, "ANSONLY", msgs, kind=kind, benchmark="sql_bird_v2")
        applied = tr["ansonly_tier"]["contract"] + tr["ansonly_tier"]["nl"]
        ck(applied == len(bodies),
           f"B/{proto}: ANSONLY applied to {applied} of {len(bodies)} blocks "
           f"(failed {tr['ansonly_extract_failed']})")
        txt = "\n".join(m.get("content") or "" for m in new)
        ck(GOOD in txt, f"B/{proto}: ANSONLY dropped the query instead of keeping it")
        # FORD re-attaches its own answer-format instruction, which itself names "Explanation: ";
        # that is protocol text, not peer prose, so it is removed before the prose check.
        peer_txt = txt.replace(EST.FORD_INSTRUCTION_SQL, "") if proto == "ford" else txt
        ck("Explanation" not in peer_txt and "Reasoning" not in peer_txt
           and "question asks" not in peer_txt,
           f"B/{proto}: ANSONLY kept the peer's prose")
        before = sum(len(m.get("content") or "") for m in msgs)
        after = sum(len(m.get("content") or "") for m in new)
        ck(after < before, f"B/{proto}: ANSONLY did not shrink the prompt")
        if proto == "ford":
            ck("Answer: %s" % GOOD in txt,
               "B/ford: ANSONLY did not render the query in FORD's answer form")
            ck("is more plausible" not in txt.split("Explanation")[0],
               "B/ford: the choice-style sentence was used for a query")

        if proto != "ford":
            if proto == "moa":
                EA.reset()
                for b in bodies:
                    EA.observe_reply(b)
            _new, tr = AT.transform(proto, "UNIFSEL", msgs, kind=kind, benchmark="sql_bird_v2",
                                    seed=11)
            ck(tr["n_peers_rendered"] < len(bodies),
               f"B/{proto}: UNIFSEL kept every peer")
        out[proto] = {"blocks": len(bodies), "ansonly_applied": applied,
                      "size_ratio": round(after / before, 4)}

    # the answer kind itself
    ck(AX.nl_answer(PROSE_REPLY, kind) == GOOD, "B: the SQL nl tier cannot read a fenced query")
    ck(AX.is_well_formed(GOOD, kind), "B: a query is not well formed")
    ck(not AX.is_well_formed(PROSE_REPLY, kind), "B: prose with a query passes well-formedness")
    ck(AX.nl_answer("I cannot write this query.", kind) is None, "B: the nl tier invents a query")
    return out


# ---------------------------------------------------------------- C. execution
def test_execution() -> Dict[str, Any]:
    import ext_eval_sql as EV
    db = "california_schools"
    import sql_common as SQ
    # a query that is certainly valid on this instance, used as both candidate and reference
    valid = "SELECT County FROM schools LIMIT 5"
    r_match = EV.execute_isolated(db, valid, valid)
    ck(r_match["outcome"] == "match", f"C: identical queries scored {r_match}")
    r_mis = EV.execute_isolated(db, "SELECT County FROM schools LIMIT 5",
                                "SELECT County FROM schools LIMIT 7")
    ck(r_mis["outcome"] in ("match", "mismatch"), f"C: two valid queries scored {r_mis}")
    r_syn = EV.execute_isolated(db, "SELEC County FROM schools", valid)
    ck(r_syn["outcome"] == "syntax_error", f"C: a syntax error classified as {r_syn}")
    r_mis2 = EV.execute_isolated(db, "SELECT x FROM no_such_table_here", valid)
    ck(r_mis2["outcome"] == "missing_object", f"C: a missing table classified as {r_mis2}")
    # read-only: a write must be refused by the database, not by convention
    r_w = EV.execute_isolated(db, "CREATE TABLE ext_probe_should_not_exist(x int)", valid)
    ck(r_w["outcome"] in ("syntax_error", "other_exec_error", "missing_object"),
       f"C: a CREATE TABLE was not refused: {r_w}")
    ck("readonly" in str(r_w.get("error", "")).lower()
       or "not authorized" in str(r_w.get("error", "")).lower(),
       f"C: the write failed for a reason other than read-only: {r_w.get('error')}")
    # and it really did not write
    con = SQ._connect(db)
    try:
        names = {n for (n,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    finally:
        con.close()
    ck("ext_probe_should_not_exist" not in names, "C: the probe table was created -- NOT read only")
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
               "SELECT count(*) FROM c")
    t0 = __import__("time").time()
    r_to = EV.execute_isolated(db, runaway, valid)
    el = __import__("time").time() - t0
    ck(r_to["outcome"] == "exec_timeout", f"C: a runaway query scored {r_to}")
    ck(el < EV.HARD_TIMEOUT_S + 10,
       f"C: the runaway query took {el:.1f}s, past the parent's {EV.HARD_TIMEOUT_S}s limit")
    # the taxonomy must be total
    for msg, want in (("near \"SELEC\": syntax error", "syntax_error"),
                      ("no such column: foo", "missing_object"),
                      ("misuse of aggregate: count()", "type_or_arg_error"),
                      ("something nobody predicted", "other_exec_error")):
        ck(EV.classify_error(msg) == want, f"C: classify_error({msg!r}) != {want}")
    return {"match": r_match["outcome"], "syntax": r_syn["outcome"],
            "missing": r_mis2["outcome"], "write_refused": str(r_w.get("error"))[:80],
            "runaway": r_to["outcome"], "runaway_seconds": round(el, 2)}


# ---------------------------------------------------------------- D. no execution in collection
def test_collection_cannot_execute() -> Dict[str, Any]:
    import ext_unit_driver as EUD
    EUD.ensure_installed("sql")
    import sql_common as SQ
    out = {}
    for name in ("execute", "gold_sql", "execution_match"):
        fn = getattr(SQ, name)
        try:
            fn("california_schools", "SELECT 1", "SELECT 1")
            out[name] = "DID NOT RAISE"
        except RuntimeError as exc:
            out[name] = "raised: " + str(exc)[:60]
        except TypeError:
            out[name] = "raised: TypeError (poisoned, arity-agnostic)"
        ck(out[name] != "DID NOT RAISE",
           f"D: sql_common.{name} is callable inside a collection process")
    # the schema read must still work: a policy may see a schema
    ck(len(SQ.schema_text("california_schools")) > 100,
       "D: schema_text was poisoned too, so no SQL prompt could be built")
    return out


def main() -> int:
    report: Dict[str, Any] = {"benchmark": "sql_bird_v2", "model_calls": 0}
    print("== A. all five protocols build a SQL prompt (3 of them had no SQL adaptation before)")
    report["prompts"] = test_prompts()
    for p, v in report["prompts"].items():
        print("   %-16s %d message(s), %s chars" % (p, v["messages"], f"{v['chars']:,}"))
    print("== B. the arms on SQL peer prompts")
    report["arms"] = test_arms()
    for p, v in report["arms"].items():
        print("   %-16s blocks %d  ANSONLY applied %d  size ratio %s"
              % (p, v["blocks"], v["ansonly_applied"], v["size_ratio"]))
    print("== C. read-only, bounded, isolated execution and the error taxonomy")
    report["execution"] = test_execution()
    e = report["execution"]
    print("   match=%s syntax=%s missing=%s runaway=%s in %.1fs; write refused: %s"
          % (e["match"], e["syntax"], e["missing"], e["runaway"], e["runaway_seconds"],
             e["write_refused"]))
    print("== D. no execution path inside a collection process")
    report["poison"] = test_collection_cannot_execute()
    for k, v in report["poison"].items():
        print("   %-16s %s" % (k, v))

    report["checks"] = N["n"]
    report["failures"] = FAIL
    out = D.OUT / "ACCEPTANCE_sql_offline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print()
    for f in FAIL:
        print("  FAIL " + f)
    print("ext_test_sql_offline: %s (%d checks) -> %s"
          % ("PASS" if not FAIL else "FAIL", N["n"], out.relative_to(ROOT)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
