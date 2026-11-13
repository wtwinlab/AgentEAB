r"""Freeze this round's inputs for one benchmark. Read-only SQL execution, no model call.

    python3 code/build_inputs.py --bench bird
    python3 code/build_inputs.py --bench spider      # separate process: spider_common redirects sql_common

Writes
  data/requests_{B,C,G}_<bench>.jsonl     Qwen3/GLM-pool requests, one per question
  data/newpool_candidates_<bench>.jsonl   parsed SQL + execution record per slot (prompt input)
  data/newpool_labels_<bench>.jsonl       execution_match per slot -- SCORING ONLY, never read by a builder
  logs/build_<bench>.json                 contract checks and counts

Contract checks, fatal on failure:
  1. this module's selector builder, applied to the original pool, reproduces every W6 arm-B prompt
  2. this module's generation builder, applied to the original pool, reproduces every stored Qwen3
     generation prompt
  3. the Qwen3/GLM pool rules re-derived here equal the stored strongpool_per_question.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
W3 = REPO / "experiments/15_stronger_peers"
for p in ("experiments/04_sql_selection_generation/code", "experiments/02_controlled_sharing/code",
          "experiments/common/code"):
    sys.path.insert(0, str(REPO / p))
sys.path.insert(0, str(W3 / "code"))

STRONG = REPO / "data/derived/sql_rule_pairing/data/strong_pool/output/raw"
STRONG_RULES = REPO / "data/derived/sql_rule_pairing/output/strongpool_per_question.csv"
GEN_IN = REPO / "experiments/08_cross_family/data"
GEN_RAW = REPO / "experiments/08_cross_family/output/raw"
SLOTS = ("glm47-flash|s0", "glm47-flash|s1", "qwen3-32b|s0", "qwen3-32b|s1")   # sorted base order
PERM_SEED = 20261101
RULES = ("released", "valid_majority", "nonempty_majority")


def sqlmod(bench):
    import sql_common as S
    if bench == "bird":
        S.DBDIR = "${BIRD_DB_DIR}"        # ASCII path; sqlite URIs reject the repo path
        return S
    import spider_common as SP
    SP.DBDIR = "${SPIDER_DB_DIR}"
    return SP


def final_channel(content: str) -> str:
    """As the final_channel rule of the cross-family SQL rescoring."""
    c = content or ""
    for tag in ("</think>", "</thinking>"):
        if tag in c:
            c = c[c.rindex(tag) + len(tag):]
    return c.strip()


def selector_prompt(Asel, bench, q, cands_in_base_order):
    """w6_selector.PROMPT/feedback with the w6_arms.permuted permutation. Returns (prompt, id_map, order)."""
    live = [c for c in cands_in_base_order if (c.get("sql") or "").strip()]
    order = [c["model"] for c in live]
    random.Random(f"{PERM_SEED}|{bench}|{q['question_id']}").shuffle(order)
    by = {c["model"]: c for c in live}
    blocks, mapping = [], {}
    for i, m in enumerate(order):
        cid = f"C{i + 1}"
        mapping[cid] = m
        rec = by[m]
        blocks.append(f"{cid}:\n  SQL: {(rec.get('sql') or '').strip()}\n"
                      f"  {Asel.feedback(rec.get('exec'), rec.get('exec_status'))}\n")
    ev = q.get("evidence")
    prompt = Asel.PROMPT.format(db_id=q["db_id"], question=(q.get("question") or "").strip(),
                                evidence=f"\nHint: {str(ev).strip()}" if ev else "",
                                candidates="\n".join(blocks), first="C1")
    return prompt, mapping, order, len(live)


def generation_prompt(S, A, q, cands_in_base_order):
    pl = [c for c in cands_in_base_order if (c.get("sql") or "").strip()]
    if len(pl) < 2:
        return None, len(pl)
    return S.render(q) + A.PEER_FRAME.format(block=A.peer_block(pl, "F3")), len(pl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=("bird", "spider"))
    bench = ap.parse_args().bench
    S = sqlmod(bench)
    import collect_sql_arms as A
    import b1_selectors as B1
    import w6_selector as Asel
    qs = {int(q["question_id"]): q for q in S.load_questions()}
    log = dict(benchmark=bench)

    # --- check 1 and 2 on the original pool
    peers = B1.load_peers(bench)
    w6b = {int(r["question_id"]): r["messages"][0]["content"]
           for r in map(json.loads, open(W3 / "raw/w6_armB_main.jsonl")) if r["benchmark"] == bench}
    ok1 = bad1 = 0
    for qid, p in w6b.items():
        cands = [dict(p_, model=m) for m, p_ in ((m, peers[qid][m]) for m in Asel.POOL if m in peers[qid])]
        pr, _, _, _ = selector_prompt(Asel, bench, qs[qid], cands)
        ok1 += pr == p
        bad1 += pr != p
    gin = {r["unit"]: r for r in map(json.loads, open(GEN_IN / f"sql_{bench}_inputs.jsonl"))
           if r["recipient"] == "qwen3-32b"}
    graw = {r["unit_id"]: r for r in map(json.loads, open(GEN_RAW / f"sql_{bench}_qwen3-32b.jsonl"))}
    ok2 = bad2 = 0
    gen_seed = {}
    for u, r in gin.items():
        qid = r["question_id"]
        cands = [dict(peers[qid][m], model=m) for m in A.PEER_MODELS if m in peers[qid]]
        pr, _ = generation_prompt(S, A, qs[qid], cands)
        ok2 += pr == r["messages"][0]["content"]
        bad2 += pr != r["messages"][0]["content"]
        gen_seed[qid] = graw[u]["request_body"]["seed"]
    log.update(check1_selector_prompts_identical=ok1, check1_differing=bad1,
               check2_generation_prompts_identical=ok2, check2_differing=bad2)
    print(log, flush=True)
    assert bad1 == 0 and ok1 == len(w6b), "check 1 failed"
    assert bad2 == 0 and ok2 == len(gin), "check 2 failed"

    # --- Qwen3/GLM pool: parse, execute read-only, label (scoring only)
    gold = {int(q["question_id"]): q["SQL"] for q in qs.values()}
    pool = collections.defaultdict(dict)
    acq = collections.Counter()
    for fam in ("glm47-flash", "qwen3-32b"):
        for r in map(json.loads, open(STRONG / f"cand_{bench}_{fam}.jsonl")):
            assert r["status"] == "ok"
            pool[int(r["question_id"])][f"{fam}|s{r['sample_index']}"] = r
            u = r.get("usage") or {}
            acq["calls"] += 1; acq["input"] += u.get("prompt_tokens") or 0
            acq["output"] += u.get("completion_tokens") or 0
    log["newpool_acquisition"] = dict(acq)
    cand_rows, lab_rows = [], []
    cands_by_q = {}
    for qid in sorted(qs):
        db = qs[qid]["db_id"]
        cl = []
        for slot in SLOTS:
            r = pool[qid].get(slot)
            sql = S.parse_sql(final_channel(r["final_content"])) if r else None
            if not sql:
                c = dict(model=slot, sql="", exec={"status": "unparsed", "n_rows_returned": 0},
                         exec_status="unparsed")
                match = False
            else:
                ex = S.execute(db, sql)
                c = dict(model=slot, sql=sql, exec=ex, exec_status=ex.get("status"))
                match = bool(S.execution_match(db, sql, gold[qid]).get("match") is True)
            cl.append(c)
            cand_rows.append(dict(benchmark=bench, question_id=qid, db_id=db, **c))
            lab_rows.append(dict(benchmark=bench, question_id=qid, db_id=db, model=slot, match=match))
        cands_by_q[qid] = cl

    # --- check 3: rules re-derived equal the stored ones
    labels = {(r["question_id"], r["model"]): r["match"] for r in lab_rows}
    stored = {int(r["question_id"]): r for r in csv.DictReader(open(STRONG_RULES)) if r["benchmark"] == bench}
    mism = collections.Counter()
    for qid, cl in cands_by_q.items():
        live = [c for c in cl if c["sql"]]
        st = stored[qid]
        avail = int(any(labels[(qid, c["model"])] for c in cl))
        mism["pool_had_correct"] += avail != int(st["pool_had_correct"])
        for s in RULES:
            pick = B1.SELECTORS[s](live, None) if len(live) >= 2 else None
            corr = int(bool(pick) and labels[(qid, pick["model"])])
            mism[s] += corr != int(st[f"{s}_correct"])
    log["check3_rule_mismatches"] = dict(mism)
    print("check3", dict(mism), flush=True)
    assert not any(mism.values()), "check 3 failed"

    # --- requests
    reqB, reqC, reqG, skipped = [], [], [], collections.Counter()
    for qid in sorted(qs):
        q, cl = qs[qid], cands_by_q[qid]
        pr, mapping, order, nlive = selector_prompt(Asel, bench, q, cl)
        base = dict(benchmark=bench, question_id=qid, db_id=q["db_id"], pool="qwen3_glm")
        if nlive >= 1:
            for arm, lst in (("B", reqB), ("C", reqC)):
                lst.append(dict(base, unit_id=f"pool2|{arm}|{bench}|{qid}", arm=arm, n_candidates=nlive,
                                id_to_model=mapping, order=order,
                                messages=[{"role": "user", "content": pr}], seed=PERM_SEED))
        else:
            skipped["selector_no_candidate"] += 1
        gp, npl = generation_prompt(S, A, q, cl)
        if gp is None:
            skipped["generation_fewer_than_two"] += 1
        else:
            reqG.append(dict(base, unit_id=f"pool2|G|{bench}|{qid}", arm="G", n_peers_shown=npl,
                             messages=[{"role": "user", "content": gp}], seed=gen_seed[qid]))
    log.update(questions=len(qs), requests=dict(B=len(reqB), C=len(reqC), G=len(reqG)),
               skipped=dict(skipped),
               exec_status=dict(collections.Counter(c["exec_status"] for c in cand_rows)),
               newpool_available=sum(any(labels[(q, s)] for s in SLOTS) for q in qs))
    D = WORK / "data"
    D.mkdir(exist_ok=True)
    for name, rows in ((f"requests_B_{bench}", reqB), (f"requests_C_{bench}", reqC),
                       (f"requests_G_{bench}", reqG), (f"newpool_candidates_{bench}", cand_rows),
                       (f"newpool_labels_{bench}", lab_rows)):
        with open(D / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (WORK / "logs").mkdir(exist_ok=True)
    (WORK / "logs" / f"build_{bench}.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
