r"""Score the DeepSeek-V4-Pro arms. Offline; read-only DBs via the existing executors.

    python3 code/score_ds.py <raw_Sal.jsonl> <raw_G.jsonl> <out_per_question.csv>

S_al: `score_sal.parse` on `content` (final channel) -> picked pool model -> stored candidate label.
G: `final_channel` + benchmark `parse_sql` -> `execution_match` against the reference (gold read
only here). Non-selection, unparsed, error or truncation without an answer scores 0; flags kept.
"""
import collections
import csv
import json
import pathlib
import subprocess
import sys

sys.dont_write_bytecode = True
W = pathlib.Path(__file__).resolve().parents[1]
REPO = W.parents[1]
PT = REPO / "experiments/common/code"
sys.path.insert(0, str(PT))
import score_sal as SS                                                     # noqa: E402
import build_inputs as BI                                                  # noqa: E402

raw_sal, raw_g, out = sys.argv[1:4]
labels = {}
for b in ("bird", "spider"):
    for r in map(json.loads, open(SS.DEC / f"cand_{b}.jsonl")):
        labels[(b, int(r["question_id"]), r["model"])] = r.get("match") is True
inp = {json.loads(l)["unit"]: json.loads(l) for l in open(W / "data/ds_inputs_Sal_P0.jsonl")}


def last(p):
    d = {}
    for r in map(json.loads, open(p)):
        if r["unit_id"] not in d or r["status"] == "ok":
            d[r["unit_id"]] = r
    return d


def content(r):
    ch = ((r.get("response") or {}).get("choices") or [{}])[0]
    return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason")


rows = {}
for u, r in last(raw_sal).items():
    i = inp[u]
    txt, fin = content(r) if r["status"] == "ok" else ("", "error")
    pm, why = SS.parse(BI.final_channel(txt), i["id_to_model"]) if r["status"] == "ok" else (None, "error")
    k = (i["benchmark"], i["question_id"])
    usage = (r.get("response") or {}).get("usage") or {}
    rows.setdefault(k, dict(benchmark=k[0], question_id=k[1], db_id=i["db_id"]))
    rows[k].update(DS_S_al=int(bool(pm) and labels.get((k[0], k[1], pm), False)), DS_S_al_pick=pm, DS_S_al_why=why,
                   DS_S_al_finish=fin, DS_S_al_in=usage.get("prompt_tokens"), DS_S_al_out=usage.get("completion_tokens"))
G = last(raw_g)
for b in ("bird", "spider"):
    S = BI.sqlmod(b)
    gold = {int(q["question_id"]): (q["db_id"], q["SQL"]) for q in S.load_questions()}
    for u, r in G.items():
        if f"|{b}|" not in u:
            continue
        q = int(u.split("|")[2])
        txt, fin = content(r) if r["status"] == "ok" else ("", "error")
        sql = S.parse_sql(BI.final_channel(txt)) if r["status"] == "ok" else None
        db, g = gold[q]
        m = bool(sql) and bool(S.execution_match(db, sql, g).get("match") is True)
        usage = (r.get("response") or {}).get("usage") or {}
        rows.setdefault((b, q), dict(benchmark=b, question_id=q, db_id=db))
        rows[(b, q)].update(DS_Generation=int(m), DS_G_parsed=bool(sql), DS_G_finish=fin,
                            DS_G_in=usage.get("prompt_tokens"), DS_G_out=usage.get("completion_tokens"))
with open(out, "w", newline="") as f:
    keys = list({k: None for r in rows.values() for k in r})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for k in sorted(rows):
        w.writerow(rows[k])
c = collections.Counter()
for r in rows.values():
    c["sal_nonselect"] += r.get("DS_S_al_pick") is None and "DS_S_al" in r
    c["sal_length"] += r.get("DS_S_al_finish") == "length"
    c["g_unparsed"] += r.get("DS_G_parsed") is False
    c["g_length"] += r.get("DS_G_finish") == "length"
print(len(rows), dict(c), "S_al", sum(r.get("DS_S_al", 0) for r in rows.values()), "G", sum(r.get("DS_Generation", 0) for r in rows.values()))
