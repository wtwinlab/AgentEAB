r"""W6 diagnostic: is the selector's preference for C1 a position effect or a quality judgement?

    python3 code/w6_position.py     # -> output/w6_position.csv

Uses only the recorded picks and the existing candidate labels; no new call. C1 is always
qwen2.5-7b because presentation order is the frozen pool order, so position and model are
confounded by design. This separates them as far as the data allow: how often C1 is picked, how
often C1 is correct, and -- the informative case -- how often C1 is picked while C1 is wrong and
another offered candidate is right.
"""
import collections, csv, json, pathlib
W = pathlib.Path(__file__).resolve().parents[1]; R = W.parents[1]
POOL = ("qwen2.5-7b", "llama-3-8b", "qwen2.5-14b", "qwen2.5-32b")
lab = {}
for b in ("bird", "spider"):
    for l in open(R / "data/derived/sql_decomposition" / f"cand_{b}.jsonl"):
        r = json.loads(l); lab[(b, int(r["question_id"]), r["model"])] = bool(r["match"])
rows = []
for b in ("bird", "spider"):
    rs = [json.loads(l) for l in open(W / "raw/w6_selector_main.jsonl")]
    rs = [r for r in rs if r["benchmark"] == b]
    for cid, m in zip(("C1", "C2", "C3", "C4"), POOL):
        offered = [r for r in rs if cid in r["id_to_model"]]
        picked = [r for r in offered if r.get("picked_id") == cid]
        correct = [r for r in offered if lab.get((b, r["question_id"], m))]
        bad_pick = [r for r in picked if not lab.get((b, r["question_id"], m))
                    and any(lab.get((b, r["question_id"], x)) for x in r["id_to_model"].values())]
        rows.append(dict(benchmark=b, id=cid, model=m, offered=len(offered),
                         picked_pct=round(100 * len(picked) / len(offered), 2),
                         candidate_correct_pct=round(100 * len(correct) / len(offered), 2),
                         picked_while_wrong_and_another_right=len(bad_pick),
                         picked_while_wrong_and_another_right_pct_of_picks=
                         round(100 * len(bad_pick) / max(1, len(picked)), 2)))
with open(W / "output/w6_position.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(" ", r)
