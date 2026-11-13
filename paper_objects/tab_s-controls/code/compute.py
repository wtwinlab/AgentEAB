r"""Supplementary Table tab:s-controls and the control paragraphs of Sections VI-C and S6-E (sec:s-controls):
full BIRD dev, difficulty strata, leave-one-database-out, minimum detectable effects, placebo peers,
Du-MAD at temperature 0.7, the unified SQL answer format, and initial-answer agreement of paired runs.

    python3 code/compute.py --out output

Reads only ../inputs. Writes <out>/values.json and <out>/controls.json (every contrast with both intervals).
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402

INP = HERE.parent / "inputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    R = E.Rows(INP)
    V, full = {}, {}
    OR = lambda arm: ("orig", arm)                                         # noqa: E731
    RX = lambda arm: ("rx", arm)                                           # noqa: E731
    U198 = R.U198

    def put(key, b):
        V[key] = b["est"]
        V[key + "_q_lo"], V[key + "_q_hi"] = b["q_ci"]
        V[key + "_db_lo"], V[key + "_db_hi"] = b["db_ci"]
        full[key] = {k: b[k] for k in ("est", "q_ci", "db_ci", "n_questions", "detected_both", "mde80_q", "mde80_db")}

    # ---- interaction effect and last-round difference on BIRD dev
    put("1333|4cell|dpeer", R.boot(R.fulldev_items(E.C4, "E1", sets=("new",))))
    for g, cells in E.GROUPS.items():
        put(f"1531|{g}|dpeer", R.boot(R.fulldev_items(cells, "E1")))
        put(f"1531|{g}|last_diff", R.boot(R.fulldev_items(cells, "last")))
    put("1531|4cell|r0_diff", R.boot(R.fulldev_items(E.C4, "r0")))
    V["mde80_q_1531"] = full["1531|4cell|dpeer"]["mde80_q"]
    V["mde80_db_1531"] = full["1531|4cell|dpeer"]["mde80_db"]
    for lev in ("simple", "moderate", "challenging"):
        sub = [u for u in R.NEWU if R.DIFF.get(u) == lev]
        sub2 = [u for u in U198 if R.DIFF.get(u) == lev]
        b = R.boot(R.fulldev_items(E.C4, "E1", new_units=sub, u198_units=sub2))
        put(f"stratum_{lev}", b)
        V[f"stratum_{lev}_n"] = b["n_questions"]
    lodo = {}
    for db in sorted({R.DB[u] for u in R.NEWU + U198}):
        b = R.boot(R.fulldev_items(E.C4, "E1", new_units=[u for u in R.NEWU if R.DB[u] != db],
                                   u198_units=[u for u in U198 if R.DB[u] != db]))
        lodo[db] = b
    V["lodo1531_min"] = min(b["est"] for b in lodo.values())
    V["lodo1531_max"] = max(b["est"] for b in lodo.values())
    V["lodo1531_n_both_excl0"] = sum(E.excl0(b["q_ci"]) and E.excl0(b["db_ci"]) for b in lodo.values())
    full["lodo_1531"] = {k: {x: b[x] for x in ("est", "q_ci", "db_ci")} for k, b in lodo.items()}

    # ---- 198 sampled questions (per-run scores of Table IV)
    S = E.Sample198(INP)
    for term in ("last_diff", "dpeer"):
        for g in ("4cell", "DuMAD_2cell"):
            put(f"198|{g}|{term}", S.evaluate(term, E.GROUPS[g]))
    b = S.evaluate("dpeer", E.C4)
    V["mde80_q_198"], V["mde80_db_198"] = b["mde80_q"], b["mde80_db"]
    cell_mde = [S.evaluate("dpeer", [c]) for c in E.C4]
    V["cell_mde80_min_198"] = min(min(x["mde80_q"], x["mde80_db"]) for x in cell_mde)
    V["cell_mde80_max_198"] = max(max(x["mde80_q"], x["mde80_db"]) for x in cell_mde)
    lo198 = S.lodo()
    V["lodo198_min"] = min(x["est"] for x in lo198.values())
    V["lodo198_max"] = max(x["est"] for x in lo198.values())
    V["lodo198_n_db_excl0"] = sum(x["db_ci"][1] < 0 for x in lo198.values())
    V["lodo198_n_q_excl0"] = sum(x["q_ci"][1] < 0 for x in lo198.values())
    V["n_databases"] = len(lo198)
    full["lodo_198"] = lo198

    # ---- placebo peers (198; change from initial answers to the last round)
    put("placebo|4cell|real_minus_placebo", R.boot(R.diff_items(OR("FULL"), RX("PLACEBO"), E.C4, "E1", U198)))
    put("placebo|4cell|placebo_minus_nopeer", R.boot(R.diff_items(RX("PLACEBO"), R.nop_198, E.C4, "E1", U198)))
    for c in E.C4:
        lab = f"{c[0]}|{c[1]}"
        put(f"placebo|{lab}|real_minus_placebo", R.boot(R.diff_items(OR("FULL"), RX("PLACEBO"), [c], "E1", U198)))
        put(f"placebo|{lab}|placebo_minus_nopeer", R.boot(R.diff_items(RX("PLACEBO"), R.nop_198, [c], "E1", U198)))
    ratios = []
    for c in E.C4:
        a = R.per_unit("rx", c, "PLACEBO", "r1_prompt_tokens", U198)
        bb = R.per_unit("orig", c, "FULL", "r1_prompt_tokens", U198)
        com = set(a) & set(bb)
        ratios.append(float(np.mean([a[u] / max(1, bb[u]) for u in com])))
    V["placebo_token_ratio_min"] = round(min(ratios), 3)
    V["placebo_token_ratio_max"] = round(max(ratios), 3)

    # ---- Du-MAD at temperature 0.7 (198)
    put("t07|DuMAD_2cell|dpeer", R.boot(R.diff_items(RX("T07_FULL"), RX("T07_NOPEERC"), E.DU, "E1", U198)))
    for c in E.DU:
        put(f"t07|{c[1]}|dpeer", R.boot(R.diff_items(RX("T07_FULL"), RX("T07_NOPEERC"), [c], "E1", U198)))
    put("t07|DuMAD_2cell|diverse_share_change", R.boot(R.diff_items(RX("T07_FULL"), OR("FULL"), E.DU, "diverse", U198)))
    strat = []
    for c in E.DU:
        d = R.IDX[("rx", c[0], c[1], "T07_FULL")]
        dv = {u: any(r["r0_diverse"] for r in d[u]) for u in U198 if d.get(u)}
        strat += [(u, v) for u, v in R.diff_items(RX("T07_FULL"), RX("T07_NOPEERC"), [c], "E1", U198) if dv.get(u)]
    put("t07|DuMAD_2cell|dpeer_drafts_disagree", R.boot(strat))
    for c in E.DU:
        for src, arm, tag in (("orig", "FULL", "greedy"), ("rx", "T07_FULL", "t07")):
            d = R.per_unit(src, c, arm, "diverse", U198)
            V[f"diverse_share|{c[1]}|{tag}"] = E.r2(np.mean(list(d.values())))

    # ---- unified SQL answer format (198)
    put("fmt|4cell|final_change", R.boot(R.diff_items(RX("FMT_FULL"), OR("FULL"), E.C4, "final", U198)))
    put("fmt|4cell|dpeer_change", R.boot(R.did_items(
        RX("FMT_FULL"), lambda c: RX("FMT_NOPEERC") if c[0] == "du_mad" else RX("FMT_NOPEER"),
        OR("FULL"), lambda c: RX("NOPEERC") if c[0] == "du_mad" else OR("NOPEER"), E.C4, U198)))
    for c in E.C4:
        lab = f"{c[0]}|{c[1]}"
        put(f"fmt|{lab}|final_change", R.boot(R.diff_items(RX("FMT_FULL"), OR("FULL"), [c], "final", U198)))
        V[f"fmt|{lab}|r0_reader_share_change"] = R.boot(
            R.diff_items(RX("FMT_FULL"), OR("FULL"), [c], "r0_strict_share", U198))["est"]
        V[f"fmt|{lab}|final_level_fmt"] = E.r2(np.mean(list(R.per_unit("rx", c, "FMT_FULL", "final", U198).values())))
        V[f"fmt|{lab}|final_level_orig"] = E.r2(np.mean(list(R.per_unit("orig", c, "FULL", "final", U198).values())))

    # ---- share of runs whose initial queries return >= 2 distinct execution results (198, greedy)
    for c in E.C4:
        lab = f"{c[0]}|{c[1]}"
        w = R.per_unit("orig", c, "FULL", "diverse", U198)
        V[f"r0_diverse_with|{lab}"] = E.r2(np.mean(list(w.values())))
        nop = R.nop_198(c)
        o = R.per_unit(nop[0], c, nop[1], "diverse", U198)
        V[f"r0_diverse_pooled|{lab}"] = E.r2(np.mean([(w[u] + o[u]) / 2 for u in w]))

    # ---- later-round prompt length per call, with vs without peers (198)
    for c in E.C4:
        lab = f"{c[0]}|{c[1]}"
        for src, arm, tag in (("orig", "FULL", "with"), (*R.nop_198(c), "without")):
            rs = [r for us in R.IDX[(src, c[0], c[1], arm)].values() for r in us if r["unit"] in set(U198)]
            V[f"later_prompt_tokens|{lab}|{tag}"] = round(float(np.mean(
                [r["r1_prompt_tokens"] / max(1, r["n_calls"] // 2) for r in rs])), 1)
        V[f"later_prompt_shorter_pct|{lab}"] = round(
            100 * (1 - V[f"later_prompt_tokens|{lab}|without"] / V[f"later_prompt_tokens|{lab}|with"]), 2)

    # ---- initial-answer agreement of paired runs (with peers vs the no-peer records of the same cell)
    raw = {}
    for l in (INP / "round0_pairs.jsonl").open():
        d = json.loads(l)
        raw[tuple(d["key"])] = d
    rows = {}
    for r in R.rows:
        rows[(r["source"], r["unit"], r["protocol"], r["arm"], r["backbone"], int(r["repeat"]))] = r
    agree = {}
    for c in E.C4:
        pairs = {"new": (("rx", "FULL"), R.nop_new(c), R.NEWU), "u198": (("orig", "FULL"), R.nop_198(c), U198)}
        for pname, (wa, wb, units) in pairs.items():
            st = collections.Counter()
            for u in units:
                for rep in (1, 2):
                    ka, kb = (wa[0], u, c[0], wa[1], c[1], rep), (wb[0], u, c[0], wb[1], c[1], rep)
                    a, b, ra, rb = raw.get(ka), raw.get(kb), rows.get(ka), rows.get(kb)
                    if not (a and b and ra and rb):
                        st["missing"] += 1
                        continue
                    st["n"] += 1
                    st["prompts_equal"] += a["prompts"] == b["prompts"]
                    st["calls"] += 3
                    st["calls_reply_equal"] += sum(x == y for x, y in zip(a["replies"], b["replies"]))
                    st["r0_equal"] += bool(ra["r0"]) == bool(rb["r0"])
            agree[f"{c[0]}|{c[1]}|{pname}"] = dict(
                st, prompts_equal_pct=round(100 * st["prompts_equal"] / st["n"], 2),
                calls_reply_equal_pct=round(100 * st["calls_reply_equal"] / st["calls"], 2),
                r0_equal_pct=round(100 * st["r0_equal"] / st["n"], 2))
    full["round0_agreement"] = agree
    du = [v for k, v in agree.items() if k.startswith("du_mad")]
    m2 = [v for k, v in agree.items() if k.startswith("madm2")]
    V["r0_reply_identical_du_min"] = min(v["calls_reply_equal_pct"] for v in du)
    V["r0_reply_identical_du_max"] = max(v["calls_reply_equal_pct"] for v in du)
    V["r0_correct_agree_du_min"] = min(v["r0_equal_pct"] for v in du)
    V["r0_correct_agree_du_max"] = max(v["r0_equal_pct"] for v in du)
    V["r0_correct_agree_m2_min"] = min(v["r0_equal_pct"] for v in m2)
    V["r0_correct_agree_m2_max"] = max(v["r0_equal_pct"] for v in m2)
    V["r0_prompts_identical_min"] = min(v["prompts_equal_pct"] for v in agree.values())
    V["r0_missing_pairs"] = sum(v.get("missing", 0) for v in agree.values())

    (out / "values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    (out / "controls.json").write_text(json.dumps(full, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
