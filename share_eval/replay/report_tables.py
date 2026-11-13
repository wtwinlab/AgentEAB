r"""Render markdown tables from factorial_replay.json.

    python3 -I share_eval/replay/report_tables.py [--out output/share_eval/replay]

Writes <out>/tables.md and refreshes <out>/SHA256SUMS. No computation beyond formatting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
NAMES = {"du_mad": "Du-MAD", "madm2_objective": "MAD-M²", "bccs": "BCCS", "moa": "MoA"}


def f(x, bold=True):
    if x is None:
        return "—"
    s = f"{x['est']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]"
    return f"**{s}**" if bold and (x["lo"] > 0 or x["hi"] < 0) else s


def nm(k):
    p, b = k.split("|")
    return f"{NAMES.get(p, p)} / {b}" if p != "POOLED_interactive" else "mean of 6 interactive cells"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "output/share_eval/replay"),
                    help="directory holding factorial_replay.json (default %(default)s)")
    out = pathlib.Path(ap.parse_args().out)
    B = json.loads((out / "factorial_replay.json").read_text())
    cells = dict(B["cells"])
    order = [k for k in cells if not k.startswith("moa")] + [k for k in cells if k.startswith("moa")]
    prim = {k: cells[k]["primary"] for k in order}
    if B.get("pooled_interactive"):
        prim["POOLED_interactive|-"] = B["pooled_interactive"]
        order = order[:-1] + ["POOLED_interactive|-"] + order[-1:] if any(k.startswith("moa") for k in order) \
            else order + ["POOLED_interactive|-"]
    L = []
    L.append(f"Source: `{B['source']}`; tasks excluded (gold_failed): {B['exclusions']['gold_failed_units']} "
             f"({B['exclusions']['runs_dropped_by_unit_exclusion']} runs). Bootstrap {B['boot_draws']} task "
             f"resamples, seed {B['seed']}. pp; bold = 95% CI excludes 0.\n")

    L.append("### T1. Reference level, main effects and two-way interactions (identity = exec_filt)\n")
    L.append("| cell | n | ref (strict,string,last,FULL) | E lenient−strict | I exec−string | R last−r0 | A FULL−NOPEER |")
    L.append("|---|---|---|---|---|---|---|")
    for k in order:
        p = prim[k]
        e = p["effects"]
        L.append(f"| {nm(k)} | {p['n_units']} | {p['ledger']['reference_level']:.2f} | {f(e['E'])} | {f(e['I'])} | {f(e['R'])} | {f(e['A'])} |")
    L.append("\n| cell | E×I | E×R | E×A | I×R | I×A | R×A (=Δpeer avg) |")
    L.append("|---|---|---|---|---|---|---|")
    for k in order:
        e = prim[k]["effects"]
        L.append(f"| {nm(k)} | {f(e['ExI'])} | {f(e['ExR'])} | {f(e['ExA'])} | {f(e['IxR'])} | {f(e['IxA'])} | {f(e['RxA'])} |")
    L.append("\n| cell | E×I×R | E×I×A | E×R×A | I×R×A | E×I×R×A |")
    L.append("|---|---|---|---|---|---|")
    for k in order:
        e = prim[k]["effects"]
        L.append(f"| {nm(k)} | {f(e['ExIxR'])} | {f(e['ExIxA'])} | {f(e['ExRxA'])} | {f(e['IxRxA'])} | {f(e['ExIxRxA'])} |")

    L.append("\n### T2. Δpeer = (last−r0)_FULL − (last−r0)_NOPEER per (extraction, identity); repeats averaged\n")
    keys = ["strict|string", "strict|exec", "lenient|string", "lenient|exec"]
    L.append("| cell | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for k in order:
        d = prim[k]["delta_peer"]
        L.append(f"| {nm(k)} | " + " | ".join(f(d[x]) for x in keys) + " |")

    L.append("\n### T2b. Δpeer per independent repeat (r1 FULL vs r1 NOPEER; r2 vs r2)\n")
    L.append("| cell | repeat | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for k in order:
        if k.startswith("POOLED"):
            continue
        for rep, v in cells[k]["by_repeat"].items():
            L.append(f"| {nm(k)} | {rep} | " + " | ".join(f(v["delta_peer"][x]) for x in keys) + " |")

    L.append("\n### T2c. last−r0 within each arm (lenient, exec)\n")
    L.append("| cell | last−r0 FULL | last−r0 NOPEER | Δpeer |")
    L.append("|---|---|---|---|")
    for k in order:
        d = prim[k]["delta_peer"]["lenient|exec"]
        L.append(f"| {nm(k)} | {f(d['last_minus_r0_FULL'])} | {f(d['last_minus_r0_NOPEER'])} | {f(d)} |")

    L.append("\n### T3. Identity and extraction simple effects at r0 and at last (FULL arm)\n")
    L.append("| cell | I @strict,r0 | I @strict,last | I @lenient,r0 | I @lenient,last | E @string,r0 | E @string,last | E @exec,r0 | E @exec,last |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in order:
        s = prim[k]["simple_effects"]
        g = lambda t: f(s[t])
        L.append(f"| {nm(k)} | {g('I@E=strict|R=r0|A=FULL')} | {g('I@E=strict|R=last|A=FULL')} | "
                 f"{g('I@E=lenient|R=r0|A=FULL')} | {g('I@E=lenient|R=last|A=FULL')} | "
                 f"{g('E@I=string|R=r0|A=FULL')} | {g('E@I=string|R=last|A=FULL')} | "
                 f"{g('E@I=exec|R=r0|A=FULL')} | {g('E@I=exec|R=last|A=FULL')} |")

    L.append("\n### T4. Shared-reference ledger: reference = (strict, string, last, FULL)\n")
    L.append("Each row switches the listed factors away from the reference: E→lenient, I→exec, R→r0, A→NOPEER. "
             "`joint` is the observed change; `sum` adds the single switches; `non-add.` = joint − sum.\n")
    show = ["E", "I", "R", "A", "E+I", "R+A", "I+R", "E+I+R", "E+I+R+A"]
    for k in order:
        led = prim[k]["ledger"]
        L.append(f"**{nm(k)}** — reference level {led['reference_level']:.2f}\n")
        L.append("| switched | cell | level | joint Δ | sum of singles | non-additivity |")
        L.append("|---|---|---|---|---|---|")
        for r in led["rows"]:
            if r["switched"] in show:
                L.append(f"| {r['switched']} | {r['cell']} | {r['level']:.2f} | {f(r['joint'])} | "
                         f"{r['sum_of_singles']:+.2f} | {f(r['nonadditivity'])} |")
        fr = prim[k].get("final_reference")
        if fr:
            L.append(f"| *reference row: released `final`* | FULL | {fr['level_final_FULL']:.2f} | "
                     f"{f(fr['final_minus_refcell_FULL'])} | | |")
        L.append("")

    L.append("### T5. Released `final` (reference row only)\n")
    L.append("| cell | final FULL | final NOPEER | final − (lenient,exec,r0,FULL) | final − (lenient,exec,last,FULL) |")
    L.append("|---|---|---|---|---|")
    for k in order:
        fr = prim[k].get("final_reference")
        if fr:
            L.append(f"| {nm(k)} | {fr['level_final_FULL']:.2f} | {fr['level_final_NOPEER']:.2f} | "
                     f"{f(fr['final_minus_lenient_exec_r0_FULL'])} | {f(fr['final_minus_lenient_exec_last_FULL'])} |")

    L.append("\n### T6. Tie shares (share of runs whose winning label was tied), FULL arm\n")
    tk = [f"{e}|{i}|{r}|FULL" for r in ("r0", "last") for e in ("strict", "lenient") for i in ("string", "exec")]
    L.append("| cell | " + " | ".join(t.replace("|FULL", "") for t in tk) + " |")
    L.append("|---|" + "---|" * len(tk))
    for k in order:
        if k.startswith("POOLED"):
            continue
        t = cells[k]["ties"]
        L.append(f"| {nm(k)} | " + " | ".join(f"{t[x]['tie_share']:.2f}" for x in tk) + " |")
    L.append("\nShare of runs with no usable label (counted as wrong), FULL arm:\n")
    L.append("| cell | " + " | ".join(t.replace("|FULL", "") for t in tk) + " |")
    L.append("|---|" + "---|" * len(tk))
    for k in order:
        if k.startswith("POOLED"):
            continue
        t = cells[k]["ties"]
        L.append(f"| {nm(k)} | " + " | ".join(f"{t[x]['no_live_label_share']:.2f}" for x in tk) + " |")

    L.append("\n### T7. Completeness and early stopping\n")
    L.append("| cell | tasks | tasks with all 16 cells | both repeats × both arms | runs last=r0 (FULL/NOPEER) | tasks with any early stop |")
    L.append("|---|---|---|---|---|---|")
    for k in order:
        if k.startswith("POOLED"):
            continue
        c, es = cells[k]["completeness"], cells[k]["early_stop"]
        L.append(f"| {nm(k)} | {c['tasks_with_runs']} | {c['tasks_all_16_cells']} | {c['tasks_both_repeats_both_arms']} | "
                 f"{es['runs_last_eq_r0']} ({es['by_arm']['FULL']}/{es['by_arm']['NOPEER']}) | {es['tasks_with_any_early_stop']} |")
    sens = [(k, cells[k]["early_stop"]["sensitivity_excluding_tasks_with_any_early_stop"]) for k in order
            if not k.startswith("POOLED") and "sensitivity_excluding_tasks_with_any_early_stop" in cells[k]["early_stop"]]
    if sens:
        L.append("\nSensitivity: tasks with any early stop removed (descriptive; selection on a post-treatment event)\n")
        L.append("| cell | n | E×I | R×A | Δpeer lenient,exec | Δpeer strict,string |")
        L.append("|---|---|---|---|---|---|")
        for k, s in sens:
            L.append(f"| {nm(k)} | {s['n_units']} | {f(s['effects']['ExI'])} | {f(s['effects']['RxA'])} | "
                     f"{f(s['delta_peer']['lenient|exec'])} | {f(s['delta_peer']['strict|string'])} |")

    L.append("\n### T8. Secondary identity = exec_ident (failures one cluster): key terms\n")
    L.append("| cell | I | E×I | I×R | R×A | Δpeer lenient,exec_ident |")
    L.append("|---|---|---|---|---|---|")
    for k in order:
        if k.startswith("POOLED"):
            continue
        s = cells[k]["secondary_exec_ident"]
        L.append(f"| {nm(k)} | {f(s['effects']['I'])} | {f(s['effects']['ExI'])} | {f(s['effects']['IxR'])} | "
                 f"{f(s['effects']['RxA'])} | {f(s['delta_peer']['lenient|exec'])} |")
    L.append("\n### T9. Tie-break sensitivity (first = primary; last; seeded random)\n")
    L.append("| cell | tie | ref level | I | E×I | I×R | R×A | Δpeer strict,string | Δpeer lenient,exec | non-add. E+I+R | final − (lenient,exec,r0,FULL) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        src = prim[k] if k.startswith("POOLED") else cells[k]["primary"]
        ts = prim[k].get("tie_sensitivity", {}) if k.startswith("POOLED") else cells[k]["tie_sensitivity"]
        variants = [("first", src)] + list(ts.items())
        for tie, v in variants:
            e = v["effects"]
            na = next(r for r in v["ledger"]["rows"] if r["switched"] == "E+I+R")["nonadditivity"]
            fr = (v.get("final_reference") or {}).get("final_minus_lenient_exec_r0_FULL")
            L.append(f"| {nm(k)} | {tie} | {v['ledger']['reference_level']:.2f} | {f(e['I'])} | {f(e['ExI'])} | {f(e['IxR'])} | "
                     f"{f(e['RxA'])} | {f(v['delta_peer']['strict|string'])} | {f(v['delta_peer']['lenient|exec'])} | {f(na)} | {f(fr)} |")

    x = B.get("verdict_inconsistency_recheck")
    if x:
        L.append("\n### T10. Execution-class inconsistencies re-checked on this host\n")
        L.append(f"{x['suspects_total']} suspect queries (audit list + signature/verdict class disagreement); "
                 f"{x['suspects_in_candidate_sets']} occur in a factorial candidate set. Largest change in any cell level: "
                 f"{x['max_abs_level_change_pp']} pp.\n")
        L.append("| task | source | cached sig / verdict | this host sig / verdict |")
        L.append("|---|---|---|---|")
        for sx in x["suspects"]:
            L.append(f"| {sx['unit']} | {sx['source']} | {sx['stored_or_cached']['sig_status']} / {sx['stored_or_cached']['verdict']} | "
                     f"{sx['this_host']['sig_status']} / {sx['this_host']['verdict']} |")
        if x["affected_cell_levels"]:
            L.append("\nAffected cell levels (pp):\n")
            L.append("| cell | arm | ext | ident | round | tie | change |")
            L.append("|---|---|---|---|---|---|---|")
            for r in x["affected_cell_levels"]:
                L.append(f"| {r['cell']} | {r['arm']} | {r['ext']} | {r['ident']} | {r['round']} | {r['tie']} | {r['level_change_pp']:+.3f} |")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    print((out / "tables.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
