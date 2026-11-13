r"""Figures, with the numbers behind each panel written out beside them.

    python3.13 experiments/05_matching_downstream/code/plot.py

Three figure groups, deliberately different shapes rather than three variants of one chart:

  three_stages      a slope chart across A, B, C per method, plus a diverging decomposition of the
                    scoring-universe change against the merge-operator effect
  topology          asserted links against implied pairs on log axes, the component-size tail, and
                    precision/recall movement from A to C as arrows
  local_downstream  every single-edge deletion as a point in (t/b, ΔF1) with the F/2 threshold drawn,
                    deletion-outcome counts per method, and the sharing access effect at stage A
                    against the same contrast at stage C as a dumbbell

Each panel's series is also written to `figdata/*.csv`, and `figdata/figure_sources.json` records the
sha256 of every analysis file the figures were built from.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
A = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
FD = os.path.join(BASE, "figdata")

SHORT = {"single:qwen2.5-7b": "Q7", "single:llama-3-8b": "L8", "single:mistral-7b": "M7",
         "single:qwen2.5-14b": "Q14", "single:qwen2.5-32b": "Q32", "single:qwen2.5-72b": "Q72",
         "vote:majority": "vote", "GOLD_edge_replay": "gold"}
ORDER = ["single:qwen2.5-7b", "single:llama-3-8b", "single:mistral-7b", "single:qwen2.5-14b",
         "single:qwen2.5-32b", "single:qwen2.5-72b", "vote:majority", "GOLD_edge_replay"]
C_OK, C_BAD, C_NEU, C_GOLD = "#1f6fb4", "#c0392b", "#7f8c8d", "#b7950b"


def J(n: str) -> Dict[str, Any]:
    with open(os.path.join(A, n), encoding="utf-8") as f:
        return json.load(f)


def wcsv(name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(FD, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def decollide(ax, items, min_gap_frac: float = 0.045) -> List[float]:
    """Nudge annotation y-positions apart so labels stay readable. Data is never moved."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = sorted(range(len(items)), key=lambda i: items[i])
    ys = list(items)
    for a, b in zip(order, order[1:]):
        if ys[b] - ys[a] < gap:
            ys[b] = ys[a] + gap
    return ys


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


def fig_three_stages(st: Dict[str, Any]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    fig.subplots_adjust(wspace=0.32)
    rows, ends, cols = [], [], []
    for m in ORDER:
        s = st["methods"][m]
        ys = [s["A"]["f1"], s["B"]["f1"], s["C"]["f1"]]
        col = C_GOLD if m == "GOLD_edge_replay" else (C_BAD if ys[2] < 0.5 else C_OK)
        ax1.plot([0, 1, 2], ys, marker="o", color=col, lw=1.8, ms=5,
                 ls="--" if m == "GOLD_edge_replay" else "-", zorder=3)
        ends.append(ys[2])
        cols.append(col)
        rows.append({"method": m, "A_f1": ys[0], "B_f1": ys[1], "C_f1": ys[2],
                     "rank_A": st["ranks"]["A"].get(m), "rank_B": st["ranks"]["B"].get(m),
                     "rank_C": st["ranks"]["C"].get(m)})
    ax1.set_ylim(-0.03, 1.05)
    for m, y0, y1, col in zip(ORDER, ends, decollide(ax1, ends), cols):
        ax1.annotate(SHORT[m], (2.06, y1), fontsize=9, color=col, va="center",
                     annotation_clip=False)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["A\ncandidate pairs", "B\nall pairs,\ndirect links",
                         "C\nall pairs,\nafter closure"], fontsize=9)
    ax1.set_ylabel("F1")
    ax1.set_xlim(-0.12, 2.42)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("(a) one universe change, one operator", fontsize=10, loc="left")
    ax1.legend(handles=[Line2D([], [], color=C_OK, marker="o", label="closure survivable"),
                        Line2D([], [], color=C_BAD, marker="o", label="closure collapses it"),
                        Line2D([], [], color=C_GOLD, marker="o", ls="--",
                               label="gold-edge replay")],
               fontsize=8, loc="lower left", framealpha=0.9)

    ms = [m for m in ORDER if m != "GOLD_edge_replay"]
    y = range(len(ms))
    ub = [st["methods"][m]["universe_change_A_to_B_f1"] for m in ms]
    op = [st["methods"][m]["operator_effect_B_to_C_f1"] for m in ms]
    ax2.barh([i + 0.2 for i in y], ub, height=0.38, color=C_NEU, label="A→B scoring universe")
    ax2.barh([i - 0.2 for i in y], op, height=0.38,
             color=[C_OK if v >= 0 else C_BAD for v in op], label="B→C merge operator")
    for i, v in zip(y, op):
        ax2.annotate(f"{v:+.3f}", (v, i - 0.2), fontsize=7.5, va="center",
                     ha="right" if v < 0 else "left",
                     xytext=(-4 if v < 0 else 4, 0), textcoords="offset points")
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xscale("symlog", linthresh=0.01)
    ax2.set_xticks([-0.8, -0.1, -0.01, 0, 0.01, 0.1])
    ax2.set_xticklabels(["-0.8", "-0.1", "-0.01", "0", "+0.01", "+0.1"], fontsize=8)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([SHORT[m] for m in ms])
    ax2.set_xlabel("change in F1 (symlog, linear within ±0.01)")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_title("(b) the operator does the work on this workload", fontsize=10, loc="left")
    ax2.legend(fontsize=8, loc="lower left")
    for m, u, o in zip(ms, ub, op):
        for r in rows:
            if r["method"] == m:
                r["universe_change_A_to_B"] = u
                r["operator_effect_B_to_C"] = o
    wcsv("three_stages.csv", rows)
    save(fig, "three_stages")


def fig_topology(st: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))
    fig.subplots_adjust(wspace=0.30)
    ax1, ax2, ax3 = axes
    rows, xs, ys_, cs, labs = [], [], [], [], []
    for m in ORDER:
        t = st["methods"][m]["topology"]
        col = C_GOLD if m == "GOLD_edge_replay" else (C_BAD if t["closure_amplification"] > 2
                                                      else C_OK)
        ax1.scatter(t["asserted_links"], t["implied_pairs"], s=70, color=col, zorder=3,
                    marker="D" if m == "GOLD_edge_replay" else "o")
        xs.append(t["asserted_links"])
        ys_.append(t["implied_pairs"])
        cs.append(col)
        labs.append(f"{SHORT[m]} ({t['closure_amplification']:.1f}×)")
        rows.append({"method": m, **{k: v for k, v in t.items()
                                     if k != "component_size_distribution"}})
    lo = min(st["methods"][m]["topology"]["asserted_links"] for m in ORDER)
    hi = max(st["methods"][m]["topology"]["asserted_links"] for m in ORDER)
    ax1.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls=":")
    ax1.set_yscale("log")
    ax1.set_xlim(1450, 2700)
    # a legend rather than per-point labels: six of the eight points sit on top of each other,
    # so any annotation ladder would be unreadable or would imply positions the data does not have
    ax1.legend(handles=[Line2D([], [], color=c, marker="D" if m == "GOLD_edge_replay" else "o",
                               ls="", label=lb)
                        for m, c, lb in zip(ORDER, cs, labs)]
                       + [Line2D([], [], color="k", ls=":", label="no amplification (1x)")],
               fontsize=8, loc="upper left", framealpha=0.95, ncol=2, handletextpad=0.4,
               columnspacing=0.9)
    ax1.set_xlabel("asserted links")
    ax1.set_ylabel("implied pairs after closure")
    ax1.grid(alpha=0.25, which="both")
    ax1.set_ylim(1.5e3, 2.2e5)
    ax1.set_title("(a) what a link commits you to", fontsize=10, loc="left")

    for m in ORDER:
        d = st["methods"][m]["topology"]["component_size_distribution"]
        sizes = sorted((int(k) for k in d), reverse=True)
        cum, xs, ys = 0, [], []
        for s in sizes:
            cum += d[str(s)]
            xs.append(s)
            ys.append(cum)
        col = C_GOLD if m == "GOLD_edge_replay" else (
            C_BAD if st["methods"][m]["topology"]["closure_amplification"] > 2 else C_OK)
        ax2.step(xs, ys, where="post", color=col, lw=1.6,
                 ls="--" if m == "GOLD_edge_replay" else "-")
        ax2.annotate(SHORT[m], (xs[0], ys[0]), textcoords="offset points",
                     xytext=(5, -4 - 7 * (ORDER.index(m) % 3)), fontsize=8, color=col)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("component size (descending)")
    ax2.set_ylabel("components at least this large")
    ax2.grid(alpha=0.25, which="both")
    ax2.set_title("(b) the component-size tail", fontsize=10, loc="left")

    pr3: List[Any] = []
    for m in ORDER:
        s = st["methods"][m]
        col = C_GOLD if m == "GOLD_edge_replay" else (C_BAD if s["C"]["f1"] < 0.5 else C_OK)
        ax3.annotate("", xy=(s["C"]["recall"], s["C"]["precision"]),
                     xytext=(s["A"]["recall"], s["A"]["precision"]),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=0.9))
        ax3.scatter([s["A"]["recall"]], [s["A"]["precision"]], s=26, color=col, zorder=3)
        pr3.append((s["C"]["recall"], s["C"]["precision"], SHORT[m], col))
    ax3.set_xlabel("recall")
    ax3.set_ylabel("precision")
    ax3.set_xlim(-0.03, 1.22)
    ax3.set_ylim(-0.03, 1.08)
    for (rx, ry, lb, col), yy in zip(pr3, decollide(ax3, [p[1] for p in pr3], 0.055)):
        ax3.annotate(lb, (rx + 0.02, yy), fontsize=8.5, color=col, va="center")
    ax3.grid(alpha=0.25)
    ax3.set_title("(c) stage A → stage C, arrow per method", fontsize=10, loc="left")
    wcsv("topology.csv", rows)
    save(fig, "topology")


def fig_local_downstream(cf: Dict[str, Any], sh: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))
    fig.subplots_adjust(wspace=0.34)
    ax1, ax2, ax3 = axes
    rows = list(csv.DictReader(open(os.path.join(A, "edge_deletion.csv"))))
    br = [r for r in rows if r["is_bridge"] == "True" and r["t_over_b"]]
    xs_c = [float(r["t_over_b"]) for r in br if r["edge_is_correct"] == "True"]
    ys_c = [float(r["delta_f1"]) for r in br if r["edge_is_correct"] == "True"]
    xs_w = [float(r["t_over_b"]) for r in br if r["edge_is_correct"] == "False"]
    ys_w = [float(r["delta_f1"]) for r in br if r["edge_is_correct"] == "False"]
    ax1.scatter(xs_w, ys_w, s=16, color=C_BAD, alpha=0.65, label="deleted edge is incorrect")
    ax1.scatter(xs_c, ys_c, s=16, color=C_OK, alpha=0.65, label="deleted edge is correct")
    f2 = [float(r["F_over_2"]) for r in br]
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axvspan(min(f2), max(f2), color="#f1c40f", alpha=0.18,
                label=f"F/2 range [{min(f2):.2f}, {max(f2):.2f}]")
    ax1.set_xlabel("t / b, true share of the pairs the deletion separates")
    ax1.set_ylabel("change in cluster F1")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax1.set_title("(a) every bridge deletion, against the t/b > F/2 condition",
                  fontsize=10, loc="left")

    ms = [m for m in ORDER if m in cf["methods"]]
    keys = [("incorrect_improves", "incorrect edge, deleting helps", C_OK),
            ("incorrect_harms", "incorrect edge, deleting hurts", C_BAD),
            ("correct_improves", "correct edge, deleting helps", "#8e44ad")]
    w = 0.26
    for i, (k, lab, col) in enumerate(keys):
        vals = [cf["methods"][m]["deletion_counts"].get(k, 0) for m in ms]
        ax2.bar([j + (i - 1) * w for j in range(len(ms))], vals, width=w, color=col, label=lab)
        for j, v in enumerate(vals):
            if v:
                ax2.annotate(str(v), (j + (i - 1) * w, v), ha="center", fontsize=7.5,
                             xytext=(0, 2), textcoords="offset points")
    ax2.set_xticks(range(len(ms)))
    ax2.set_xticklabels([SHORT[m] for m in ms], fontsize=9)
    ax2.set_ylabel("edges")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.set_title("(b) deletion outcomes; gains are not additive", fontsize=10, loc="left")

    lab, ya, yc = [], [], []
    for rec, byc in sh["access_contrasts"].items():
        for c, v in byc.items():
            lab.append(f"Qwen {rec.split('-')[-1].upper()}\n{c.split(' - ')[0]}")
            ya.append(v["diff"])
            yc.append(sh["stageC"][rec][c]["delta_f1_stageC"])
    yy = range(len(lab))
    for i, (a_, c_) in enumerate(zip(ya, yc)):
        ax3.plot([a_, c_], [i, i], color=C_NEU, lw=1.6, zorder=1)
    ax3.scatter(ya, list(yy), s=62, color=C_OK, zorder=3, label="stage A (candidate pairs)")
    ax3.scatter(yc, list(yy), s=62, color=C_BAD, zorder=3, marker="s",
                label="stage C (after closure)")
    ax3.axvline(0, color="k", lw=0.8)
    ax3.set_yticks(list(yy))
    ax3.set_yticklabels(lab, fontsize=8.5)
    ax3.set_xlabel("effect of peer access on F1, against no_peers")
    ax3.grid(axis="x", alpha=0.25)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.set_title("(c) a local gain is not a downstream gain", fontsize=10, loc="left")

    wcsv("local_downstream.csv",
         [{"method": m, **cf["methods"][m]["deletion_counts"],
           "edges": cf["methods"][m]["edges"], "bridges": cf["methods"][m]["bridges"],
           "f1": cf["methods"][m]["f1"], "F_over_2": cf["methods"][m]["F_over_2"]}
          for m in ms]
         + [{"method": "ACCESS_" + l.replace("\n", "_"), "stage_A_delta": a_,
             "stage_C_delta": c_} for l, a_, c_ in zip(lab, ya, yc)])
    save(fig, "local_downstream")


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(FD, exist_ok=True)
    st, cf = J("stages.json"), J("counterfactual.json")
    sh = J("sharing.json")
    fig_three_stages(st)
    fig_topology(st)
    fig_local_downstream(cf, sh)
    srcs = {}
    for n in ("stages.json", "counterfactual.json", "sharing.json", "edge_deletion.csv",
              "stages_long.csv", "sharing_long.csv", "blocking.json", "recheck.json"):
        p = os.path.join(A, n)
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for c in iter(lambda: f.read(1 << 20), b""):
                    h.update(c)
            srcs[n] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
    with open(os.path.join(FD, "figure_sources.json"), "w", encoding="utf-8") as f:
        json.dump({"figures": ["three_stages", "topology", "local_downstream"],
                   "plotting_data": sorted(os.listdir(FD)),
                   "analysis_sources": srcs,
                   "note": ("each panel's series is in the matching csv; stage B and C carry no "
                            "interval by design, so no error bars are drawn for them")},
                  f, indent=2, sort_keys=True)
    print(f"  wrote figdata/figure_sources.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
