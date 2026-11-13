#!/usr/bin/env python3
"""Draw Fig. fig:sql-family from output/plotted_values.csv (needs matplotlib).

    /path/to/python-with-matplotlib code/plot.py [--out DIR]     # writes DIR/sql_family.pdf

Generation minus voting with 95% intervals over databases, one panel per benchmark, one row per
generator, one marker per voting rule.
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE.parent / "output"))
out = pathlib.Path(ap.parse_args().out)
rows = list(csv.DictReader(open(HERE.parent / "output" / "plotted_values.csv", encoding="utf-8")))
models = [("qwen3-32b", "Qwen3"), ("gpt-oss-120b", "gpt-oss"), ("gemma-4-26b", "Gemma")]
rules = [("released", "Unfiltered", "o", "#2a78d6"), ("valid_majority", "Valid", "s", "#eb6834"),
         ("nonempty_majority", "Nonempty", "^", "#1baf7a")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 6.5, "pdf.fonttype": 42})
f, axs = plt.subplots(1, 2, figsize=(3.5, 1.3), sharey=True)
for ax, b in zip(axs, ["bird", "spider"]):
    ax.axvline(0, color="#777777", lw=0.6, ls="--", zorder=0)
    for i, (m, _name) in enumerate(models):
        y = len(models) - 1 - i
        for j, (rule, _rlab, mk, c) in enumerate(rules):
            r = next(r for r in rows if r["benchmark"] == b and r["recipient"] == m and r["rule"] == rule)
            e, lo, hi = (float(r[k]) for k in ("est", "lo", "hi"))
            ax.errorbar(e, y + (1 - j) * 0.23, xerr=[[e - lo], [hi - e]], fmt=mk, color=c, ms=2.8, lw=0.8,
                        capsize=1.2, zorder=3)
    ax.set_title("BIRD" if b == "bird" else "Spider", pad=3)
    ax.set_xlim(-6, 17)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.grid(axis="x", color="#e8e8e8", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(models) - 0.4)
axs[0].set_yticks(list(reversed(range(len(models)))), [n for _, n in models])
handles = [plt.Line2D([], [], marker=mk, color=c, lw=0.8, markersize=3, label=l) for _, l, mk, c in rules]
f.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.02), ncol=3, frameon=False, fontsize=6,
         columnspacing=1.0, handlelength=1.2)
f.supxlabel("Generation minus voting (pp, 95% CI over databases)", fontsize=6.5, y=0.02)
f.subplots_adjust(left=0.13, right=0.99, bottom=0.26, top=0.74, wspace=0.12)
out.mkdir(parents=True, exist_ok=True)
f.savefig(out / "sql_family.pdf")
