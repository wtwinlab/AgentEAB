"""Draw Fig. stage-act from output/plotted_values.csv (written by compute.py).

    /path/to/python-with-matplotlib code/plot.py      # -> output/stage_act.pdf, output/stage_act.png

Left: share of all tasks on which the stage changes the result, split into gains, losses and both wrong,
with the stage's total effect (pp) at the bar end. Right: the stage's share of its pipeline's LLM calls and tokens.
"""
import csv
import pathlib
from decimal import ROUND_HALF_UP, Decimal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
rows = [{k: (v if k == "label" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(OUT / "plotted_values.csv"))]
for r in rows:
    assert abs(r["gain"] + r["loss"] + r["both"] - r["pa"]) < 0.02

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 6.5, "pdf.fonttype": 42})
f, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.55), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
ys = list(reversed(range(len(rows))))
for y, r in zip(ys, rows):
    a1.barh(y, r["gain"], color="#2a78d6", height=0.62)
    a1.barh(y, r["loss"], left=r["gain"], color="#eb6834", hatch="////", edgecolor="white", lw=0, height=0.62)
    a1.barh(y, r["both"], left=r["gain"] + r["loss"], color="#c9c9c9", height=0.62)
    total = f"{Decimal(str(r['total'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+}".replace("-", "−")
    a1.text(r["pa"] + 0.5, y, total, va="center", ha="left", fontsize=5.8, color="#333333")
    a2.plot(r["calls"], y + 0.14, marker="o", color="#2a78d6", ms=3.2, ls="none")
    a2.plot(r["tokens"], y - 0.14, marker="D", color="#2a78d6", ms=2.6, mfc="white", ls="none")
a1.set_yticks(ys, [r["label"] for r in rows])
a1.set_xlim(0, 36); a1.set_xlabel("% of all tasks"); a1.set_title("Stage changes the result", pad=11)
a2.set_xlim(-8, 108); a2.set_xticks([0, 50, 100]); a2.set_xlabel("% of calls, tokens"); a2.set_title("Cost share", pad=11)
for ax in (a1, a2):
    ax.grid(axis="x", color="#e8e8e8", lw=0.5); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa"); ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(rows) - 0.4)
h1 = [plt.Rectangle((0, 0), 1, 1, color="#2a78d6"),
      plt.Rectangle((0, 0), 1, 1, facecolor="#eb6834", hatch="////", edgecolor="white", lw=0),
      plt.Rectangle((0, 0), 1, 1, color="#c9c9c9")]
a1.legend(h1, ["gain", "loss", "both wrong"], loc="lower left", bbox_to_anchor=(-0.02, 0.995), ncol=3, frameon=False,
          fontsize=5.6, handlelength=0.8, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.0, borderpad=0.0)
h2 = [plt.Line2D([], [], marker="o", color="#2a78d6", ls="none", ms=3.2),
      plt.Line2D([], [], marker="D", color="#2a78d6", ls="none", ms=2.6, mfc="white")]
a2.legend(h2, ["calls", "tokens"], loc="upper right", frameon=False, fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
f.subplots_adjust(left=0.37, right=0.965, bottom=0.21, top=0.81, wspace=0.22)
f.savefig(OUT / "stage_act.pdf")
f.savefig(OUT / "stage_act.png", dpi=220)
