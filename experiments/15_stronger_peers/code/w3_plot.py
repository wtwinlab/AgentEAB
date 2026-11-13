r"""W3 gap display: recipient − peer own accuracy (x) against sharing effect (y).

    python3 code/w3_plot.py        # reads output/w3_gap_points.csv -> output/w3_gap.pdf, .png

Associational display only. Points are recipient x benchmark x source cells that share tasks and
models, so they are not independent samples; no fitted line or slope is drawn or reported.
"""
import csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = pathlib.Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(W / "output/w3_gap_points.csv")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
MARK = {"llama3-8b": "o", "qwen25-7b": "s", "qwen3-32b-new": "D"}
LAB = {"llama3-8b": "Llama-3-8B peers (old)", "qwen25-7b": "Qwen2.5-7B peers (old)",
       "qwen3-32b-new": "Qwen3-32B peers (new)"}
COL = {"qwen3-32b": "#2F4668", "gpt-oss-120b": "#BB6B32", "gemma-4-26b": "#4E8A5B"}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for ax, cond, title in zip(axs, ("full", "answer_only"), ("Full solutions", "Answers only")):
    for r in rows:
        if r["condition"] != cond or not r["gap_pp"]:
            continue
        ax.scatter(float(r["gap_pp"]), float(r["effect_pp"]), marker=MARK[r["source"]],
                   s=22, color=COL[r["recipient"]], alpha=.85,
                   edgecolor="white" if r["source"] != "qwen3-32b-new" else "black", linewidth=.5)
    ax.axhline(0, color=".6", lw=.7); ax.axvline(0, color=".85", lw=.7)
    ax.set_title(title, fontsize=8); ax.set_xlabel("Recipient − peer accuracy (pp)")
    ax.grid(alpha=.15)
axs[0].set_ylabel("Effect vs. no peers (pp)")
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="", marker=MARK[s], color=".35", label=LAB[s]) for s in MARK] + \
    [Line2D([], [], ls="", marker="o", color=c, label=m) for m, c in COL.items()]
fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6, frameon=False,
           bbox_to_anchor=(.5, -.02), handletextpad=.2, columnspacing=.8)
fig.subplots_adjust(bottom=.30, wspace=.08, left=.08, right=.99, top=.9)
fig.savefig(W / "output/w3_gap.pdf"); fig.savefig(W / "output/w3_gap.png", dpi=200)
print("wrote output/w3_gap.pdf, .png from", len(rows), "points")
