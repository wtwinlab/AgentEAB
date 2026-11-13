"""Draw Fig. merge-validation from output/plot_data.json (written by compute.py).

    python3 code/plot.py [--data output] [--out output]
    # -> merge_validation.pdf, merge_validation.png

(a,b): F1 on the supplied candidate pairs, on direct links over all record pairs, and on the
relations implied by the final groups, for each model alone and the six-model vote.
(c): change from no peers when Qwen2.5-7B or Qwen2.5-32B receives the peers' decisions (D) or full
responses (F), in pair F1 and in cluster F1 under connected components (Settlements). F1 in (a,b) is on
a 0-1 scale; (c) is in pp.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE.parent / "output"))
ap.add_argument("--out", default=str(HERE.parent / "output"))
a = ap.parse_args()
d = json.loads((Path(a.data) / "plot_data.json").read_text())
O = Path(a.out)
O.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
f = plt.figure(figsize=(4.6, 136 / 72))
axs = [f.add_axes(x) for x in [[.065, .35, .235, .55], [.375, .35, .235, .55], [.79, .35, .195, .55]]]
names = ["Q7", "L8", "M7", "Q14", "Q32", "Q72", "Vote"]
for ax, key, title in zip(axs[:2], ("wdc", "settlements"), ("(a) WDC", "(b) Settlements")):
    v = d[key]
    ax.imshow(v, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for y, row in enumerate(v):
        for x, z in enumerate(row):
            ax.text(x, y, f"{z:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if z > .55 else "#202020")
    ax.axvline(.5, color="white", lw=2)
    ax.set_xticks(range(3), ["Pairs", "Direct\nlinks", "Groups"])
    ax.set_yticks(range(7), names)
    ax.tick_params(length=0, labelsize=7, pad=2)
    ax.set_title(title, fontsize=8, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
ax = axs[2]
for i, x in enumerate(d["sharing"]):
    p, c = x["pair_pp"], x["cluster_pp"]
    ax.plot([p, c], [i, i], color="#969CA5", lw=1.2)
    ax.scatter(p, i, s=19, color="#2F4668", label="Pair F1" if i == 0 else None, zorder=3)
    ax.scatter(c, i, s=23, marker="D", color="#BB6B32", label="Cluster F1" if i == 0 else None, zorder=3)
ax.set_yticks(range(4), ["Q7 / D", "Q7 / F", "Q32 / D", "Q32 / F"])
ax.set_ylim(3.5, -.55)
ax.set_xlim(-4.3, 10.5)
ax.axvline(0, color="#B2B7BF", lw=.8)
ax.grid(axis="x", alpha=.12)
ax.set_xticks([0, 5, 10])
ax.tick_params(labelsize=7, pad=2)
ax.set_xlabel("F1 change (pp)", fontsize=7, labelpad=2)
ax.set_title("(c) Sharing", fontsize=8, pad=4)
f.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(.69, .075), ncol=2, fontsize=7,
         frameon=False, handlelength=.7, handletextpad=.3, columnspacing=.6, borderpad=0, markerscale=.8)
f.text(.012, .105, "D: peer decisions; F: full responses.  (a,b): F1 on a 0\u20131 scale.", fontsize=6.8)
f.text(.012, .018, "L8: Llama-3-8B; M7: Mistral-7B; Qn: Qwen2.5-nB.", fontsize=7)
f.savefig(O / "merge_validation.pdf")
f.savefig(O / "merge_validation.png", dpi=240)
