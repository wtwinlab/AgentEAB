"""Draw Fig. fig:readout-cells from output/plotted_values.csv (written by compute.py).

    /path/to/python-with-matplotlib code/plot.py [--out output]

Two panels: the protocol extractor with exact-match voting (left) and the lenient extractor with
execution voting (right); each shows the six protocol-backbone pairs and their mean under three
tie rules. Writes readout_cells.pdf and readout_cells.png into the output directory.
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(HERE.parent / 'output'))
OUT = pathlib.Path(ap.parse_args().out)
pts = {(r['readout'], r['cell'], r['tie']): (float(r['est']), float(r['lo']), float(r['hi']))
       for r in csv.DictReader(open(OUT / 'plotted_values.csv'))}
cells = [('du_mad|qwen2.5-7b', 'Du-MAD / Q7'), ('du_mad|llama3-8b', 'Du-MAD / L8'),
         ('madm2_objective|qwen2.5-7b', 'MAD-M$^2$ / Q7'), ('madm2_objective|llama3-8b', 'MAD-M$^2$ / L8'),
         ('bccs|qwen2.5-7b', 'BCCS / Q7'), ('bccs|llama3-8b', 'BCCS / L8'), ('mean', 'Mean of six')]
panels = [('strict|string', 'Protocol extractor\nExact-match vote'),
          ('lenient|exec', 'Lenient extractor\nExecution vote')]
ties = [('first', 'First agent', 'o', '#2a78d6'), ('last', 'Last agent', 's', '#eb6834'),
        ('random', 'Random', '^', '#1baf7a')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.7, 'axes.titlesize': 7.0, 'pdf.fonttype': 42})
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.6), sharex=True, sharey=True)
for ax, (ro, title) in zip(axs.flat, panels):
    ax.axvline(0, color='#777777', lw=.6, ls='--', zorder=0)
    ax.axhline(.5, color='#bbbbbb', lw=.5, zorder=0)
    for i, (cell, lab) in enumerate(cells):
        for j, (tie, tlab, marker, color) in enumerate(ties):
            e, lo, hi = pts[ro, cell, tie]
            ax.errorbar(e, 6 - i + (1 - j) * .24, xerr=[[e - lo], [hi - e]], fmt=marker, color=color,
                        ms=2.6, lw=.8, capsize=1.2, zorder=3, mfc=color if tie == 'first' else 'white')
    ax.set_title(title, pad=4, linespacing=1.1)
    ax.set_xlim(-11, 15)
    ax.set_xticks([-10, 0, 10])
    ax.set_ylim(-.6, 6.6)
    ax.grid(axis='x', color='#e8e8e8', lw=.5)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', length=2, pad=2)
axs[0].set_yticks(list(reversed(range(7))), [lab for _, lab in cells])
handles = [plt.Line2D([], [], marker=m, color=c, lw=.8, markersize=3, mfc=c if t == 'first' else 'white', label=l)
           for t, l, m, c in ties]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.58, 1.01), ncol=3, frameon=False, fontsize=6.7,
           columnspacing=1, handlelength=1.2)
fig.supxlabel('Peer interaction (pp; 95% CI over questions)', fontsize=7, y=.015)
fig.subplots_adjust(left=.265, right=.99, bottom=.19, top=.73, wspace=.16)
fig.savefig(OUT / 'readout_cells.pdf')
fig.savefig(OUT / 'readout_cells.png', dpi=240)
print('wrote', OUT / 'readout_cells.pdf')
