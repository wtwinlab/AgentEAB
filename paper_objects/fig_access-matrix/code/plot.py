"""Write Fig. fig:access-matrix as the LaTeX table the paper inputs (figures/access_matrix.tex),
from output/plotted_values.csv (written by compute.py).

    python3 code/plot.py [--out output]      # -> output/access_matrix.tex

Rows are benchmarks, columns the protocol configurations; each cell prints the sharing effect (pp,
one decimal) on a shaded box whose intensity is 4/3 of the printed magnitude (positive: colour
`asrun`, negative: colour `rulec`, both defined by the paper). Dagger: the sign depends on how
unreadable replies are counted; section sign: no answer is delivered in either condition; dash: no
estimate.
"""
import argparse
import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(HERE.parent / 'output'))
OUT = pathlib.Path(ap.parse_args().out)
BENCH = [('gsm8k', 'GSM8K'), ('strategyqa', 'StrategyQA'), ('mmlu_pro', 'MMLU-Pro'), ('gpqa', 'GPQA'),
         ('wdc_products', 'WDC'), ('geographic_settlements', 'Settlements'), ('sql_bird_v2', 'BIRD')]
COLS = [('ford', 'roster'), ('moa', 'roster'), ('du_mad', 'llama3-8b'), ('du_mad', 'qwen2.5-7b'),
        ('madm2_objective', 'llama3-8b'), ('madm2_objective', 'qwen2.5-7b'), ('bccs', 'llama3-8b'),
        ('bccs', 'qwen2.5-7b')]
pts = {(r['protocol'], r['backbone'], r['benchmark']): r for r in csv.DictReader(open(OUT / 'plotted_values.csv'))}


def cell(r):
    if r['with_minus_without_pp'] == '':
        return '---'
    t = f"{float(r['with_minus_without_pp']):.1f}"
    if t == '-0.0':
        t = '0.0'
    v = float(t)
    shade = f"{'rulec' if v < 0 else 'asrun'}!{round(abs(v) * 4 / 3)}"
    if r['no_answer_in_either_condition'] == 'True':
        t += '^{\\S}'
    elif r['unresolved_readings'] == 'True':
        t += '^{\\dagger}'
    return '\\effectcell{%s}{%s}' % (shade, t)


lines = ['\\begingroup\\scriptsize',
         '\\setlength{\\tabcolsep}{1pt}\\setlength{\\fboxsep}{0.3pt}\\renewcommand{\\arraystretch}{0.9}',
         '\\newcommand{\\effectcell}[2]{\\colorbox{#1}{\\makebox[18pt]{$#2$}}}',
         '\\begin{tabular}{@{}l*{8}{c}@{}}',
         '& FORD & MoA & \\multicolumn{2}{c}{Du-MAD} & \\multicolumn{2}{c}{\\mad{}} & \\multicolumn{2}{c}{BCCS}\\\\',
         '& mixed & mixed & L8 & Q7 & L8 & Q7 & L8 & Q7\\\\\\midrule']
for b, name in BENCH:
    lines.append(name + ' & ' + ' & '.join(cell(pts[(p, bb, b)]) for p, bb in COLS) + '\\\\')
lines += ['\\end{tabular}', '\\endgroup']
(OUT / 'access_matrix.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote', OUT / 'access_matrix.tex')
