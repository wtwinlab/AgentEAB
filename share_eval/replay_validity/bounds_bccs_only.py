r"""BCCS-only bounds from bccs_divergence_runs.jsonl alone (for debates without replay scores, e.g. the
Qwen2.5-14B BCCS debates). Same estimands and bootstrap as bounds.py; never pooled with Llama-3-8B or
Qwen2.5-7B.

    python3 -I share_eval/replay_validity/bounds_bccs_only.py --div output/share_eval/replay_validity_backbones/bccs_divergence_runs.jsonl \
            --out output/share_eval/replay_validity_backbones/bounds_bccs.json
"""
import argparse, collections, json, pathlib, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bounds import summarise, BOOT, SEED  # noqa: E402

TERMS = {
    "Dpeer (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL"), (-1, "EI", "last", "NOPEER"), (1, "EI", "r0", "NOPEER")],
    "FULL last - r0 (lenient, exec)": [(1, "EI", "last", "FULL"), (-1, "EI", "r0", "FULL")],
    "E+I vs ref (FULL, last)": [(1, "EI", "last", "FULL"), (-1, "ref", "last", "FULL")],
    "Dpeer (strict, string), first tie": [(1, "ref", "last", "FULL"), (-1, "ref", "r0", "FULL"), (-1, "ref", "last", "NOPEER"), (1, "ref", "r0", "NOPEER")],
}


def iv(x, mp, rnd):
    d = x[mp]
    return (d["R_r0"],) * 3 if rnd == "r0" else (d["O_lo"], d["O_hi"], d["R_last"])


def main():
    ap = argparse.ArgumentParser(description="BCCS-only bounds of Lemma 1")
    ap.add_argument("--div", required=True, help="bccs_divergence_runs.jsonl"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [json.loads(l) for l in pathlib.Path(a.div).read_text().splitlines()]
    res = {}
    for bb in sorted({r["backbone"] for r in runs}):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in runs:
            if r["backbone"] == bb:
                by[r["unit"]][r["arm"]].append(r)
        units = sorted(u for u, v in by.items() if {"FULL", "NOPEER"} <= set(v))
        idx = np.random.default_rng(SEED).integers(0, len(units), size=(BOOT, len(units)))
        res[bb] = {"n_tasks": len(units)}
        for name, terms in TERMS.items():
            E, L, H = (np.zeros(len(units)) for _ in range(3))
            for k, u in enumerate(units):
                for sg, mp, rnd, arm in terms:
                    vals = [iv(x, mp, rnd) for x in by[u][arm]]
                    l_, h_, R_ = (np.mean([v[j] for v in vals]) for j in range(3))
                    E[k] += sg * R_
                    L[k] += l_ if sg > 0 else -h_
                    H[k] += h_ if sg > 0 else -l_
            res[bb][name] = summarise(E, L, H, idx)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
