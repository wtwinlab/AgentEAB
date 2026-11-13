"""EM adapter: grouping rules on fixed judgments, scored by relation counts.

Rules and per-cell evaluation are the existing `eb1_grouping.evaluate` (imported from
experiments/09_grouping_transfer, unmodified); relation counts are
`metrics.cluster_scores`'s T (true intra pairs), N (predicted intra pairs), G (gold pairs).
Cluster F1 is formed from those counts, never as a mean over record rows.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys

import core as C

GT = C.REPO / "experiments/09_grouping_transfer"
sys.path.insert(0, str(GT / "code"))
import eb1_grouping as EB                                                   # noqa: E402

_CACHE = {}


def _evaluate():
    if "ev" not in _CACHE:
        insts = EB.rule_instances()
        keys = [EB.key_of(r, p) for r, p, _ in insts]
        ev = {}
        for ds in EB.DATASETS:
            for split in ("dev", "test"):
                entry = json.loads((GT / f"cache/{ds}_{split}.json").read_text(encoding="utf-8"))
                with contextlib.redirect_stdout(io.StringIO()):
                    ev[(ds, split)] = EB.evaluate(entry, insts, split)
        _CACHE["ev"], _CACHE["keys"] = ev, keys
    return _CACHE["ev"], _CACHE["keys"]


def load(spec):
    ev, keys = _evaluate()
    recs, units = [], set()
    for (ds, split), cells in ev.items():
        for cellkey, c in cells.items():
            unit = (ds, f"{c['recipient']}|{c['arm']}")
            draw = c["draw"] if c["draw"] not in (None, "") else 0
            if split == "test":
                units.add(unit)
            st = dict(nodes=True, judgments=c["n_pairs_decided"] or None, vote_fraction=True)
            G = c["gold_pairs_in_universe"]
            for k in keys + ["connected_components_identity"]:
                rk = "connected_components" if k == "connected_components_identity" else k
                rr = c["rules"][rk]
                recs.append(C.Record(
                    op=k, pool=f"{ds}:{split}", unit=unit, cluster=ds, repeat=draw,
                    counts=dict(T=rr["true_intra_pairs"], N=rr["predicted_intra_pairs"], G=G),
                    value=rr["cluster_f1"], state=st,
                    inputs=[dict(output="judgments", produced_at_step="pairwise_judgment", producer="agent_matching")],
                    cost=dict(cpu_seconds=rr["cpu_seconds"])))
    return recs, {"full": {}}


def analyze(spec, recs, pops):
    ev, keys = _evaluate()
    # E3: F1 from relation counts equals the stored cluster F1
    e3 = sum(abs(C.f1_from_counts(**r.counts) - r.value) <= 1e-12 for r in recs if r.op != "connected_components_identity")
    n3 = sum(1 for r in recs if r.op != "connected_components_identity")
    # E1: every authoritative rule cell
    rc = list(csv.DictReader(open(GT / "output/eb1_rule_cells.csv")))
    e1 = 0
    for row in rc:
        d = ev[(row["dataset"], "dev")][row["cell"]]["rules"][row["rule"]]
        t = ev[(row["dataset"], "test")][row["cell"]]["rules"][row["rule"]]
        e1 += abs(C.f1_from_counts(d["true_intra_pairs"], d["predicted_intra_pairs"],
                                   ev[(row["dataset"], "dev")][row["cell"]]["gold_pairs_in_universe"]) - float(row["dev_cluster_f1"])) <= 1e-12 \
            and abs(C.f1_from_counts(t["true_intra_pairs"], t["predicted_intra_pairs"],
                                     ev[(row["dataset"], "test")][row["cell"]]["gold_pairs_in_universe"]) - float(row["test_cluster_f1"])) <= 1e-12
    # E2: development selection (labels: dev only), test-best as oracle reference
    sel = {(r["dataset"], r["cell"]): r for r in csv.DictReader(open(GT / "output/eb1_selection.csv"))}
    rows, e2, ident = [], 0, 0
    for (ds, cell), s in sorted(sel.items()):
        dv, tv = ev[(ds, "dev")][cell], ev[(ds, "test")][cell]
        f1 = lambda c, k: C.f1_from_counts(c["rules"][k]["true_intra_pairs"], c["rules"][k]["predicted_intra_pairs"],
                                           c["gold_pairs_in_universe"])
        chosen, tied = C.dev_select({k: f1(dv, k) for k in keys}, keys)
        best = max(keys, key=lambda k: f1(tv, k))
        regret = 100 * (f1(tv, best) - f1(tv, chosen))
        e2 += chosen == s["selected"] and abs(regret - float(s["regret_pp"])) <= 1e-9
        ident += f1(tv, "connected_components") == tv["rules"]["connected_components"]["cluster_f1"]
        rows.append(dict(dataset=ds, cell=cell, dev_tie_set_size=len(tied), selected=chosen,
                         selected_test_cluster_f1=f1(tv, chosen), test_best_oracle_reference=best,
                         test_best_cluster_f1=f1(tv, best), regret_pp=regret,
                         T=tv["rules"][chosen]["true_intra_pairs"], N=tv["rules"][chosen]["predicted_intra_pairs"],
                         G=tv["gold_pairs_in_universe"]))
    got = dict(E1=dict(value=[int(e1), len(rc)]), E2=dict(value=[int(e2), len(sel)]),
               E3=dict(value=[int(e3), n3]), ID1=dict(value=[int(ident), len(sel)]))
    cost_rows = [dict(component="pairwise_judgment_calls", note="shared prefix, identical across rules; per cell in eb1_selection.csv calls/prompt_tokens/completion_tokens")]
    for k in keys:
        cpu = [r.cost["cpu_seconds"] for r in recs if r.op == k and r.pool.endswith(":test")]
        cost_rows.append(dict(component="grouping_cpu", operation=k, cells=len(cpu), total_cpu_seconds_test=round(sum(cpu), 4)))
    with open(C.ROOT / "output" / "em_selection_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return got, cost_rows, dict(selection_rows_file="output/em_selection_rows.csv")
