"""Select DBLP–ACM content on development pairs and score every alternative on held-out entities. No model calls."""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
OUT = WORK / "output"
RAW = OUT / "raw"
DATA = WORK / "data"
for _p in (str(REPO / "code"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mt1_build_dblp_acm as MT1                                           # noqa: E402

MENU = MT1.MENU
BOOT_DRAWS = 10000
BOOT_SEED = 20261028
RULE_ORDER = {a: i for i, a in enumerate(sorted(MENU))}


def read_decision(text: str) -> Optional[str]:
    """The answer the recipient delivered, read from the final channel only."""
    if "Answer:" not in text:
        return None
    tail = text.split("Answer:")[-1].lower()
    i_non = min([x for x in (tail.find("non-match"), tail.find("non match")) if x >= 0],
                default=-1)
    i_m = tail.find("match")
    if i_non >= 0 and (i_m < 0 or i_non <= i_m):
        return "non-match"
    if i_m >= 0:
        return "match"
    return None


def prf(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """F1 on the match class. An unreadable delivery is charged as a non-match prediction."""
    tp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] == "match")
    fp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] != "match")
    fn = sum(1 for r in rows if r["pred"] != "match" and r["gold"] == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    acc = sum(1 for r in rows if (r["pred"] or "non-match") == r["gold"]) / len(rows)
    return dict(f1=(2 * p * rc / (p + rc) if p + rc else 0.0), precision=p, recall=rc,
                accuracy=acc, tp=tp, fp=fp, fn=fn, n=len(rows))


def cluster_boot_diff(pairs: List[Tuple[str, float]], rng) -> Dict[str, float]:
    """Resample entity clusters, not pairs: two records of one entity are one unit."""
    by = collections.defaultdict(list)
    for cid, d in pairs:
        by[cid].append(d)
    ids = sorted(by)
    per = np.array([float(np.mean(by[c])) for c in ids])
    draws = per[rng.integers(0, len(per), size=(BOOT_DRAWS, len(per)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(mean=float(per.mean()), lo=float(lo), hi=float(hi), n_clusters=len(ids),
                p=float(min(1.0, max(p, 1.0 / BOOT_DRAWS))))


def holm(ps: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def main() -> int:
    inputs = {json.loads(l)["unit"]: json.loads(l)
              for l in open(DATA / "mt_recipient_inputs.jsonl")}
    gold = json.loads((DATA / "mt_dblp_acm_gold.json").read_text(encoding="utf-8"))

    # the entity cluster each pair belongs to, rebuilt by the same construction as the split
    pairs_all = MT1.load_pairs()
    node_side, _ = MT1.entity_split(pairs_all)
    par: Dict[Any, Any] = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par.setdefault(par[x], par[x])
            x = par[x]
        return x
    for l, r, lab, _ in pairs_all:
        find(("A", l))
        find(("B", r))
        if lab == 1:
            ra, rb = find(("A", l)), find(("B", r))
            if ra != rb:
                par[ra] = rb

    def cluster_id(u: Dict[str, Any]) -> str:
        a, b = find(("A", u["ltable_id"])), find(("B", u["rtable_id"]))
        return "|".join(sorted(f"{t}{i}" for t, i in {a, b}))

    per_model: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    health: List[Dict[str, Any]] = []
    for path in sorted(RAW.glob("mt_*.jsonl")):
        if path.name.startswith("mt_peer"):
            continue
        model = path.name[len("mt_"):-len(".jsonl")]
        rows: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        statuses = collections.Counter()
        for line in open(path):
            r = json.loads(line)
            statuses[r["status"]] += 1
            if r.get("status") != "ok":
                continue
            u = inputs.get(r["unit_id"])
            if u is None:
                continue
            ch = r["response"]["choices"][0]
            content = ch["message"].get("content") or ""
            usage = r["response"].get("usage") or {}
            det = usage.get("completion_tokens_details") or {}
            key = f"{u['split']}|{u['arm']}"
            rows[key].append(dict(
                unit=r["unit_id"], split=u["split"], arm=u["arm"],
                cluster=cluster_id(u),
                gold=gold[f"dblp_acm|{u['split']}|L:{u['ltable_id']}|R:{u['rtable_id']}"],
                pred=read_decision(content),
                unreadable=read_decision(content) is None,
                empty_final=not content.strip(),
                truncated=ch.get("finish_reason") == "length",
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
                reasoning_tokens=det.get("reasoning_tokens") or 0,
                latency_s=r.get("latency_s") or 0.0))
        per_model[model] = rows
        for key, rs in sorted(rows.items()):
            sp, arm = key.split("|")
            health.append(dict(recipient=model, split=sp, arm=arm, requests=len(rs),
                               unreadable_share=round(sum(r["unreadable"] for r in rs) / len(rs), 5),
                               empty_final_share=round(sum(r["empty_final"] for r in rs) / len(rs), 5),
                               truncated_share=round(sum(r["truncated"] for r in rs) / len(rs), 5),
                               mean_prompt_tokens=round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                               mean_completion_tokens=round(float(np.mean([r["completion_tokens"] for r in rs])), 1),
                               mean_reasoning_tokens=round(float(np.mean([r["reasoning_tokens"] for r in rs])), 1),
                               mean_latency_s=round(float(np.mean([r["latency_s"] for r in rs])), 3)))
        print(f"{model}: statuses={dict(statuses)} cells={len(rows)}", flush=True)

    # ------------------------------------------------- development selection, before any test read
    selection: Dict[str, Dict[str, Any]] = {}
    for model, rows in sorted(per_model.items()):
        dev = {arm: prf(rows[f"dev|{arm}"]) for arm in MENU if rows.get(f"dev|{arm}")}
        if not dev:
            continue
        tok = {arm: float(np.mean([r["prompt_tokens"] for r in rows[f"dev|{arm}"]]))
               for arm in dev}
        best = max(d["f1"] for d in dev.values())
        tied = [a for a in dev if dev[a]["f1"] == best]
        chosen = sorted(tied, key=lambda a: (tok[a], RULE_ORDER[a]))[0]
        selection[model] = dict(dev_f1={a: round(100 * dev[a]["f1"], 3) for a in sorted(dev)},
                                dev_accuracy={a: round(100 * dev[a]["accuracy"], 3) for a in sorted(dev)},
                                dev_mean_prompt_tokens={a: round(tok[a], 1) for a in sorted(dev)},
                                tied_arms=sorted(tied), selected_arm=chosen)
        print(f"{model}: development selection -> {chosen} (dev F1 "
              f"{100*dev[chosen]['f1']:.2f}, tied {sorted(tied)})", flush=True)

    # ------------------------------------------------- one test read
    levels, contrasts = [], []
    for model, rows in sorted(per_model.items()):
        if model not in selection:
            continue
        chosen = selection[model]["selected_arm"]
        test = {arm: rows.get(f"test|{arm}") for arm in MENU}
        for arm, rs in test.items():
            if not rs:
                continue
            s = prf(rs)
            levels.append(dict(recipient=model, arm=arm, is_dev_selected=(arm == chosen),
                               requests=len(rs),
                               f1_match=round(100 * s["f1"], 3),
                               precision=round(100 * s["precision"], 3),
                               recall=round(100 * s["recall"], 3),
                               accuracy=round(100 * s["accuracy"], 3),
                               tp=s["tp"], fp=s["fp"], fn=s["fn"]))
        base = test.get("no_peer")
        if base:
            bykey = {r["unit"].rsplit("|", 1)[0]: r for r in base}
            for arm in MENU:
                if arm == "no_peer" or not test.get(arm):
                    continue
                rng = np.random.default_rng(BOOT_SEED)
                diffs, ben, harm = [], 0, 0
                for r in test[arm]:
                    b = bykey.get(r["unit"].rsplit("|", 1)[0])
                    if b is None:
                        continue
                    ok_a = (r["pred"] or "non-match") == r["gold"]
                    ok_b = (b["pred"] or "non-match") == b["gold"]
                    diffs.append((r["cluster"], float(ok_a) - float(ok_b)))
                    ben += int(ok_a and not ok_b)
                    harm += int(ok_b and not ok_a)
                v = cluster_boot_diff(diffs, rng)
                contrasts.append(dict(
                    recipient=model, contrast=f"{arm} - no_peer",
                    is_dev_selected=(arm == chosen), paired_pairs=len(diffs),
                    entity_clusters=v["n_clusters"],
                    accuracy_effect_pp=round(100 * v["mean"], 3),
                    lo_pp=round(100 * v["lo"], 3), hi_pp=round(100 * v["hi"], 3),
                    ci_half_width_pp=round(100 * (v["hi"] - v["lo"]) / 2, 3),
                    ci_excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0), p_boot=v["p"],
                    benefited_pairs=ben, harmed_pairs=harm,
                    unchanged_pairs=len(diffs) - ben - harm,
                    residual_errors=sum(1 for r in test[arm]
                                        if (r["pred"] or "non-match") != r["gold"])))
    if contrasts:
        for c, f in zip(contrasts, holm([c["p_boot"] for c in contrasts])):
            c[f"holm_significant_in_family_of_{len(contrasts)}"] = bool(f)

    # the selected arm against every fixed arm, on test
    vs_fixed = []
    for model in sorted(selection):
        chosen = selection[model]["selected_arm"]
        lv = {r["arm"]: r for r in levels if r["recipient"] == model}
        if chosen not in lv:
            continue
        best_arm = max(lv, key=lambda a: lv[a]["f1_match"])
        for arm in sorted(lv):
            vs_fixed.append(dict(recipient=model, fixed_arm=arm, selected_arm=chosen,
                                 selected_minus_fixed_f1_pp=round(
                                     lv[chosen]["f1_match"] - lv[arm]["f1_match"], 3)))
        vs_fixed.append(dict(recipient=model, fixed_arm="RETROSPECTIVE_BEST",
                             selected_arm=chosen,
                             selected_minus_fixed_f1_pp=round(
                                 lv[chosen]["f1_match"] - lv[best_arm]["f1_match"], 3),
                             retrospective_best_arm=best_arm))

    summary = dict(manifest=json.loads((DATA / "mt_dblp_acm_manifest.json").read_text()),
                   health=health, selection=selection, test_levels=levels,
                   test_contrasts_vs_no_peer=contrasts, selected_vs_fixed_on_test=vs_fixed,
                   design=dict(menu=list(MENU), boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                               independent_unit="entity cluster",
                               answer_source="final channel only",
                               abstention="charged as a non-match prediction and reported separately"))
    (OUT / "mt2_scored.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                         encoding="utf-8")
    for name, rows in (("mt2_test_levels", levels), ("mt2_test_contrasts", contrasts),
                       ("mt2_selected_vs_fixed", vs_fixed), ("mt2_health", health)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("\ntest levels (all fixed arms reported)")
    for r in levels:
        print(f"  {r['recipient']:14s} {r['arm']:22s} {'*' if r['is_dev_selected'] else ' '} "
              f"F1 {r['f1_match']:6.2f}  acc {r['accuracy']:6.2f}  P {r['precision']:6.2f} "
              f"R {r['recall']:6.2f}")
    print("\ntest contrasts against no_peer")
    for r in contrasts:
        hk = [k for k in r if k.startswith("holm_")][0]
        print(f"  {r['recipient']:14s} {r['contrast']:30s} {r['accuracy_effect_pp']:+7.2f} pp "
              f"[{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}] holm={r[hk]} "
              f"benefit={r['benefited_pairs']} harm={r['harmed_pairs']}")
    print("\nselected arm minus each fixed arm, test F1(match) pp")
    for r in vs_fixed:
        print(f"  {r['recipient']:14s} vs {r['fixed_arm']:22s} "
              f"{r['selected_minus_fixed_f1_pp']:+7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
