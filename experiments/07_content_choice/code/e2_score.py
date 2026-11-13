"""Recompute development-selected content, held-out pair F1 and nested development-size comparisons from saved final-channel responses. No model calls."""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
OUT = WORK / "output"
RAW = OUT / "raw"
DATA = WORK / "data"
sys.path.insert(0, str(HERE))
import e2_build as E2                                                      # noqa: E402

MENU = E2.MENU
BOOT_DRAWS = 10000
BOOT_SEED = 20261028
ARM_ORDER = {a: i for i, a in enumerate(sorted(MENU))}
DEV_SIZES = (125, 250, 500)

#: Published token rates used to convert measured usage
PRICES = [
    dict(name="DeepInfra Qwen/Qwen3-32B", inp=0.08, out=0.28),
    dict(name="Groq openai/gpt-oss-120b", inp=0.15, out=0.60),
]


def read_decision(text: str) -> Optional[str]:
    if "Answer:" not in text:
        return None
    tail = text.split("Answer:")[-1].lower()
    i_non = min([x for x in (tail.find("non-match"), tail.find("non match")) if x >= 0],
                default=-1)
    i_m = tail.find("match")
    if i_non >= 0 and (i_m < 0 or i_non <= i_m):
        return "non-match"
    return "match" if i_m >= 0 else None


def prf(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] == "match")
    fp = sum(1 for r in rows if r["pred"] == "match" and r["gold"] != "match")
    fn = sum(1 for r in rows if r["pred"] != "match" and r["gold"] == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=(2 * p * rc / (p + rc) if p + rc else 0.0), precision=p, recall=rc,
                accuracy=sum(1 for r in rows if (r["pred"] or "non-match") == r["gold"]) / len(rows),
                tp=tp, fp=fp, fn=fn, n=len(rows))


def cluster_boot(pairs: List[Tuple[str, float]], rng) -> Dict[str, float]:
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
              for l in open(DATA / "e2_recipient_inputs.jsonl")}
    gold = json.loads((DATA / "e2_gold.json").read_text(encoding="utf-8"))
    ent = json.loads((DATA / "e2_entity_of.json").read_text(encoding="utf-8"))

    def cluster_of(u: Dict[str, Any]) -> str:
        a = ent.get(f"A{u['ltable_id']}", f"A{u['ltable_id']}")
        b = ent.get(f"B{u['rtable_id']}", f"B{u['rtable_id']}")
        return a if a == b else "|".join(sorted((a, b)))

    per_model: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    health: List[Dict[str, Any]] = []
    for path in sorted(RAW.glob("e2_recipient_*.jsonl")):
        model = path.name[len("e2_recipient_"):-len(".jsonl")]
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
            pred = read_decision(content)
            rows[f"{u['split']}|{u['arm']}"].append(dict(
                pair=f"L:{u['ltable_id']}|R:{u['rtable_id']}", split=u["split"], arm=u["arm"],
                cluster=cluster_of(u),
                gold=gold[f"amazon_google|{u['split']}|L:{u['ltable_id']}|R:{u['rtable_id']}"],
                pred=pred, unreadable=pred is None, empty_final=not content.strip(),
                truncated=ch.get("finish_reason") == "length",
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
                reasoning_tokens=det.get("reasoning_tokens") or 0,
                latency_s=r.get("latency_s") or 0.0))
        per_model[model] = rows
        for key, rs in sorted(rows.items()):
            sp, arm = key.split("|")
            health.append(dict(
                recipient=model, split=sp, arm=arm, requests=len(rs),
                unreadable_share=round(sum(r["unreadable"] for r in rs) / len(rs), 5),
                empty_final_share=round(sum(r["empty_final"] for r in rs) / len(rs), 5),
                truncated_share=round(sum(r["truncated"] for r in rs) / len(rs), 5),
                mean_prompt_tokens=round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                mean_completion_tokens=round(float(np.mean([r["completion_tokens"] for r in rs])), 1),
                mean_reasoning_tokens=round(float(np.mean([r["reasoning_tokens"] for r in rs])), 1),
                mean_latency_s=round(float(np.mean([r["latency_s"] for r in rs])), 3)))
        print(f"{model}: statuses={dict(statuses)} cells={len(rows)}", flush=True)

    # ---------------------------------------------- development selection, before any test read
    selection: Dict[str, Dict[str, Any]] = {}
    devsize: List[Dict[str, Any]] = []
    for model, rows in sorted(per_model.items()):
        dev = {arm: prf(rows[f"dev|{arm}"]) for arm in MENU if rows.get(f"dev|{arm}")}
        if len(dev) < len(MENU):
            print(f"{model}: development side incomplete ({sorted(dev)}); skipping")
            continue
        tok = {arm: float(np.mean([r["prompt_tokens"] for r in rows[f"dev|{arm}"]])) for arm in dev}
        best = max(d["f1"] for d in dev.values())
        tied = [a for a in dev if dev[a]["f1"] == best]
        chosen = sorted(tied, key=lambda a: (tok[a], ARM_ORDER[a]))[0]
        selection[model] = dict(
            dev_f1={a: round(100 * dev[a]["f1"], 3) for a in sorted(dev)},
            dev_accuracy={a: round(100 * dev[a]["accuracy"], 3) for a in sorted(dev)},
            dev_mean_prompt_tokens={a: round(tok[a], 1) for a in sorted(dev)},
            tied_arms=sorted(tied), selected_arm=chosen)
        print(f"{model}: development selection -> {chosen} "
              f"(dev F1 {100*dev[chosen]['f1']:.2f})", flush=True)

        # nested development subsets, cached outputs only
        order = sorted(rows[f"dev|{MENU[0]}"], key=lambda r: r["pair"])
        for k in DEV_SIZES:
            keep = {r["pair"] for r in order[:k]}
            sub = {arm: prf([r for r in rows[f"dev|{arm}"] if r["pair"] in keep]) for arm in MENU}
            b = max(d["f1"] for d in sub.values())
            t = [a for a in sub if sub[a]["f1"] == b]
            pick = sorted(t, key=lambda a: (tok[a], ARM_ORDER[a]))[0]
            devsize.append(dict(recipient=model, dev_pairs=min(k, len(order)),
                                selected_arm=pick,
                                same_as_full_dev=bool(pick == chosen),
                                dev_f1={a: round(100 * sub[a]["f1"], 3) for a in sorted(sub)},
                                note="nested subset of the same development pairs; not an "
                                     "independent repeat"))

    # ---------------------------------------------- one test read
    levels, contrasts, props = [], [], []
    for model, rows in sorted(per_model.items()):
        if model not in selection:
            continue
        chosen = selection[model]["selected_arm"]
        test = {arm: rows.get(f"test|{arm}") for arm in MENU}
        if any(not v for v in test.values()):
            print(f"{model}: test side incomplete; skipping test read")
            continue
        for arm, rs in test.items():
            s = prf(rs)
            levels.append(dict(recipient=model, arm=arm, is_dev_selected=(arm == chosen),
                               requests=len(rs), f1_match=round(100 * s["f1"], 3),
                               precision=round(100 * s["precision"], 3),
                               recall=round(100 * s["recall"], 3),
                               accuracy=round(100 * s["accuracy"], 3),
                               tp=s["tp"], fp=s["fp"], fn=s["fn"]))
        base = {r["pair"]: r for r in test["no_peer"]}
        for arm in MENU:
            if arm == "no_peer":
                continue
            diffs, ben, harm = [], 0, 0
            for r in test[arm]:
                b = base.get(r["pair"])
                if b is None:
                    continue
                ok_a = (r["pred"] or "non-match") == r["gold"]
                ok_b = (b["pred"] or "non-match") == b["gold"]
                diffs.append((r["cluster"], float(ok_a) - float(ok_b)))
                ben += int(ok_a and not ok_b)
                harm += int(ok_b and not ok_a)
            v = cluster_boot(diffs, np.random.default_rng(BOOT_SEED))
            contrasts.append(dict(
                recipient=model, contrast=f"{arm} - no_peer",
                is_dev_selected=(arm == chosen), paired_pairs=len(diffs),
                entity_clusters=v["n_clusters"],
                accuracy_effect_pp=round(100 * v["mean"], 3),
                lo_pp=round(100 * v["lo"], 3), hi_pp=round(100 * v["hi"], 3),
                ci_half_width_pp=round(100 * (v["hi"] - v["lo"]) / 2, 3),
                ci_excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0), p_boot=v["p"],
                benefited_pairs=ben, harmed_pairs=harm,
                unchanged_pairs=len(diffs) - ben - harm))

        # the three propositions, kept apart
        lv = {r["arm"]: r for r in levels if r["recipient"] == model}
        worst = min(lv, key=lambda a: lv[a]["f1_match"])
        bestt = max(lv, key=lambda a: lv[a]["f1_match"])
        gain = next((c["accuracy_effect_pp"] for c in contrasts
                     if c["recipient"] == model and c["contrast"] == f"{chosen} - no_peer"), 0.0)
        res = {}
        for arm in MENU:
            rs = rows[f"test|{arm}"]
            res[arm] = dict(
                mean_prompt_tokens=round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                mean_completion_tokens=round(float(np.mean([r["completion_tokens"] for r in rs])), 1))
        d_in = res[chosen]["mean_prompt_tokens"] - res["no_peer"]["mean_prompt_tokens"]
        d_out = res[chosen]["mean_completion_tokens"] - res["no_peer"]["mean_completion_tokens"]
        usd = {p["name"]: round((d_in * p["inp"] + d_out * p["out"]) / 1e6 * 1000, 6)
               for p in PRICES}
        props.append(dict(
            recipient=model, selected_arm=chosen,
            avoiding_loss_vs_each_fixed_arm={
                a: round(lv[chosen]["f1_match"] - lv[a]["f1_match"], 3) for a in sorted(lv)},
            avoiding_loss_vs_worst_fixed_arm=dict(
                arm=worst, f1_gap_pp=round(lv[chosen]["f1_match"] - lv[worst]["f1_match"], 3)),
            obtaining_gain_vs_no_peer_pp=gain,
            near_optimality_regret_pp=round(lv[bestt]["f1_match"] - lv[chosen]["f1_match"], 3),
            retrospective_best_arm=bestt,
            full_cost=dict(delta_input_tokens=round(d_in, 1),
                           delta_output_tokens=round(d_out, 1),
                           usd_per_1000_decisions=usd,
                           note="measured tokens priced at a published rate; a pricing scenario, "
                                "not an observed bill")))
    if contrasts:
        for c, f in zip(contrasts, holm([c["p_boot"] for c in contrasts])):
            c["holm_significant_in_declared_family"] = bool(f)
            c["declared_family_size"] = len(contrasts)

    summary = dict(manifest=json.loads((DATA / "e2_manifest.json").read_text(encoding="utf-8")),
                   ceiling_check=json.loads((OUT / "e2_devprobe_ceiling.json").read_text())
                   if (OUT / "e2_devprobe_ceiling.json").exists() else None,
                   health=health, selection=selection, dev_size_stability=devsize,
                   test_levels=levels, test_contrasts_vs_no_peer=contrasts,
                   three_propositions=props,
                   design=dict(menu=list(MENU), boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                               independent_unit="entity ID for within-entity pairs; unordered entity-ID pair otherwise",
                               answer_source="final channel only",
                               multiplicity="Holm across the nine accuracy contrasts",
                               abstention="charged as a non-match prediction and reported separately"))
    (OUT / "e2_scored.json").write_text(json.dumps(summary, indent=1, sort_keys=True),
                                        encoding="utf-8")
    for name, rows_ in (("e2_test_levels", levels), ("e2_test_contrasts", contrasts),
                        ("e2_health", health), ("e2_dev_size_stability", devsize)):
        if not rows_:
            continue
        keys = sorted({k for r in rows_ for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_)

    print("\ntest levels (all fixed arms)")
    for r in levels:
        print(f"  {r['recipient']:14s} {r['arm']:22s}{'*' if r['is_dev_selected'] else ' '} "
              f"F1 {r['f1_match']:6.2f}  acc {r['accuracy']:6.2f}  P {r['precision']:6.2f} "
              f"R {r['recall']:6.2f}")
    print("\ntest contrasts vs no_peer")
    for r in contrasts:
        print(f"  {r['recipient']:14s} {r['contrast']:30s} {r['accuracy_effect_pp']:+7.2f} pp "
              f"[{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}] "
              f"holm={r.get('holm_significant_in_declared_family')} "
              f"benefit={r['benefited_pairs']} harm={r['harmed_pairs']}")
    print("\nthe three propositions, kept apart")
    for p in props:
        print(f"  {p['recipient']:14s} selected={p['selected_arm']}")
        print(f"     avoiding loss vs worst fixed arm ({p['avoiding_loss_vs_worst_fixed_arm']['arm']}): "
              f"{p['avoiding_loss_vs_worst_fixed_arm']['f1_gap_pp']:+.2f} pp F1")
        print(f"     obtaining gain vs no_peer: {p['obtaining_gain_vs_no_peer_pp']:+.2f} pp accuracy")
        print(f"     regret vs retrospective best ({p['retrospective_best_arm']}): "
              f"{p['near_optimality_regret_pp']:+.2f} pp F1")
    print("\ndev-size stability (nested, cached only)")
    for r in devsize:
        print(f"  {r['recipient']:14s} dev={r['dev_pairs']:4d} -> {r['selected_arm']:22s} "
              f"same_as_full={r['same_as_full_dev']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
