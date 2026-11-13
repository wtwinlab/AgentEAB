#!/usr/bin/env python3
"""MAD-M²'s released pruning pair as the which-peers selection contrast. Zero model calls.

    python3 analyze_selection_variants.py
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "code", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "workspace/controlled_sharing"
SNAP = OUT / "reused_sources/protocol_all_matching_20261018T192619Z"
DERIVED = OUT / "derived"
OKMARK = '"status": "ok"'
MATCHING = ("wdc_products", "geographic_settlements")
CKPT = {"llama3-8b-instruct": "llama3-8b", "qwen2.5-7b-instruct": "qwen2.5-7b"}
BOOT = 2000
SEED = 20261019
SCRIPT_VERSION = "1"


def cluster_bootstrap(pairs: List[Tuple[str, float]]) -> Tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    by: Dict[str, List[float]] = collections.defaultdict(list)
    for u, d in pairs:
        by[u].append(d)
    units = list(by)
    point = sum(d for ds in by.values() for d in ds) / sum(len(ds) for ds in by.values())
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOT):
        draw = [by[units[rng.randrange(len(units))]] for _ in units]
        flat = [d for ds in draw for d in ds]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    if len(means) < 20:
        return point, float("nan"), float("nan")
    return point, means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main() -> int:
    import arm_transform as AT
    import collect_all_matching as C
    import collect_controlled as CC
    import matching_tasks as MT
    MT.install()
    C._ensure_installed()

    ready = json.loads((OUT / "reused_sources/READY.json").read_text(encoding="utf-8"))
    print(f"snapshot: {ready['files']} files, published {ready['published_utc']}\n")

    DERIVED.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"script": pathlib.Path(__file__).name,
                              "script_version": SCRIPT_VERSION,
                              "snapshot_published_utc": ready["published_utc"],
                              "params": {"bootstrap": BOOT, "seed": SEED,
                                         "repeats_used": [0, 1]},
                              "benchmarks": {}}

    for bench in MATCHING:
        frozen_list, frozen_sha = CC.frozen_units(bench)
        frozen = set(frozen_list)
        gold = {}
        pop, _ = C.load_population(bench)
        for row in pop:
            k = row.get("unit_id") if isinstance(row, dict) else row[0]
            g = row.get("gold") if isinstance(row, dict) else (row[2] if len(row) > 2 else None)
            if k in frozen and g is not None:
                # the population's gold is 0/1 while the protocols answer `match`/`non-match`.
                gold[k] = ("match" if str(g).strip() in ("1", "true", "True", "match")
                           else "non-match")

        # ---- per variant: the answer and the peer count actually rendered
        ans: Dict[Tuple[str, str, str, int], Optional[str]] = {}
        peers = collections.defaultdict(list)
        qsha = collections.defaultdict(dict)
        counts = collections.Counter()
        for variant in ("madm2_objective", "madm2_naive"):
            for shard in sorted(glob.glob(str(SNAP / "raw" / f"{bench}__{variant}__*.jsonl"))):
                for line in open(shard, encoding="utf-8"):
                    if OKMARK not in line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    parts = (r.get("unit_id") or "").split("|")
                    if len(parts) < 6:
                        continue
                    unit = "|".join(parts[3:])
                    if unit not in frozen or r.get("repeat") not in (0, 1):
                        continue
                    rec = r.get("record") or {}
                    calls = rec.get("calls") or []
                    served = {c.get("served_model") for c in calls}
                    if len(served) != 1:
                        continue
                    bb = CKPT.get(next(iter(served)), "")
                    if not bb:
                        continue
                    key = (variant, unit, bb, r["repeat"])
                    a = rec.get("majority_answer")
                    ans[key] = (str(a).strip().lower().replace("_", "-")
                                if a not in (None, "", "None", "none") else None)
                    qsha[variant][unit] = r.get("question_sha")
                    counts[variant] += 1
                    # how many peers each rule actually rendered, from the stored prompt
                    for c in calls:
                        for m in c.get("messages_sent") or []:
                            n = len(AT.peer_blocks("madm2_objective", m.get("content") or ""))
                            if n:
                                peers[variant].append(n)

        # ---- the gate: same input, and same number of peers?
        shared_q = [u for u in qsha["madm2_objective"]
                    if u in qsha["madm2_naive"]
                    and qsha["madm2_objective"][u] == qsha["madm2_naive"][u]]
        mismatch_q = [u for u in qsha["madm2_objective"]
                      if u in qsha["madm2_naive"]
                      and qsha["madm2_objective"][u] != qsha["madm2_naive"][u]]
        mean_peers = {v: (sum(x) / len(x) if x else 0.0) for v, x in peers.items()}
        same_count = (abs(mean_peers.get("madm2_objective", 0)
                          - mean_peers.get("madm2_naive", 0)) < 0.01)

        print(f"-- {bench} --")
        print(f"   records used: objective {counts['madm2_objective']:,}, "
              f"naive {counts['madm2_naive']:,}")
        print(f"   units with an identical question sha under both rules: {len(shared_q):,}"
              + (f"  ({len(mismatch_q)} differ and are excluded)" if mismatch_q else ""))
        print(f"   mean peers rendered per peer-bearing call: "
              f"objective {mean_peers.get('madm2_objective', 0):.3f}, "
              f"naive {mean_peers.get('madm2_naive', 0):.3f}  -> "
              + ("counts match: a clean which-peers contrast"
                 if same_count else "COUNTS DIFFER: not a clean which-peers contrast"))

        # ---- paired contrast, naive minus objective. Matching is scored with F1 on the match
        import analyze_benchmark as AB
        res = {}
        shared = set(shared_q)
        for scope in ("pooled", "llama3-8b", "qwen2.5-7b"):
            rows, dlv = [], []
            for (v, unit, bb, rep), got in ans.items():
                if v != "madm2_naive" or unit not in shared:
                    continue
                if scope != "pooled" and bb != scope:
                    continue
                base_key = ("madm2_objective", unit, bb, rep)
                if base_key not in ans:
                    continue
                base = ans[base_key]
                dlv.append((unit, (1.0 if got else 0.0) - (1.0 if base else 0.0)))
                g = gold.get(unit)
                if g is not None:
                    rows.append((unit, got, base, g))
            f_naive, f_obj, ap, alo, ahi = AB.paired_f1_bootstrap(rows)
            dp, dlo, dhi = cluster_bootstrap(dlv)
            n_scored = sum(1 for _u, a_, b_, _g in rows if a_ is not None and b_ is not None)
            res[scope] = {"metric": "F1(match)", "n_pairs_scored": n_scored,
                          "n_pairs_delivery": len(dlv),
                          "f1_naive": f_naive, "f1_objective": f_obj,
                          "acc_diff": ap, "acc_lo": alo, "acc_hi": ahi,
                          "dlv_diff": dp, "dlv_lo": dlo, "dlv_hi": dhi}
            if len(dlv):
                print(f"   {scope:<12} n={n_scored:,}/{len(dlv):,}  F1 naive {f_naive:.4f} vs "
                      f"objective {f_obj:.4f}  diff {ap:+.4f} [{alo:+.4f}, {ahi:+.4f}]")
        report["benchmarks"][bench] = {
            "frozen_sha": frozen_sha, "records": dict(counts),
            "units_same_question": len(shared_q), "units_question_mismatch": len(mismatch_q),
            "mean_peers_rendered": mean_peers, "same_peer_count": same_count,
            "contrast_naive_minus_objective": res,
        }
        print()

    blob = json.dumps(report, indent=1, default=str) + "\n"
    out_json = DERIVED / "selection_variants_madm2.json"
    out_json.write_text(blob, encoding="utf-8")
    out_sha = hashlib.sha256(blob.encode()).hexdigest()

    L = ["# MAD-M²'s released pruning pair: the which-peers selection contrast", "",
         f"Derived {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
         f"`analyze_selection_variants.py` v{SCRIPT_VERSION} from the verified snapshot published "
         f"{ready['published_utc']}. Zero model calls.", "",
         "The live UNIFSEL arm keeps a uniform random half of the peers the released prompt "
         "rendered, which changes how many peers are present as well as which, so it is a point on "
         "an access-volume series. A which-peers contrast cannot be built by message surgery: the "
         "prompt holds only the peers the protocol already chose. MAD-M² ships two selection rules "
         "and both were collected on these populations, so pairing them on the same unit, backbone "
         "and repeat gives the contrast the surgery cannot.", ""]
    for bench, b in report["benchmarks"].items():
        L += [f"## {bench}", "",
              f"- records used: objective {b['records'].get('madm2_objective', 0):,}, "
              f"naive {b['records'].get('madm2_naive', 0):,} (frozen units, repeats 0 and 1)",
              f"- units with an identical question sha under both rules: "
              f"{b['units_same_question']:,}"
              + (f"; {b['units_question_mismatch']} differ and are excluded"
                 if b["units_question_mismatch"] else ""),
              f"- mean peers rendered per peer-bearing call: objective "
              f"{b['mean_peers_rendered'].get('madm2_objective', 0):.3f}, naive "
              f"{b['mean_peers_rendered'].get('madm2_naive', 0):.3f} — "
              + ("**counts match**, so the criterion is the only thing that differs"
                 if b["same_peer_count"] else
                 "**counts differ**, so this contrast mixes the selection criterion with the "
                 "amount of peer information and must not be read as a which-peers result"), "",
              "| scope | n pairs | F1 naive vs objective | difference [95% CI] | "
              "delivery diff [95% CI] |", "|---|---:|---|---|---|"]
        for scope, r in b["contrast_naive_minus_objective"].items():
            if not r["n_pairs_delivery"]:
                continue
            L.append(f"| {scope} | {r['n_pairs_scored']:,}/{r['n_pairs_delivery']:,} | "
                     f"{r.get('f1_naive', float('nan')):.4f} vs "
                     f"{r.get('f1_objective', float('nan')):.4f} | "
                     f"{r['acc_diff']:+.4f} [{r['acc_lo']:+.4f}, {r['acc_hi']:+.4f}] | "
                     f"{r['dlv_diff']:+.4f} [{r['dlv_lo']:+.4f}, {r['dlv_hi']:+.4f}] |")
        L += [""]
    L += ["## Reading these intervals", "",
          "Pairing is on (unit, backbone, repeat); the bootstrap resamples units, not rows. "
          "An interval containing zero means the two rules are **not distinguished at this sample "
          "size** — not that they are equivalent: no margin was pre-specified. Per-backbone rows "
          "are given because an effect on one backbone and not the other is a finding, and pooling "
          "would hide it.", "",
          "## Provenance", "",
          f"- input: `reused_sources/protocol_all_matching_20261018T192619Z/raw/*madm2_*` "
          f"(hashes in `reused_sources/FILE_MANIFEST.jsonl`)",
          f"- output: `derived/selection_variants_madm2.json`, SHA-256 `{out_sha[:32]}…`",
          "- the historical tree was not read by this script and is unchanged", ""]
    (DERIVED / "SELECTION_VARIANTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {DERIVED / 'SELECTION_VARIANTS.md'}")
    print(f"   output sha256 {out_sha[:32]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
