r"""Build read-only merged run directories for the analysis (zero model calls).

    python3 cn_build_records.py ext|sb|c5

Each merged dir = symlinks to every original shard, EXCEPT the Du-MAD / BCCS NOPEER shards named
below, which are replaced by the no-peer records of this collection (arm NOPEERC), relabelled arm='NOPEER' (field
`_relabelled_from='NOPEERC'` kept). For SQL dirs, `analysis/sql_per_unit.jsonl` keeps the original
rows for every non-merged key and gets FRESH rows for the merged keys (sb_fast_eval semantics:
majority_answer -> unwrap -> ext_sql_verdicts._one); `analysis/sql_verdicts.json` = original + the
new verdicts. Nothing outside work/ is written.
"""
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORK = HERE.parent / "work"
CLEAN = HERE.parent / "data"
SRC = {"ext": REPO / "data/records/benchmark_extension",
       "sb": REPO / "data/records/backbone_scale_cross_family",
       "c5": REPO / "data/records/controlled_5x5_collection"}
#: (benchmark, protocol, backbone) merged -> lane dirs of this collection
SWAP = {
    "ext": dict(lanes=["sql/raw/du78", "sql/raw/b78", "sql/raw/gpqa78"],
                cells={(b, p, bb) for b in ("sql_bird_v2", "gpqa") for p in ("du_mad", "bccs")
                       for bb in ("llama3-8b", "qwen2.5-7b")}),
    "sb": dict(lanes=["sql/raw/" + t for t in ("q14a", "q32a", "q72", "agt", "l70b", "gem", "q36", "q38",
                                               "mis32", "glm", "q3", "go20", "go", "mmx", "q7x")],
               cells={("sql_bird_v2", "du_mad", bb) for bb in (
                   "qwen25-14b-awq", "qwen25-32b-awq", "qwen25-72b-awq", "agentar-sql-32b", "llama33-70b",
                   "gemma4-26b-a4b", "qwen36-35b-a3b", "qwen38-27b", "mistral-small-32", "glm47-flash",
                   "qwen3-32b", "gpt-oss-20b", "gpt-oss-120b", "minimax-m25-awq")}
               | {("sql_bird_v2", "bccs", bb) for bb in ("qwen25-14b-awq", "qwen25-32b-awq", "qwen25-72b-awq",
                                                         "qwen2.5-7b-execid")}),
    "c5": dict(lanes=["main/raw/main78", "main/raw/match78"],
               cells={(b, "bccs", bb) for b in ("gsm8k", "strategyqa", "mmlu_pro", "wdc_products",
                                                "geographic_settlements") for bb in ("llama3-8b", "qwen2.5-7b")}
               # Du-MAD: MMLU-Pro Qwen2.5-7B only; the Du-MAD GSM8K, StrategyQA, MMLU-Pro Llama-3-8B, WDC
               # and Settlements no-peer prompts carry no peer text and keep their records.
               | {("mmlu_pro", "du_mad", "qwen2.5-7b")}),
}


def shard_cell(name):
    b, p, arm, bbtag, _r = name[:-6].split("__")
    return b, p, arm, bbtag


def bbtag(bb):
    return bb.replace(".", "")


def main(which):
    src, spec = SRC[which], SWAP[which]
    out = WORK / "merged/results" / src.name
    if out.is_symlink():
        out.unlink()
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)
    for e in src.iterdir():
        if e.name not in ("raw", "analysis"):
            os.symlink(e, out / e.name)
    if (src / "analysis").exists():
        (out / "analysis").mkdir()
        for e in (src / "analysis").iterdir():
            if e.name not in ("sql_per_unit.jsonl", "sql_verdicts.json") or which == "c5":
                os.symlink(e, out / "analysis" / e.name)
    swapped_tags = {(b, p, bbtag(bb)) for b, p, bb in spec["cells"]}
    kept = 0
    for p in sorted((src / "raw").glob("*.jsonl")):
        b, pr, arm, tag = shard_cell(p.name)
        if arm == "NOPEER" and (b, pr, tag) in swapped_tags:
            continue
        os.symlink(p, out / "raw" / p.name)
        kept += 1
    # records of this collection (last ok per key), relabelled
    recs = {}
    for lane in spec["lanes"]:
        d = CLEAN / lane
        for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
            for line in p.open():
                r = json.loads(line)
                if r.get("status") != "ok" or r.get("arm") != "NOPEERC":
                    continue
                if (r["benchmark"], r["protocol"], r["backbone"]) not in spec["cells"]:
                    continue
                recs[(r["benchmark"], r["unit"], r["protocol"], r["backbone"], int(r["repeat"]))] = r
    by_shard = collections.defaultdict(list)
    for (b, u, pr, bb, rep), r in recs.items():
        r = dict(r, arm="NOPEER", _relabelled_from="NOPEERC")
        by_shard[f"{b}__{pr}__NOPEER__{bbtag(bb)}__r{rep}.jsonl"].append(r)
    for name, rs in by_shard.items():
        with open(out / "raw" / name, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    cover = collections.Counter((b, pr, bb) for (b, u, pr, bb, rep) in recs)
    missing = sorted(f"{b}|{p}|{bb}" for b, p, bb in spec["cells"] if not cover.get((b, p, bb)))
    meta = dict(which=which, kept_symlinks=kept, swapped_records=len(recs),
                per_cell={f"{b}|{p}|{bb}": n for (b, p, bb), n in sorted(cover.items())}, missing_cells=missing)
    if which in ("ext", "sb"):
        meta.update(sql_analysis(src, out, recs))
    (out / "SWAP_META.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "per_cell"}, indent=1))


def sql_analysis(src, out, recs):
    sys.path.insert(0, str(REPO / "data/records/protocol_all_matching"))
    sys.path.insert(0, str(REPO / "code"))
    sys.path.insert(0, str(REPO / "experiments/02_controlled_sharing/code"))
    import ext_sql_verdicts as XV
    import sql_common as SQ
    import sql_tasks as ST
    gold = {f"bird|{q['question_id']}": q["SQL"] for q in json.loads((pathlib.Path(SQ.BIRD) / "dev.json").read_text())}
    vblob = json.loads((src / "analysis/sql_verdicts.json").read_text())
    verd = vblob["verdicts"]
    sw = {(u, pr, "NOPEER", bb, rep) for (b, u, pr, bb, rep) in recs if b == "sql_bird_v2"}
    rows, todo = [], {}
    for (b, u, pr, bb, rep), r in recs.items():
        if b != "sql_bird_v2":
            continue
        db = ((r.get("config") or {}).get("unit_meta") or {}).get("db_id")
        ans = (r.get("record") or {}).get("majority_answer")
        pred = ST.unwrap_sql(ans) if ans not in (None, "", "None", "none") else ""
        rows.append(((u, pr, bb, rep), r, db, pred))
        if pred and db:
            k = XV.qkey(db, pred)
            if k not in verd:
                todo[k] = (db, pred, gold[u])
    with cf.ProcessPoolExecutor(max_workers=16) as ex:
        for k, v in ex.map(XV._one, list(todo.values()), chunksize=8):
            verd[k] = v
    n_old = 0
    with open(out / "analysis/sql_per_unit.jsonl", "w") as fh:
        for line in (src / "analysis/sql_per_unit.jsonl").open():
            x = json.loads(line)
            if (x["unit"], x["protocol"], x["arm"], x["backbone"], int(x["repeat"])) in sw:
                continue
            fh.write(line if line.endswith("\n") else line + "\n")
            n_old += 1
        cnt = collections.Counter()
        for (u, pr, bb, rep), r, db, pred in rows:
            if not pred:
                rec = dict(unit=u, protocol=pr, arm="NOPEER", backbone=bb, repeat=rep, outcome="declined",
                           decline_kind="truncated_only" if (r.get("truncated_calls") or 0) else "declined",
                           truncated_calls=r.get("truncated_calls") or 0, _clean=True)
            else:
                v = verd.get(XV.qkey(db, pred)) or {"outcome": "other_exec_error"}
                rec = dict(unit=u, protocol=pr, arm="NOPEER", backbone=bb, repeat=rep, outcome=v["outcome"],
                           truncated_calls=r.get("truncated_calls") or 0, db_id=db, _clean=True)
            cnt[rec["outcome"]] += 1
            fh.write(json.dumps(rec) + "\n")
    vblob["verdicts"] = verd
    (out / "analysis/sql_verdicts.json").write_text(json.dumps(vblob))
    return dict(per_unit_rows_kept=n_old, per_unit_rows_new=len(rows), new_verdicts=len(todo),
                new_outcomes=dict(cnt))


if __name__ == "__main__":
    main(sys.argv[1])
