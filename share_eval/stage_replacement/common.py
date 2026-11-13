r"""Shared pieces of the stage decomposition (offline; read-only SQLite; no model/API/GPU call).

Paths, the two read-only executors (one per scorer whose equality relation A_u must reproduce), the
execution-class key used for A_u, and the bootstraps.

Execution class of one output (A_u compares classes, never SQL text):
- ("FAIL",)            no output, error, timeout -> scored 0 whatever the gold is;
- ("OK", frozenset)    the scorer's own set of result tuples.
Two outputs in the same class get the same EX score against any gold, so A_u = 0 implies D_u = 0.
"""
from __future__ import annotations

import pathlib
import sqlite3
import sys
import threading
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRAWS, SEED = 10000, 20261105

# Paths, set by configure() from the command line (defaults follow the data layout of the repository
# README: run directories under data/records/, the BIRD and Spider databases under data/external/).
OUT = REPO / "output/share_eval/stage_replacement"
EXT = REPO / "data/records/macsql_comem_stages"
CHESS = REPO / "data/records/chess_unit_tester"
BIRD = REPO / "data/records/dataeng_raw/sql/dev_20240627"
BIRD_DB = BIRD / "dev_databases"
SPIDER = REPO / "data/records/dataeng_raw/spider"
SPIDER_DB = REPO / "data/external/spider_data/database"


def configure(description=""):
    """Read the input and output paths shared by the stage-replacement scripts."""
    import argparse
    import os
    global OUT, EXT, CHESS, BIRD, BIRD_DB, SPIDER, SPIDER_DB
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=str(OUT), help="output directory (default %(default)s)")
    ap.add_argument("--external-run", default=str(EXT),
                    help="MAC-SQL and ComEM run directory with output/, data/llm4em/, raw/ and logs/ (default %(default)s)")
    ap.add_argument("--chess-run", default=str(CHESS),
                    help="CHESS run directory with output/, data/, raw/ and logs/ (default %(default)s)")
    ap.add_argument("--bird", default=str(BIRD), help="BIRD development release with dev.json (default %(default)s)")
    ap.add_argument("--bird-db", default=os.environ.get("BIRD_DB_DIR") or str(BIRD_DB),
                    help="BIRD dev_databases directory (default $BIRD_DB_DIR or <bird>/dev_databases)")
    ap.add_argument("--spider", default=str(SPIDER), help="Spider directory with dev.json (default %(default)s)")
    ap.add_argument("--spider-db", default=os.environ.get("SPIDER_DB_DIR") or str(SPIDER_DB),
                    help="Spider database directory (default $SPIDER_DB_DIR or %(default)s)")
    a = ap.parse_args()
    OUT, EXT, CHESS = pathlib.Path(a.out), pathlib.Path(a.external_run), pathlib.Path(a.chess_run)
    BIRD, BIRD_DB = pathlib.Path(a.bird), pathlib.Path(a.bird_db)
    SPIDER, SPIDER_DB = pathlib.Path(a.spider), pathlib.Path(a.spider_db)
    OUT.mkdir(parents=True, exist_ok=True)
    return a


def _draws_for(n_clusters, n_draws, seed):
    return np.random.default_rng(seed).integers(0, n_clusters, size=(n_draws, n_clusters))


def _core_cluster_boot(values, clusters, weighting, n_draws, seed):
    """Paired cluster bootstrap: per-cluster mean, then cluster-equal or unit weights."""
    import collections
    units = sorted(values)
    by = collections.defaultdict(list)
    for u in units:
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl]); n = np.array([len(by[c]) for c in cl], float)
    ww = np.ones(len(cl)) if weighting == "cluster_equal" else n
    pt = float((ww * per).sum() / ww.sum())
    d = _draws_for(len(cl), n_draws, seed)
    bs = (per[d] * ww[d]).sum(1) / ww[d].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return dict(point=pt, lo=float(lo), hi=float(hi), p_boot=p, units=len(units), clusters=len(cl))


def coerce(v):
    """sql_common._coerce."""
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


# ---------------------------------------------------------------- MAC-SQL scorer (extract_macsql_formal.execute)
def macsql_execute(dbroot, db, sql, cap=10000, limit=30.0):
    """Identical to the formal MAC-SQL scorer: mode=ro, query_only, decode-ignore text, fetchmany(10,000), 30 s."""
    if not sql or not isinstance(sql, str):
        return dict(status="none", rows=None, seconds=0.0)
    con = sqlite3.connect(f"file:{dbroot}/{db}/{db}.sqlite?mode=ro", uri=True, timeout=limit)
    con.execute("PRAGMA query_only=ON")
    con.text_factory = lambda b: b.decode(errors="ignore")
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > limit else 0, 1000)
    try:
        rows = con.execute(sql).fetchmany(cap)
        return dict(status="ok", rows=rows, seconds=round(time.time() - t0, 4))
    except Exception as e:                                                 # noqa: BLE001
        return dict(status="timeout" if "interrupted" in str(e).lower() else "error", rows=None,
                    error=str(e)[:200], seconds=round(time.time() - t0, 4))
    finally:
        con.close()


def macsql_class(ex):
    """sql_common._norm_result on an ok execution; every non-ok status scores 0."""
    return ("OK", frozenset(tuple(coerce(x) for x in r) for r in ex["rows"])) if ex["status"] == "ok" else ("FAIL",)


# ---------------------------------------------------------------- CHESS scorer (extract_b1.execute_ro / score)
def chess_clean(sql):
    """extract_b1.clean_sql: CHESS compare_sqls cleans the predicted SQL before executing it."""
    return sql.replace("\n", " ").replace('"', "'").strip("`.")


def chess_execute(db, sql, timeout=60):
    """Identical to extract_b1.execute_ro: mode=ro, fetchall, 60 s thread timeout, default text factory."""
    box = {}

    def run():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
            try:
                box["rows"] = con.execute(sql).fetchall()
            finally:
                con.close()
        except Exception as e:                                             # noqa: BLE001
            box["error"] = f"{type(e).__name__}: {e}"
    t0 = time.time()
    t = threading.Thread(target=run, daemon=True)
    t.start(); t.join(timeout)
    if t.is_alive():
        return dict(status="timeout", rows=None, seconds=time.time() - t0)
    if "error" in box:
        return dict(status="error", rows=None, error=box["error"][:200], seconds=time.time() - t0)
    return dict(status="ok", rows=box["rows"], seconds=time.time() - t0)


def chess_class(ex):
    """CHESS EX relation: set(rows) == set(gold_rows) on an ok execution (score() in extract_b1)."""
    return ("OK", frozenset(ex["rows"])) if ex["status"] == "ok" else ("FAIL",)


def sqlcommon_class(ex, cap=10000):
    """sql_common semantics (execution_match: first 10,000 rows, _norm_result) on the same execution."""
    return ("OK", frozenset(tuple(coerce(x) for x in r) for r in ex["rows"][:cap])) if ex["status"] == "ok" else ("FAIL",)


# ---------------------------------------------------------------- bootstraps (percentile)
def cluster_boot(values, clusters):
    """Cluster bootstrap, question-equal (unit weights), percentile 95%."""
    if not values:
        return None
    return _core_cluster_boot(values, clusters, "unit", DRAWS, SEED)


def strat_boot(groups):
    """Stratified percentile bootstrap: resample units with replacement within each stratum (dataset fixed),
    pooled mean with unit weights. groups: {stratum: 1-d array}. Empty strata are skipped."""
    rng = np.random.default_rng(SEED)
    g = {k: np.asarray(v, float) for k, v in sorted(groups.items()) if len(v)}
    if not g:
        return None
    n = sum(len(v) for v in g.values())
    tot = np.zeros(DRAWS)
    for v in g.values():
        tot += v[rng.integers(0, len(v), size=(DRAWS, len(v)))].sum(1)
    bs = tot / n
    pt = float(sum(v.sum() for v in g.values()) / n)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(point=pt, lo=float(lo), hi=float(hi), units=n, clusters=len(g))


def strat_cluster_boot(groups):
    """Stratified cluster bootstrap for pair-level ComEM: groups = {dataset: [per-anchor arrays of pair values]};
    anchors resampled within dataset; pooled pair mean."""
    rng = np.random.default_rng(SEED)
    tot = np.zeros(DRAWS); cnt = np.zeros(DRAWS); s = c = 0.0
    for k in sorted(groups):
        A = [np.asarray(a, float) for a in groups[k] if len(a)]
        if not A:
            continue
        sums = np.array([a.sum() for a in A]); lens = np.array([len(a) for a in A], float)
        d = rng.integers(0, len(A), size=(DRAWS, len(A)))
        tot += sums[d].sum(1); cnt += lens[d].sum(1); s += sums.sum(); c += lens.sum()
    if not c:
        return None
    bs = tot / cnt
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(point=float(s / c), lo=float(lo), hi=float(hi), units=int(c), clusters=None)
