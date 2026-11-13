"""Shared logic: schema check, spec/record checks, pairing, populations, repeats, estimators.

Task adapters turn existing records into `Record`s; everything here is task-independent. The
checks only see what the spec and the records declare: they cannot detect undeclared state or
hidden dependencies.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
SCHEMA = ROOT / "spec" / "comparison_spec.schema.json"

ERROR, UNSUPPORTED, INFO = "error", "unsupported", "info"


# ------------------------------------------------------------------ findings
@dataclass
class Finding:
    code: str
    severity: str
    where: str
    message: str

    def as_dict(self):
        return dict(code=self.code, severity=self.severity, where=self.where, message=self.message)


class SpecError(Exception):
    def __init__(self, findings):
        self.findings = findings
        super().__init__("; ".join(f"{f.code}: {f.message}" for f in findings))


# ------------------------------------------------------------------ schema (subset of JSON Schema)
_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list, "null": type(None)}


def _type_ok(v, t):
    if isinstance(t, list):
        return any(_type_ok(v, x) for x in t)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def schema_check(obj, schema=None, path="$") -> List[Finding]:
    """type / required / enum / properties / items -- the keywords the spec schema uses."""
    schema = schema if schema is not None else json.loads(SCHEMA.read_text())
    out = []
    if "type" in schema and not _type_ok(obj, schema["type"]):
        return [Finding("SCHEMA", ERROR, path, f"expected {schema['type']}, got {type(obj).__name__}")]
    if "enum" in schema and obj not in schema["enum"]:
        out.append(Finding("SCHEMA", ERROR, path, f"{obj!r} not in {schema['enum']}"))
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                out.append(Finding("SCHEMA", ERROR, path, f"missing required field '{k}'"))
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                out += schema_check(obj[k], sub, f"{path}.{k}")
    if isinstance(obj, list) and "items" in schema:
        for i, x in enumerate(obj):
            out += schema_check(x, schema["items"], f"{path}[{i}]")
    return out


# ------------------------------------------------------------------ spec semantics
def check_spec(spec: Dict[str, Any]) -> List[Finding]:
    f = schema_check(spec)
    if any(x.severity == ERROR for x in f):
        return f
    steps = spec["steps"]
    pos = {s: i for i, s in enumerate(steps)}
    dp = spec["decision_point"]
    for k in ("after_step", "before_step"):
        if dp[k] not in pos:
            f.append(Finding("SCHEMA", ERROR, f"decision_point.{k}", f"unknown step {dp[k]!r}"))
    if f:
        return f
    boundary = pos[dp["after_step"]]
    outs = {o["id"]: o for o in spec["outputs"]}
    ops = {o["id"]: o for o in spec["operations"]}
    for o in spec["outputs"]:
        if o["produced_at_step"] not in pos:
            f.append(Finding("SCHEMA", ERROR, f"outputs.{o['id']}", f"unknown step {o['produced_at_step']!r}"))
    for op in spec["operations"]:
        w = f"operations.{op['id']}"
        if op["role"] == "oracle_reference" and op.get("executable", False):
            f.append(Finding("ORACLE_AS_POLICY", ERROR, w, "a test-best choice is declared executable"))
        if op["label_use"] == "runtime_input":
            f.append(Finding("GOLD_AT_RUNTIME", ERROR, w, "labels declared as a runtime input"))
        for cid in op["consumes"]:
            o = outs.get(cid)
            if o is None:
                f.append(Finding("SCHEMA", ERROR, w, f"consumes undeclared output {cid!r}"))
                continue
            if o.get("role") == "label":
                if o.get("split") == "test" and op["role"] != "oracle_reference":
                    f.append(Finding("GOLD_AT_RUNTIME", ERROR, w, f"consumes test labels {cid!r}"))
                elif o.get("split") == "dev" and op["label_use"] != "dev_selection":
                    f.append(Finding("GOLD_AT_RUNTIME", ERROR, w, f"consumes dev labels {cid!r} without declaring dev_selection"))
                continue
            if pos.get(o["produced_at_step"], -1) > boundary and op["role"] != "oracle_reference":
                f.append(Finding("POST_BOUNDARY_INPUT", ERROR, w,
                                 f"{cid!r} is produced at {o['produced_at_step']!r}, after the decision point {dp['after_step']!r}"))
            if op.get("replaces") and o["producer"] == op["replaces"]:
                f.append(Finding("REPLACED_PRODUCER_INPUT", ERROR, w,
                                 f"{cid!r} is produced by {op['replaces']!r}, the operation this one replaces"))
            if o["provenance"] == "unknown":
                f.append(Finding("UNSUPPORTED_PROVENANCE", UNSUPPORTED, w, f"{cid!r} has unknown provenance/timing"))
    if spec["comparison"].get("same_information_as_original"):
        orig = [o["id"] for o in spec["operations"] if o["role"] == "original"]
        for op in spec["operations"]:
            if op["role"] != "replacement":
                continue
            for cid in op["consumes"]:
                o = outs.get(cid)
                if o and o.get("role") != "label" and not set(orig) & set(o["visible_to"]):
                    f.append(Finding("INFORMATION_NOT_VISIBLE_TO_ORIGINAL", ERROR, f"operations.{op['id']}",
                                     f"{cid!r} was not visible to the original operation {orig}, but the comparison declares "
                                     "that only the operation changes"))
    for o in spec["outputs"]:
        if o.get("role") == "label" and o.get("split") == "test":
            vis = [v for v in o["visible_to"] if ops.get(v, {}).get("role") != "oracle_reference"]
            if vis:
                f.append(Finding("GOLD_AT_RUNTIME", ERROR, f"outputs.{o['id']}", f"test labels declared visible to {vis}"))
    lab = spec["labels"]
    if lab["runtime_input"]:
        f.append(Finding("GOLD_AT_RUNTIME", ERROR, "labels", "labels.runtime_input is true"))
    if lab["test_best"] == "executable":
        f.append(Finding("ORACLE_AS_POLICY", ERROR, "labels", "test-best declared executable"))
    ds = spec["downstream"]
    if ds["affects_control_flow"] and ds["reuse_old_suffix"]:
        f.append(Finding("STALE_SUFFIX", ERROR, "downstream",
                         "the replaced output reaches control flow, so the old suffix cannot be reused"))
    m = spec["metric"]
    if re.search(r"f1", m["name"], re.I) and m["aggregator"] != "relation_counts_f1":
        f.append(Finding("F1_ROW_MEAN", ERROR, "metric", "F1 must be computed from relation counts, not a row mean"))
    for c in spec["costs"]["components"]:
        if c["shared_prefix"] and c["added_to_arms"]:
            f.append(Finding("DOUBLE_COUNT_PREFIX", ERROR, f"costs.{c['id']}",
                             "a shared prefix is added to every arm's incremental cost"))
    for c in spec["comparison"]["contrasts"]:
        for x in c[:2]:
            if x is not None and x not in ops and x not in spec.get("analysis", {}).get("derived_ops", []):
                f.append(Finding("SCHEMA", ERROR, "comparison.contrasts", f"unknown operation {x!r}"))
    for cl in spec.get("claims", []):
        for oid in cl.get("needs_provenance_of", []):
            if outs.get(oid, {}).get("provenance") == "unknown":
                f.append(Finding("UNSUPPORTED_PROVENANCE", UNSUPPORTED, f"claims.{cl['id']}",
                                 f"claim needs the timing of {oid!r}, whose provenance is unknown"))
    pops = spec["populations"]
    for e in spec.get("analysis", {}).get("estimates", []):
        if e.get("kind") == "cross_pool_change":
            p = pops.get(e["population"], {})
            if sorted(p.get("common_to", [])) != sorted(e["pools"]):
                f.append(Finding("POPULATION_MISMATCH", ERROR, f"analysis.{e['id']}",
                                 f"population {e['population']!r} is not common to pools {e['pools']}"))
    return f


# ------------------------------------------------------------------ records
@dataclass
class Record:
    op: str
    pool: str
    unit: Tuple
    cluster: str
    repeat: Optional[int]
    value: Optional[float] = None            # unit-mean metrics
    counts: Optional[Dict[str, int]] = None  # relation-count metrics
    status: str = "ok"                       # ok | failed | missing | unresolved
    state: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Dict[str, str]] = field(default_factory=list)   # [{output, produced_at_step, producer}]
    cost: Dict[str, float] = field(default_factory=dict)


def check_records(spec, records: Sequence[Record], populations: Dict[str, Dict[str, set]]) -> List[Finding]:
    """populations: name -> pool -> set(unit)."""
    f = []
    pos = {s: i for i, s in enumerate(spec["steps"])}
    boundary = pos[spec["decision_point"]["after_step"]]
    ops = {o["id"]: o for o in spec["operations"]}
    seen = collections.Counter((r.op, r.pool, r.unit, r.repeat) for r in records)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        f.append(Finding("DUPLICATE_KEY", ERROR, "records", f"{len(dups)} duplicated (op, pool, unit, repeat) keys, e.g. {dups[0]}"))
    clus = collections.defaultdict(set)
    for r in records:
        clus[r.unit].add(r.cluster)
    bad = [u for u, c in clus.items() if len(c) > 1]
    if bad:
        f.append(Finding("CLUSTER_MISMATCH", ERROR, "records", f"{len(bad)} units carry different clusters across arms/pools, e.g. {bad[0]}"))
    need = spec["decision_point"]["state_fields"]
    miss_state = [r for r in records if ops.get(r.op, {}).get("role") != "oracle_reference"
                  and any(k not in r.state or r.state[k] is None for k in need)]
    if miss_state:
        r = miss_state[0]
        f.append(Finding("MISSING_STATE", ERROR, "records",
                         f"{len(miss_state)} records lack required state {[k for k in need if k not in r.state or r.state[k] is None]}, e.g. {r.op} {r.unit}"))
    for r in records:
        op = ops.get(r.op)
        if op is None:
            continue
        for inp in r.inputs:
            if pos.get(inp.get("produced_at_step"), -1) > boundary:
                f.append(Finding("POST_BOUNDARY_INPUT", ERROR, f"records.{r.op}",
                                 f"{r.unit}: input {inp.get('output')!r} produced at {inp.get('produced_at_step')!r}"))
                break
            if op.get("replaces") and inp.get("producer") == op["replaces"]:
                f.append(Finding("REPLACED_PRODUCER_INPUT", ERROR, f"records.{r.op}",
                                 f"{r.unit}: input {inp.get('output')!r} comes from the replaced producer {op['replaces']!r}"))
                break
    # repeats
    pol = spec["unit"]["repeat"]["policy"]
    nrep = collections.defaultdict(set)
    for r in records:
        nrep[(r.op, r.pool, r.unit)].add(r.repeat)
    per_op = collections.defaultdict(set)
    for (o, p, u), s in nrep.items():
        per_op[(o, p)].add(len(s))
    unbalanced = {k: v for k, v in per_op.items() if len(v) > 1}
    if unbalanced and pol == "pool_rows":
        f.append(Finding("UNBALANCED_REPEAT", ERROR, "unit.repeat",
                         f"repeat counts differ across units for {sorted(unbalanced)[:2]} while rows are pooled"))
    elif unbalanced:
        f.append(Finding("UNBALANCED_REPEAT", INFO, "unit.repeat",
                         f"repeat counts differ across units for {len(unbalanced)} (op, pool); handled by mean within unit"))
    # population coverage; nothing is dropped silently
    have = collections.defaultdict(set)
    for r in records:
        if r.status != "missing":
            have[(r.op, r.pool)].add(r.unit)
    mp = spec["missing_policy"]
    for name, by_pool in populations.items():
        for pool, units in by_pool.items():
            for o in spec.get("analysis", {}).get("ops_by_pool", {}).get(pool, []):
                lack = units - have[(o, pool)]
                if lack:
                    sev = ERROR if mp["missing"] == "error" else INFO
                    f.append(Finding("MISSING_UNIT", sev, f"population.{name}.{pool}.{o}",
                                     f"{len(lack)} of {len(units)} units absent, e.g. {sorted(lack)[0]}"))
    failed = [r for r in records if r.status in ("failed", "unresolved")]
    if failed and mp["failed"] == "error":
        f.append(Finding("FAILED_RECORD", ERROR, "records", f"{len(failed)} failed/unresolved records and policy is error"))
    return f


def raise_on_errors(findings):
    errs = [x for x in findings if x.severity == ERROR]
    if errs:
        raise SpecError(errs)


# ------------------------------------------------------------------ estimators
def unit_values(records, op, pool, repeats=None, missing="error"):
    """unit -> mean over the selected repeats of the record values (failed kept as 0)."""
    acc = collections.defaultdict(list)
    for r in records:
        if r.op != op or r.pool != pool:
            continue
        if r.value is None and r.status not in ("failed", "unresolved", "missing"):
            raise SpecError([Finding("METRIC_UNSUPPORTED", ERROR, f"records.{op}",
                                     "record has no unit-level value (relation-count metric?); a unit mean is not defined")])
        if repeats is not None and r.repeat is not None and r.repeat not in repeats:
            continue
        if r.status == "missing":
            continue
        v = 0.0 if r.status in ("failed", "unresolved") else float(r.value)
        acc[r.unit].append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def cluster_of(records):
    return {r.unit: r.cluster for r in records}


def draws_for(clusters_sorted, n_draws, seed):
    return np.random.default_rng(seed).integers(0, len(clusters_sorted), size=(n_draws, len(clusters_sorted)))


def cluster_boot(values: Dict[Tuple, float], clusters: Dict[Tuple, str], weighting: str,
                 n_draws: int, seed: int, units=None):
    """Paired cluster bootstrap: per-cluster mean, then cluster-equal or unit weights."""
    units = sorted(values) if units is None else sorted(units)
    by = collections.defaultdict(list)
    for u in units:
        by[clusters[u]].append(values[u])
    cl = sorted(by)
    per = np.array([np.mean(by[c]) for c in cl]); n = np.array([len(by[c]) for c in cl], float)
    ww = np.ones(len(cl)) if weighting == "cluster_equal" else n
    pt = float((ww * per).sum() / ww.sum())
    d = draws_for(cl, n_draws, seed)
    bs = (per[d] * ww[d]).sum(1) / ww[d].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return dict(point=pt, lo=float(lo), hi=float(hi), p_boot=p, units=len(units), clusters=len(cl))


def f1_from_counts(T: int, N: int, G: int) -> float:
    """Pairwise F1 = 2T/(N+G): T true intra pairs, N predicted intra pairs, G gold pairs."""
    return (2 * T / (N + G)) if (N + G) else 0.0


def dev_select(dev_scores: Dict[str, float], order: Sequence[str]) -> Tuple[str, List[str]]:
    """Highest development score; ties resolved by the fixed order. Labels: development only."""
    top = max(dev_scores.values())
    tied = sorted([k for k, v in dev_scores.items() if v == top], key=order.index)
    return tied[0], tied


def transitions(base: Dict, trt: Dict, clusters: Dict, weighting="unit"):
    """Binary paired transition decomposition: Recovered = (1-q)c, Lost = qr, Net (pp)."""
    keys = sorted(set(base) & set(trt))
    b = np.array([base[k] for k in keys]); t = np.array([trt[k] for k in keys])
    q = b.mean(); c = t[b == 0].mean() if (b == 0).any() else float("nan")
    r = (1 - t[b == 1]).mean() if (b == 1).any() else float("nan")
    return dict(n=len(keys), q_pct=100 * q, c_pct=100 * c, r_pct=100 * r,
                recovered_pp=100 * (1 - q) * c, lost_pp=100 * q * r, net_pp=100 * ((1 - q) * c - q * r))


# ------------------------------------------------------------------ provenance of inputs
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def hash_inputs(spec) -> List[Dict[str, str]]:
    out = []
    for i in spec["inputs"]:
        p = REPO / i["path"]
        out.append(dict(path=i["path"], role=i["role"], sha256=sha256(p) if p.is_file() else "MISSING",
                        bytes=p.stat().st_size if p.is_file() else None))
    return out


def run_estimates(spec, records, populations, restrict=None):
    """Generic level / contrast / cross-pool estimates over unit-mean records.

    Record.unit is the pairing key without the pool; a cross-pool quantity is formed per unit
    before resampling, so both pools, both arms and all repeats of a unit stay together."""
    bs = spec.get("bootstrap", {"draws": 10000, "seed": 20261105})
    cl = cluster_of(records)
    wt = spec["unit"]["weighting"]
    out = {}
    for e in spec["analysis"]["estimates"]:
        if restrict and e["id"] not in restrict:
            continue
        reps = set(e["repeats"]) if e.get("repeats") else None

        def y(op, pool):
            return unit_values(records, op, pool, reps)
        if e["kind"] == "cross_pool_change":
            new, old = e["pools"]
            units = populations[e["population"]][old] & populations[e["population"]][new]
            ya_n, ya_o = y(e["a"], new), y(e["a"], old)
            yb_n, yb_o = (y(e["b"], new), y(e["b"], old)) if e.get("b") else ({}, {})
            val = lambda u: (ya_n[u] - (yb_n[u] if e.get("b") else 0)) - (ya_o[u] - (yb_o[u] if e.get("b") else 0))
        else:
            pool = e["pool"]
            units = set(populations[e["population"]][pool])
            ya = y(e["a"], pool)
            yb = y(e["b"], pool) if e.get("b") else None
            val = (lambda u: ya[u] - yb[u]) if yb is not None else (lambda u: ya[u])
        if e.get("benchmark"):
            units = {u for u in units if u[0] == e["benchmark"]}
        try:
            vals = {u: val(u) for u in units}
        except KeyError as k:
            raise SpecError([Finding("MISSING_UNIT", ERROR, f"analysis.{e['id']}", f"no outcome for unit {k}")])
        r = cluster_boot(vals, cl, wt, bs["draws"], bs["seed"])
        out[e["id"]] = {k: (100 * v if k in ("point", "lo", "hi") else v) for k, v in r.items()}
        out[e["id"]]["estimate"] = e
    return out


def compare_target(t, got: Dict[str, float]):
    """Locked target vs recomputed estimate."""
    res = dict(id=t["id"], quantity=t["quantity"], source=t["source"])
    if isinstance(t.get("value"), list):
        res.update(target_value=t["value"], got_value=got.get("value"), tolerance=0,
                   reproduced=list(got.get("value") or []) == t["value"])
        return res
    ok = True
    for k in ("point", "lo", "hi", "value"):
        if k in t:
            tol = t.get(f"tol_{k}", t.get("tol", 0.01))
            v = got.get(k)
            res[f"target_{k}"] = t[k]
            res[f"got_{k}"] = None if v is None else round(v, 4)
            ok = ok and v is not None and abs(v - t[k]) <= tol + 1e-9
    res["tolerance"] = {k: t.get(f"tol_{k}", t.get("tol", 0.01)) for k in ("point", "lo", "hi", "value") if k in t}
    res["reproduced"] = ok
    return res
