r"""Server-side G1 gate for one backbone (no execution, json only). gates:
truncation <= 2% per (protocol, arm, round); no <think> in replies; logprobs present on MAD-M² and
BCCS calls; NOPEER traces render 0 peers after round 0; no errored unit-run.

    python sb_g1_gate.py <backbone> <max_tokens>     -> exit 0 pass / 1 fail; writes G1_<bb>.json
The execution-based round-0 diagnostics are produced locally by sb_g1_check.py.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB                                                 # noqa: E402


def main():
    bb, mt = sys.argv[1], int(sys.argv[2])
    n, tr = collections.Counter(), collections.Counter()
    think = nolp = peer_text = err = 0
    runs = 0
    for p in sorted((SB.OUT / "raw").glob(f"sql_bird_v2__*__{bb.replace('.', '')}__r*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            r = json.loads(line)
            if r.get("backbone") != bb or r.get("max_tokens") != mt:
                continue
            if r.get("status") != "ok":
                err += 1
                continue
            runs += 1
            calls = r["record"]["calls"]
            rd = {c.get("seq"): int(c.get("round") or 0) for c in calls}
            for c in calls:
                k = (r["protocol"], r["arm"], int(c.get("round") or 0))
                n[k] += 1
                tr[k] += c.get("finish_reason") == "length"
                raw = c.get("raw_reply") or ""
                think += ("<think>" in raw or "</think>" in raw)
                if r["protocol"] in ("madm2_objective", "bccs") and not (c.get("n_logprob_tokens") or 0):
                    nolp += 1
            if r["arm"] == "NOPEER":
                for t in r.get("arm_traces") or []:
                    if rd.get(t.get("call_seq"), 0) > 0 and (t.get("n_peers_rendered") or t.get("peer_indices_kept")):
                        peer_text += 1
    cells = {"|".join(map(str, k)): round(tr[k] / n[k], 4) for k in sorted(n)}
    bad = [k for k, v in cells.items() if v > 0.02]
    res = dict(backbone=bb, max_tokens=mt, runs=runs, errors=err, think_hits=think, nologprob_calls=nolp,
               nopeer_peer_text=peer_text, trunc_share=cells, trunc_fail=bad)
    total_calls = sum(n.values())
    res["think_share"] = round(think / max(1, total_calls), 5)
    # A10: the thinking gate is a <= 2% share of calls, the same threshold as GLM's peer-text gate
    res["pass"] = bool(runs) and not (bad or nolp or peer_text or err) and res["think_share"] <= 0.02
    (SB.OUT / f"G1_{bb}_mt{mt}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
