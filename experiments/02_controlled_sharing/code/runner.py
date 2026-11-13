#!/usr/bin/env python3
"""Dynamic shard dispatcher for the three-domain, all-workflow collection.

    python3 runner.py --phase a                 # dispatch until the phase is done
    python3 runner.py --status                  # what is running, what remains
    python3 runner.py --pause "reason"          # stop new dispatch, let in-flight finish
    python3 runner.py --resume
    python3 runner.py --release-gpu 0 "reason"  # stop dispatching to that card's aliases
    python3 runner.py --readmit-gpu 0
    python3 runner.py --stop                    # stop dispatch and signal in-flight shards
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NR = ROOT / "results/protocol_all_matching_20261018T192619Z"
LOGS, RAW = NR / "logs", NR / "raw"
GATE = NR / "GATE.json"
DISPATCH_PID = LOGS / "dispatcher.pid"
HOST = os.environ.get("REMOTE_SSH_TARGET", "user@inference-host")


def _on_gpu_host() -> bool:
    """True when this process already runs on the machine that holds the cards.
    """
    global _ON_HOST
    if _ON_HOST is None:
        try:
            _ON_HOST = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                      timeout=20).returncode == 0
        except Exception:
            _ON_HOST = False
    return _ON_HOST


_ON_HOST = None


def remote_sh(cmd: str, timeout: int):
    """Run a shell command on the GPU host: locally when we are already there, else over ssh."""
    argv = ["sh", "-c", cmd] if _on_gpu_host() else ["ssh", "-o", "BatchMode=yes", HOST, cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout

#: alias -> (gpu, port). GPU 0 is the shared card.
ALIAS_GPU = {"llama3-8b-g1": (1, 9211), "llama3-8b-g3": (3, 9213),
             "qwen25-7b-g2": (2, 9212), "qwen25-7b-g0": (0, 9210),
             "llama3-8b-g5": (5, 9215), "qwen25-7b-g5": (5, 9216),
             "qwen25-7b-g1b": (1, 9221), "llama3-8b-g2b": (2, 9222),
             "qwen25-7b-g3b": (3, 9223),
             "gemma4-g5": (2, 9224)}
SHARED_GPUS = {0}
#: our own serving processes per GPU, so another tenant's growth is distinguishable from ours
OUR_CONTAINERS = {0: "pxt-qwen25-7b-g0", 1: "pxt-llama3-8b-g1",
                  2: "pxt-qwen25-7b-g2", 3: "pxt-llama3-8b-g3",
                  5: "pxt-llama3-8b-g5 pxt-qwen25-7b-g5"}
#: every card except the shared GPU 0 now serves both backbones from two servers
SECOND_SERVER = {1: "pxt-qwen25-7b-g1b", 2: "pxt-llama3-8b-g2b", 3: "pxt-qwen25-7b-g3b"}
#: GPU 4 is deliberately absent: it is not authorised for this run.
MIN_FREE_MIB = {0: 8000, 1: 4000, 2: 4000, 3: 4000, 5: 4000}
#: Shards one card may carry at once. Without a cap the pool fills with whatever sorted first and a
#: card whose shards are all still pending never gets a slot, which is how GPU 3 sat at 0%.
MAX_PER_GPU = 8

#: Client concurrency per shard, by how SEQUENTIAL the workflow is.
WORKERS = {"dmad": 48, "dov_vote": 16, "bccs": 64, "liang_mad": 32,
           "reconcile": 32, "moa": 32, "madm2_naive": 32, "madm2_objective": 32,
           "du_mad": 32, "ford": 48}


def routed():
    """alias -> (gpu, port) AFTER ROUTES.json. Using the static table instead blocked three aliases
as dead while they were re-pointed onto live ports: llama3-8b-g3, llama3-8b-g5 and
qwen25-7b-g5 had moved to 9211/9212 when their own cards were handed back."""
    rp = {}
    f = HERE / "ROUTES.json"
    if f.exists():
        try:
            rp = {k: int(v) for k, v in json.loads(f.read_text(encoding="utf-8")).items()}
        except Exception:
            rp = {}
    port_gpu = {p: g for _a, (g, p) in ALIAS_GPU.items()}
    out = {}
    for a, (g, p) in ALIAS_GPU.items():
        np_ = rp.get(a, p)
        out[a] = (port_gpu.get(np_, g), np_)
    return out
#: GPU 0 was rebuilt with max-num-seqs 384 and a 1,107,952-token KV cache, against 96 and 313,712
#: elsewhere, so it can hold more shards.
MAX_PER_GPU_OVERRIDE = {0: 10}
#: Replicas of one checkpoint. Which replica serves a cell is a serving-location choice, not a
#: design factor: the backbone is the factor.
REPLICAS = {"llama3-8b": ["llama3-8b-g1", "llama3-8b-g3", "llama3-8b-g5", "llama3-8b-g2b"],
            "qwen2.5-7b": ["qwen25-7b-g2", "qwen25-7b-g0", "qwen25-7b-g5",
                           "qwen25-7b-g1b", "qwen25-7b-g3b"]}
#: FORD is one Llama agent and one Qwen agent. Its default roster names one replica of each, so
#: every FORD shard hit the same two endpoints and left three cards idle.
DEFAULT_HETERO_ROSTER = {"ford": ("llama3-8b-g1", "qwen25-7b-g2"),
                         "moa": ("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5"),
                         "reconcile": ("llama3-8b-g1", "qwen25-7b-g2", "gemma4-g5")}
HETERO_ROSTER = {"ford": {0: ("llama3-8b-g1", "qwen25-7b-g2"),
                          1: ("llama3-8b-g3", "qwen25-7b-g3b"),
                          2: ("llama3-8b-g5", "qwen25-7b-g0")}}

POPS = {"wdc_products": 2615, "geographic_settlements": 4719, "musicbrainz20k": 32199}
#: Strict cheapest-first, by measured output-token volume over the three populations: ford 25M,
#: du_mad 131M, liang_mad 354M, madm2_objective 393M, madm2_naive 409M, bccs 803M, dov_vote 872M,
#: dmad 2,133M.
PROTO_ORDER = ["ford", "moa", "reconcile", "du_mad", "liang_mad", "madm2_objective",
               "madm2_naive", "bccs", "dov_vote", "dmad"]

#: The four configurations the original cross-task goal names: BCCS, DMAD and both MAD-M2 variants.
TARGET_FIRST = {"bccs", "dmad", "madm2_naive", "madm2_objective"}
TARGET_FIRST_POP = "wdc_products"

#: Populations are finished in size order, and MusicBrainz waits.
POP_LANE = {"wdc_products": 0, "geographic_settlements": 1, "musicbrainz20k": 2}
HETERO = {"ford", "moa", "reconcile"}
REP_ALIAS = {0: ("llama3-8b-g1", "qwen25-7b-g2"), 1: ("llama3-8b-g3", "qwen25-7b-g0"),
             2: ("llama3-8b-g1", "qwen25-7b-g1b")}


# ------------------------------------------------------------------------------- gate
def gate_read():
    if GATE.exists():
        return json.loads(GATE.read_text(encoding="utf-8"))
    return {"paused": False, "reason": "", "blocked_gpus": [], "updated": ""}


def gate_write(g):
    g["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    GATE.parent.mkdir(parents=True, exist_ok=True)
    GATE.write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
    return g


# ---------------------------------------------------------------------- resource check
def gpu_state():
    """index -> {free_mib, util, our_mib, other_mib}. Ownership comes from the container's pid,
not from utilisation: another tenant at 0% still owns its memory."""
    q = ("nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu "
         "--format=csv,noheader,nounits; echo ---; "
         "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; "
         "echo ---; nvidia-smi --query-gpu=index,uuid --format=csv,noheader; echo ---; "
         # Ownership by cgroup container id, not by the container's init pid: nvidia-smi reports
         # vLLM's EngineCore child, whose pid never equals `docker inspect .State.Pid`, and the
         # cgroup line carries the 64-hex container id rather than the name.
         "cmap=$(mktemp); "
         "docker ps --no-trunc --format '{{.ID}} {{.Names}}' > \"$cmap\"; "
         "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
         "cid=$(grep -oE '[0-9a-f]{64}' /proc/$p/cgroup 2>/dev/null | head -1); "
         # An EMPTY cid must not match.
         "if [ -n \"$cid\" ]; then "
         "echo $p $(grep -F \"$cid\" \"$cmap\" 2>/dev/null | awk '{print $2}' | head -1); "
         "else echo $p __NOT_A_CONTAINER__; fi"
         "; done")
    try:
        out = remote_sh(q, 45)
    except Exception:
        return {}
    parts = out.split("---")
    if len(parts) < 4:
        return {}
    st = {}
    for ln in parts[0].strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 5:
            st[int(f[0])] = {"total": int(f[1]), "used": int(f[2]), "free": int(f[3]),
                             "util": int(f[4]), "our_mib": 0, "other_mib": 0}
    uuid_gpu = {}
    for ln in parts[2].strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) == 2:
            uuid_gpu[f[1]] = int(f[0])
    ours = set()
    for ln in parts[3].strip().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and f[0].isdigit() and "pxt-" in f[1]:
            ours.add(int(f[0]))
    for ln in parts[1].strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) != 3:
            continue
        g = uuid_gpu.get(f[0])
        if g is None or g not in st:
            continue
        mib = int(re.sub(r"[^0-9]", "", f[2]) or 0)
        pid = int(f[1]) if f[1].isdigit() else -1
        st[g]["our_mib" if pid in ours else "other_mib"] += mib
    return st


#: liveness cache, refreshed once per dispatch cycle. One batched
#: probe per cycle, rather than one per `alias_ok` call, gives the same guarantee at a fixed cost.
_LIVE = {"at": 0.0, "ports": {}}


def refresh_liveness(ports):
    cmd = "; ".join(f"echo {p} $(curl -s -m 4 http://localhost:{p}/v1/models | head -c 8)"
                    for p in sorted(set(ports)))
    live = {}
    try:
        out = remote_sh(cmd, 60)
        for ln in out.splitlines():
            f = ln.split()
            if f and f[0].isdigit():
                live[int(f[0])] = len(f) > 1
    except Exception:
        # an unreadable probe must not block dispatch; treat as live and let a real failure surface
        live = {p: True for p in ports}
    _LIVE["ports"] = live
    _LIVE["at"] = time.time()
    return live


def endpoint_live(port):
    """Whether something answers on that port, from the per-cycle cache. A roster member that is not
served blocks its workflow rather than letting every call fail: gemma4 was stopped with GPU 5,
and MoA and ReConcile would otherwise have spent slots collecting transport errors."""
    return _LIVE["ports"].get(port, True)


def alias_ok(alias, st, g):
    """Why this alias may not be dispatched to, or `""` when it may."""
    gpu, _port = routed().get(alias, ALIAS_GPU[alias])
    if not endpoint_live(_port):
        return f"{alias} is not served on port {_port}"
    if gpu in (g.get("blocked_gpus") or []):
        return f"GPU{gpu} released on request"
    s = st.get(gpu)
    if not s:
        return "GPU state unreadable"
    if s["free"] < MIN_FREE_MIB.get(gpu, 4000):
        return f"GPU{gpu} only {s['free']} MiB free"
    if gpu in SHARED_GPUS and s["other_mib"] > 0:
        # another tenant holds memory here; keep the headroom we promised and stay out if it grew
        budget = s["total"] - s["other_mib"] - MIN_FREE_MIB[gpu]
        if s["our_mib"] > budget:
            return (f"GPU{gpu} shared: other tenant holds {s['other_mib']} MiB, "
                    f"our {s['our_mib']} MiB exceeds the {budget} MiB budget")
    return ""


# -------------------------------------------------------------------------- work list
def shard_list(phase):
    """Bind each cell to a replica. A cell with units already on disk keeps the replica that wrote
them; only an untouched cell may be moved onto a newly available card, so no unit is ever
collected under two paths."""
    out = []
    spread = 0
    for pop in POPS:
        for proto in PROTO_ORDER:
            if proto in HETERO:
                out += [(pop, proto, "team", r) for r in (0, 1, 2)]
                continue
            for r in (0, 1, 2):
                for bb in ("llama3-8b", "qwen2.5-7b"):
                    reps = REPLICAS[bb]
                    started = [a for a in reps
                               if units_done(out_path((pop, proto, a, r))) > 0]
                    if started:
                        out += [(pop, proto, a, r) for a in started]
                    else:
                        # A (backbone, repeat) slot with NO file anywhere was invisible to the
                        # dispatcher: it enumerated work from the files that existed, so a slot that
                        # never got dispatched stayed never-dispatched and its leg sat at 5/6 for
                        # ever.
                        out.append((pop, proto, reps[spread % len(reps)], r))
                        spread += 1
    # interleave populations so a long musicbrainz shard is always paired with a short one
    out.sort(key=lambda s: (PROTO_ORDER.index(s[1]), POPS[s[0]], s[3]))
    return out


def out_path(s):
    pop, proto, al, rep = s
    return RAW / f"{pop}__{proto}__{al}__r{rep}.jsonl"


def units_done(p):
    """OK records only. Counting error records as done made a shard that was writing nothing but transport
failures look finished, so the dispatcher stopped re-dispatching it: that is why DMAD appeared to
stall at 17.7% while 73.5% of its records were failures."""
    if not p.exists():
        return 0
    n = 0
    with p.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                if json.loads(ln).get("status") == "ok":
                    n += 1
            except Exception:
                pass
    return n


def units_retired(p):
    """Units the collector gave up on, with their cause, from the sidecar it writes.
    """
    f = p.with_suffix(p.suffix + ".retired.json")
    if not f.exists():
        return 0
    try:
        return len(json.loads(f.read_text(encoding="utf-8")).get("retired") or {})
    except Exception:
        return 0


def live_paths():
    """Output paths currently owned by a live collector, so one path has one writer."""
    try:
        ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    except Exception:
        return {}
    owned = {}
    for ln in ps.splitlines():
        if "collect_all_matching.py" not in ln:
            continue
        # Match on TOKENS, not substrings. That is how GPU 3 sat idle.
        try:
            toks = shlex.split(ln.strip())
        except ValueError:
            continue
        if len(toks) < 3 or not toks[0].isdigit():
            continue
        if not any(t.endswith("collect_all_matching.py") for t in toks):
            continue
        if "--out" not in toks:
            continue
        i = toks.index("--out")
        if i + 1 >= len(toks):
            continue
        # The aliases a live shard ACTUALLY uses, read from its own argv.
        als = []
        if "--alias" in toks:
            als = [toks[toks.index("--alias") + 1]]
        elif "--roster" in toks:
            als = [x for x in toks[toks.index("--roster") + 1].split(",") if x]
        else:
            proto = toks[toks.index("--protocol") + 1] if "--protocol" in toks else ""
            als = list(DEFAULT_HETERO_ROSTER.get(proto, ()))
        owned[toks[i + 1]] = {"pid": int(toks[0]), "aliases": als}
    return owned


# ------------------------------------------------------------------------------ actions
def do_status():
    g, st = gate_read(), gpu_state()
    owned = live_paths()
    shards = shard_list("a")
    done = {s: units_done(out_path(s)) for s in shards}
    tot_done = sum(done.values())
    tot_exp = sum(POPS[s[0]] for s in shards)
    print(f"gate: {'PAUSED - ' + g['reason'] if g['paused'] else 'running'}"
          f"   blocked GPUs: {g.get('blocked_gpus') or 'none'}   checked {g.get('updated') or '-'}")
    print(f"dispatcher pid: {DISPATCH_PID.read_text().strip() if DISPATCH_PID.exists() else '-'}"
          f"   live collectors: {len(owned)}")
    print(f"\nGPU                free    util   ours  other  dispatch")
    for gpu in sorted(st):
        s = st[gpu]
        al = [a for a, (gg, _p) in ALIAS_GPU.items() if gg == gpu]
        why = alias_ok(al[0], st, g) if al else "no alias"
        print(f"  GPU{gpu}  {s['free']:>9} MiB {s['util']:>4}% {s['our_mib']:>6} "
              f"{s['other_mib']:>6}  {'ok' if not why else why}")
    print(f"\nprogress: {tot_done:,} / {tot_exp:,} units "
          f"({100*tot_done/max(tot_exp,1):.2f}%) over {len(shards)} shards")
    print(f"{'protocol':16s}{'population':24s}{'done':>9s}{'expected':>10s}{'state':>12s}")
    for s in shards:
        exp = POPS[s[0]]
        p = str(out_path(s))
        state = ("RUNNING" if p in owned else "complete" if done[s] >= exp
                 else "partial" if done[s] else "pending")
        if state != "complete":
            print(f"  {s[1]:14s}{s[0]:24s}{done[s]:9,d}{exp:10,d}{state:>12s}  {s[2]} r{s[3]}")
    return 0


def do_run(phase, pool, workers):
    if DISPATCH_PID.exists():
        old = DISPATCH_PID.read_text().strip()
        if old.isdigit() and pathlib.Path(f"/proc/{old}").exists() or _alive(old):
            print(f"REFUSING: dispatcher {old} is alive. Two dispatchers would share output paths.")
            return 1
    DISPATCH_PID.parent.mkdir(parents=True, exist_ok=True)
    DISPATCH_PID.write_text(str(os.getpid()) + "\n")
    shards = shard_list(phase)
    log = (LOGS / f"phase_{phase}.log").open("a", encoding="utf-8")
    t_run = t_wait = 0.0
    last = time.time()
    cycles_failed = 0
    while True:
      try:
        g, st = gate_read(), gpu_state()
        refresh_liveness([p for _a, (_g, p) in routed().items()])
        owned = live_paths()
        # A shard is pending when OK records plus retired units still fall short of the population.
        pending = [s for s in shards
                   if units_done(out_path(s)) + units_retired(out_path(s)) < POPS[s[0]]
                   and str(out_path(s)) not in owned]
        # `only` concentrates capacity on a subset without dropping anything: the excluded shards
        # stay pending and resume the moment the filter is cleared.
        only = g.get("only") or []
        if only:
            pending = [s for s in pending
                       if s[1] in only or f"{s[0]}:{s[1]}" in only]
        if not pending and not owned:
            log.write(f"=== PHASE {phase.upper()} COMPLETE "
                      f"{time.strftime('%H:%M', time.gmtime())} ===\n")
            log.flush()
            break
        free = pool - len(owned)
        started = 0
        # how many live shards each GPU already carries, so the next dispatch goes to the
        # emptiest card instead of following list order and starving three of them
        load = collections.Counter()
        R = routed()
        for _path, info in owned.items():
            for a in info["aliases"]:
                if a in R:
                    load[R[a][0]] += 1

        def shard_gpus(sh):
            ros = HETERO_ROSTER.get(sh[1], {}).get(sh[3])
            als = list(ros) if (sh[2] == "team" and ros) else \
                ([] if sh[2] == "team" else [sh[2]])
            R = routed()
            return [R[a][0] for a in als if a in R]

        def shard_load(sh):
            return max((load[x] for x in shard_gpus(sh)), default=0)

        def priority(sh):
            # 0 = a target-goal workflow on the smallest population, 1 = everything else
            target = 0 if (sh[1] in TARGET_FIRST and sh[0] == TARGET_FIRST_POP) else 1
            return (POP_LANE[sh[0]], target, shard_load(sh),
                    PROTO_ORDER.index(sh[1]), sh[3])

        pending.sort(key=priority)
        if not g["paused"]:
            for s in pending:
                if started >= free:
                    break
                ros = HETERO_ROSTER.get(s[1], {}).get(s[3])
                # A team shard whose protocol has no HETERO_ROSTER entry fell through to an EMPTY
                # check list, so its roster members were never tested for liveness.
                checks = (list(ros) if (s[2] == "team" and ros)
                          else (list(DEFAULT_HETERO_ROSTER.get(s[1], ()))
                                if s[2] == "team" else [s[2]]))
                why = next((w for w in (alias_ok(a, st, g) for a in checks) if w), "")
                if why:
                    continue
                if any(load[x] >= MAX_PER_GPU_OVERRIDE.get(x, MAX_PER_GPU)
                       for x in shard_gpus(s)):
                    continue
                cmd = [sys.executable, str(HERE / "collect_all_matching.py"),
                       "--population", s[0], "--protocol", s[1], "--repeat", str(s[3]),
                       "--workers", str(WORKERS.get(s[1], workers)),
                       "--out", str(out_path(s))]
                if s[2] != "team":
                    cmd += ["--alias", s[2]]
                elif ros:
                    cmd += ["--roster", ",".join(ros)]
                subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
                for x in shard_gpus(s):
                    load[x] += 1
                log.write(f"dispatch {s} {time.strftime('%H:%M:%S', time.gmtime())}\n")
                started += 1
                time.sleep(0.5)
        log.flush()
        now = time.time()
        if owned or started:
            t_run += now - last
        else:
            t_wait += now - last
        last = now
        # Re-read before writing, so an edit made during the cycle is kept.
        gate_write({**g, **gate_read(), "live_collectors": len(owned),
                    "gpu_free_mib": {str(k): v["free"] for k, v in st.items()},
                    "gpu_other_tenant_mib": {str(k): v["other_mib"] for k, v in st.items()},
                    "dispatch_seconds_with_gpu": round(t_run),
                    "dispatch_seconds_waiting": round(t_wait)})
        cycles_failed = 0
      except Exception as exc:                                       # noqa: BLE001
        # A dispatcher that dies on one bad cycle silently stops using the cards. Log and carry on;
        # give up only if every cycle keeps failing.
        cycles_failed += 1
        log.write(f"cycle error {type(exc).__name__}: {exc}\n")
        log.flush()
        if cycles_failed >= 20:
            log.write("20 consecutive cycle failures, stopping dispatch\n")
            log.flush()
            break
      time.sleep(30)
    DISPATCH_PID.unlink(missing_ok=True)
    return 0


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="a")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pause", nargs="?", const="unspecified")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--release-gpu", type=int)
    ap.add_argument("--readmit-gpu", type=int)
    ap.add_argument("--reason", default="")
    ap.add_argument("--only", default=None,
                    help="comma-separated protocols or population:protocol to dispatch "
                         "exclusively; everything else stays pending and resumes on --clear-only")
    ap.add_argument("--clear-only", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if a.status:
        return do_status()
    if a.pause is not None:
        g = gate_write({**gate_read(), "paused": True, "reason": a.pause})
        print(f"paused: {g['reason']}. In-flight shards finish and land; no new dispatch.")
        print("a pause set by hand stays until --resume, it is not lifted by resources freeing.")
        return 0
    if a.resume:
        gate_write({**gate_read(), "paused": False, "reason": ""})
        print("resumed")
        return 0
    if a.release_gpu is not None:
        g = gate_read()
        g["blocked_gpus"] = sorted(set(g.get("blocked_gpus") or []) | {a.release_gpu})
        g["reason"] = a.reason or g.get("reason", "")
        gate_write(g)
        print(f"GPU{a.release_gpu} will receive no new shards. In-flight ones finish and land.")
        print("stop our server on it with: docker stop "
              f"{OUR_CONTAINERS.get(a.release_gpu, '<container>')}")
        return 0
    if a.readmit_gpu is not None:
        g = gate_read()
        g["blocked_gpus"] = [x for x in (g.get("blocked_gpus") or []) if x != a.readmit_gpu]
        gate_write(g)
        print(f"GPU{a.readmit_gpu} readmitted; the dispatcher picks it up within 30 s.")
        return 0
    if a.only is not None:
        sel = [x.strip() for x in a.only.split(",") if x.strip()]
        gate_write({**gate_read(), "only": sel})
        print(f"dispatching only: {sel}. Everything else stays pending, not dropped.")
        return 0
    if a.clear_only:
        gate_write({**gate_read(), "only": []})
        print("filter cleared; the full order resumes within 30 s")
        return 0
    if a.stop:
        gate_write({**gate_read(), "paused": True, "reason": "stopped by hand"})
        if DISPATCH_PID.exists():
            pid = DISPATCH_PID.read_text().strip()
            if pid.isdigit():
                try:
                    os.kill(int(pid), 15)
                except Exception:
                    pass
            DISPATCH_PID.unlink(missing_ok=True)
        owned = live_paths()
        for p, info in owned.items():
            try:
                os.kill(info["pid"], 15)
            except Exception:
                pass
        print(f"dispatch stopped; signalled {len(owned)} collectors. "
              "Units in flight are the only loss; resume re-collects exactly those.")
        return 0
    if a.run:
        return do_run(a.phase, a.pool, a.workers)
    return do_status()


if __name__ == "__main__":
    raise SystemExit(main())
