r"""W3 stage 2: the three recipients under the new peer source.

    python3 code/w3_recipients.py --model qwen3-32b --port 8031
    python3 code/w3_recipients.py --model gemma-4-26b --port 8034
    python3 code/w3_recipients.py --model gpt-oss-120b --port 8034

2,880 calls per recipient: 480 tasks x 3 seats x {full, answer_only}. `no_peers` is not run; it
is panel-independent and reused from the original experiment.

The new peer text is rendered into the **frozen layout**, reproduced here from the original prompts
rather than reinvented:

    "These are the solutions to the problem from other agents:\n\n\n"
    "One agent's solution: {peer_a}\n\n"
    "One agent's solution: {peer_b}\n\n"
    "Using the solutions from other agents as additional information, give your updated answer. "
    "Let's think step by step, and put your final answer after 'Answer:'.\n\n"
    "{problem_block}"

with `peer_a`, `peer_b` the other two seats in ascending seat order -- the mapping holds on
2,499/2,499 prompts. For `answer_only` the two blocks become
`"Answer: {extracted}"`, extracted from the **same** `full` response, never generated separately.

The problem block is lifted verbatim from the existing `no_peers` prompt for the same
(task, seat, decode), so the question text, option list and trailing instruction are byte-identical
to what every other condition in this study used. Each built prompt is checked against the old
prompt of the same condition: everything outside the two peer blocks must match exactly.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import sys
import threading

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                          # noqa: E402

PEER_HEAD = "These are the solutions to the problem from other agents:\n\n\n"
PEER_BLOCK = "One agent's solution: {body}\n\n"
SHARE_INSTR = ("Using the solutions from other agents as additional information, give your "
               "updated answer. Let's think step by step, and put your final answer after "
               "'Answer:'.\n\n")
NOPEER_INSTR = ("Give your updated answer. Let's think step by step, and put your final answer "
                "after 'Answer:'.\n\n")
BASE_SEED = 20261101
CONCURRENCY = 40
SOURCE = "qwen3-32b-new"


def problem_block(no_peer_prompt: str) -> str:
    """Everything from 'The problem:' on, taken verbatim from the frozen no_peers prompt."""
    i = no_peer_prompt.find("The problem:")
    if i < 0:
        raise ValueError("no_peers prompt has no problem block")
    return no_peer_prompt[i:]


def extract_answer(text: str, answer_kind: str):
    """The peer's own stated answer, used to build answer_only from the same full response."""
    sys.path.insert(0, str(C.code_root()))
    from consensus_delivery.adapters import mad_m2 as M2
    return M2.extract_answer(text, answer_kind)


def build(items_by_task_seat: dict, units: dict) -> list:
    """One entry per (task, seat, condition) with the rendered prompt and its contract check."""
    out, problems = [], []
    for (task, seat), nop in sorted(units.items()):
        others = [s for s in (0, 1, 2) if s != seat]
        peers = [items_by_task_seat.get((task, s)) for s in others]
        if any(p is None for p in peers):
            problems.append(dict(task=task, seat=seat, reason="missing new peer generation",
                                 missing=[s for s, p in zip(others, peers) if p is None]))
            continue
        pb = problem_block(nop["messages"][-1]["content"])
        full_bodies = [p["final_text"] for p in peers]
        ans_bodies = []
        for p in peers:
            a = p.get("answer")
            ans_bodies.append("Answer: " + ("" if a is None else str(a)))
        for cond, bodies in (("full", full_bodies), ("answer_only", ans_bodies)):
            content = (PEER_HEAD + "".join(PEER_BLOCK.format(body=b) for b in bodies)
                       + SHARE_INSTR + pb)
            out.append(dict(
                unit_id=f"{SOURCE}|{task}|s{seat}|{cond}", task=task, seat=seat,
                condition=cond, source=SOURCE, benchmark=nop["benchmark"],
                answer_kind=nop["answer_kind"], decode=0,
                peer_seats=others,
                peer_units=[p["unit_id"] for p in peers],
                peer_answers=[p.get("answer") for p in peers],
                peer_chars=[len(b) for b in bodies],
                messages=[{"role": "user", "content": content}],
                rendered_sha=C.sha(content), problem_sha=C.sha(pb),
                seed=BASE_SEED + seat))
    return out, problems


def load_peers() -> dict:
    """Newest ok record per (task, seat), with its parsed answer."""
    path = C.RAW / "w3_peer_qwen3-32b.jsonl"
    by = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            by[(r["task"], r["seat"])] = r
    for r in by.values():
        if "answer" not in r:
            r["answer"] = extract_answer(r.get("final_text") or "", r["answer_kind"])
    return by


def load_nopeers() -> dict:
    units = {}
    with C.input_path("qa_replay_inputs.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] != "no_peers" or r["decode"] != 0:
                continue
            units.setdefault((r["task"], r["agent"]), r)
    return units


def contract_check(built: list) -> dict:
    """Outside the two peer blocks, a built prompt must match the old prompt of that condition.

    Compared on the problem block and the instruction, which is what "same layout, only the peer
    identity changed" means operationally.
    """
    old = collections.defaultdict(dict)
    with C.input_path("qa_replay_inputs.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["decode"] != 0 or r["condition"] == "no_peers":
                continue
            old[(r["task"], r["agent"], r["condition"])][r["panel"]] = \
                r["messages"][-1]["content"]
    ok = bad = nocmp = 0
    examples = []
    for b in built:
        cands = old.get((b["task"], b["seat"], b["condition"]))
        if not cands:
            nocmp += 1
            continue
        mine = b["messages"][0]["content"]
        my_tail = mine[mine.index(SHARE_INSTR):]
        matched = False
        for panel, oc in cands.items():
            if SHARE_INSTR not in oc:
                continue
            if oc[oc.index(SHARE_INSTR):] == my_tail:
                matched = True
                break
        if matched:
            ok += 1
        else:
            bad += 1
            if len(examples) < 3:
                examples.append(b["unit_id"])
    return dict(instruction_and_problem_identical=ok, differing=bad,
                no_old_counterpart=nocmp, examples=examples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.DECODING))
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    out = C.RAW / f"w3_recipient_{a.model}.jsonl"

    peers = load_peers()
    units = load_nopeers()
    built, problems = build(peers, units)
    print(f"peers available {len(peers)}/1440  no_peers cells {len(units)}  built {len(built)}  "
          f"unbuildable {len(problems)}", flush=True)
    if problems:
        (C.OUT / "w3_unbuildable.json").write_text(json.dumps(problems, indent=1) + "\n")
    chk = contract_check(built)
    print(f"  contract check: {chk}", flush=True)
    (C.OUT / f"w3_contract_{a.model}.json").write_text(json.dumps(chk, indent=1) + "\n")
    if chk["differing"]:
        print("FATAL: a built prompt differs from the old layout outside the peer blocks")
        return 1
    if len(built) != 2880:
        print(f"NOTE: built {len(built)} of the planned 2880; peers are still being collected")

    done = C.load_done(out)
    todo = [b for b in built if b["unit_id"] not in done]
    print(f"  done {len(done)}  to do {len(todo)}", flush=True)
    if not todo:
        return 0
    if not C.wait_ready(a.port):
        print(f"FATAL: port {a.port} never became ready")
        return 1

    lock = threading.Lock()
    n = collections.Counter()

    def run(b):
        rec = None
        for attempt in (1, 2):
            rec = C.call(a.port, a.model, b["messages"], C.QA_MAX_TOKENS,
                         C.DECODING[a.model], b["seed"], attempt)
            if rec["status"] == "ok":
                break
        o = dict(b)
        o.update(rec)
        o["messages"] = b["messages"]
        o["model"] = a.model
        o["snapshot"] = C.SNAPSHOT[a.model]
        o["final_text"] = C.final_channel(rec)
        o["finish_reason"] = C.finish_reason(rec)
        o["usage"] = C.usage(rec)
        with lock:
            C.append(out, o)
            n[rec["status"]] += 1
            if o["finish_reason"] == "length":
                n["truncated"] += 1
            t = n["ok"] + n["error"]
            if t % 100 == 0 or t == len(todo):
                print(f"  {t}/{len(todo)} {dict(n)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(run, todo))
    print(f"done {a.model}: {dict(n)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
