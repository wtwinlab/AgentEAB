r"""Freeze the inputs for this round. Offline, no model call.

    python3 code/build_requests.py

Writes
  data/requests_main.jsonl   2,880 shared-condition requests (480 tasks x 3 seats x {full, answer_only})
  data/requests_pilot.jsonl  36 interface-pilot requests
  data/no_peers_e1.jsonl     the reused Qwen2.5-7B no_peers first decode, 1,440 (task, seat) cells
  logs/inputs.sha256         sha256 of every input read and every file written

Rendering is W3's own `w3_recipients.build` on W3's own peer records, imported, not rewritten.
Every built prompt must equal the prompt W3 sent its Qwen3-32B recipient for the same unit_id.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parents[1]
W3 = WORK
sys.path.insert(0, str(W3 / "code"))
import w3_recipients as W                                                   # noqa: E402

REPLAY = REPO / "experiments/08_cross_family/data/qa_replay_inputs.jsonl"
E1 = REPO / "data/records/framework_raw/e1_units.jsonl"
W3_PEERS = W3 / "raw" / "w3_peer_qwen3-32b.jsonl"
W3_Q3_RECIPIENT = W3 / "raw" / "w3_recipient_qwen3-32b.jsonl"
SEED = 20261105
SOURCE = "qwen3-32b-new"


def sha_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    peers = W.load_peers()
    units = W.load_nopeers()
    built, problems = W.build(peers, units)
    assert not problems and len(built) == 2880, (len(built), problems[:3])

    # byte identity with what W3's Qwen3-32B recipient received
    w3 = {}
    for line in W3_Q3_RECIPIENT.open():
        r = json.loads(line)
        w3[r["unit_id"]] = r["messages"][0]["content"]
    same = sum(w3.get(b["unit_id"]) == b["messages"][0]["content"] for b in built)
    assert same == 2880, same

    main_rows = []
    for b in built:
        main_rows.append(dict(
            unit_id=b["unit_id"], task=b["task"], seat=b["seat"], condition=b["condition"],
            benchmark=b["benchmark"], answer_kind=b["answer_kind"], source=SOURCE,
            peer_seats=b["peer_seats"], peer_units=b["peer_units"],
            peer_answers=b["peer_answers"], peer_chars=b["peer_chars"],
            messages=b["messages"], rendered_sha=b["rendered_sha"],
            seed=SEED + b["seat"]))

    # reused no_peers: qwen25-7b panel, decode 0 -> E1 unit, repeat 0
    rep = {}
    for line in REPLAY.open():
        r = json.loads(line)
        if r["panel"] == "qwen25-7b" and r["decode"] == 0 and r["condition"] == "no_peers":
            rep[r["source_unit_id"]] = r
    e1 = collections.defaultdict(list)
    for line in E1.open():
        r = json.loads(line)
        if r["unit_id"] in rep:
            e1[r["unit_id"]].append(r)
    nop = []
    for uid, u in rep.items():
        att = e1[uid]
        ok = [x for x in att if x.get("status") == "ok"]
        assert len(ok) == 1 and ok[0]["repeat"] == 0, uid
        x = ok[0]
        assert x["messages_sent"][-1]["content"] == u["messages"][-1]["content"]
        assert x["served_model"] == "qwen2.5-7b-instruct"
        nop.append(dict(task=u["task"], seat=u["agent"], benchmark=u["benchmark"],
                        answer_kind=u["answer_kind"], condition="no_peers",
                        e1_unit_id=uid, replay_unit=u["unit"], attempts=len(att),
                        failed_attempts=len(att) - len(ok), finish_reason=x["finish_reason"],
                        prompt_tokens=x["prompt_tokens"],
                        completion_tokens=x["completion_tokens"], raw_reply=x["raw_reply"],
                        e1_post_correct=x["post_correct"],
                        prompt=u["messages"][-1]["content"]))
    nop.sort(key=lambda r: (r["task"], r["seat"]))
    assert len(nop) == 1440 and len({(r["task"], r["seat"]) for r in nop}) == 1440
    assert {(r["task"], r["seat"]) for r in nop} == {(r["task"], r["seat"]) for r in main_rows}

    # interface pilot: first 3 tasks per benchmark, seat 0, three conditions
    by_b = collections.defaultdict(set)
    for r in main_rows:
        by_b[r["benchmark"]].add(r["task"])
    ptasks = {t for b in sorted(by_b) for t in sorted(by_b[b])[:3]}
    pilot = [dict(r, unit_id="pilot|" + r["unit_id"]) for r in main_rows
             if r["task"] in ptasks and r["seat"] == 0]
    for r in nop:
        if r["task"] in ptasks and r["seat"] == 0:
            pilot.append(dict(unit_id=f"pilot|no_peers|{r['task']}|s0", task=r["task"], seat=0,
                              condition="no_peers", benchmark=r["benchmark"],
                              answer_kind=r["answer_kind"],
                              messages=[{"role": "user", "content": r["prompt"]}],
                              e1_prompt_tokens=r["prompt_tokens"], seed=SEED))
    assert len(pilot) == 36, len(pilot)

    D = WORK / "data"
    write(D / "requests_main.jsonl", main_rows)
    write(D / "requests_pilot.jsonl", pilot)
    write(D / "no_peers_e1.jsonl", nop)
    lines = [f"{sha_file(p)}  {p.relative_to(REPO)}" for p in
             (REPLAY, E1, W3_PEERS, W3_Q3_RECIPIENT, D / "requests_main.jsonl",
              D / "requests_pilot.jsonl", D / "no_peers_e1.jsonl")]
    (WORK / "logs").mkdir(exist_ok=True)
    (WORK / "logs" / "inputs.sha256").write_text("\n".join(lines) + "\n")
    print(f"main {len(main_rows)} (identical to W3 Qwen3 recipient prompts: {same}); "
          f"pilot {len(pilot)}; no_peers reused {len(nop)}; "
          f"E1 length {sum(r['finish_reason'] == 'length' for r in nop)}; "
          f"E1 cells with a failed attempt {sum(r['failed_attempts'] > 0 for r in nop)}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
