import collections, glob, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
st = collections.defaultdict(collections.Counter)
for p in sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True)):
    for line in open(p):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        k = f"{r['protocol']}|{r['backbone']}|{r['arm']}"
        calls = r["record"]["calls"]
        r0p = "\n".join(m["content"] for m in calls[0]["messages_sent"])
        r0n = "".join(ch for ch in r0p if ch.isalnum())
        st[k]["runs"] += 1
        for c in calls:
            rnd = int(c.get("round") or 0)
            if rnd < 1:
                continue
            sent = c.get("messages_after_arm")
            if sent is None:
                st[k]["no_logged_prompt"] += 1
                continue
            txt = "\n".join(m["content"] for m in sent if m.get("role") != "assistant")
            st[k]["calls"] += 1
            prev = [x for x in calls if x["seq"] < c["seq"]]
            if r["protocol"] == "du_mad":
                prev = [x for x in prev if x["agent_index"] != c["agent_index"]]
            hit = False
            for x in prev:
                t = x.get("raw_reply") or ""
                for i in range(0, max(1, len(t) - 40), 10):
                    w = t[i:i + 40]
                    if len(w) == 40 and w in txt and w not in r0p and "".join(ch for ch in w if ch.isalnum()) not in r0n:
                        hit = True
                        break
                if hit:
                    break
            st[k]["peer_text_calls"] += hit
            st[k]["marker_calls"] += ("solution: ```" in txt)
            st[k]["placeholder_calls"] += ("PEER_REPLY_REMOVED" in txt)
out = {k: dict(v, peer_text_pct=round(100 * v["peer_text_calls"] / max(1, v["calls"]), 3)) for k, v in st.items()}
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
