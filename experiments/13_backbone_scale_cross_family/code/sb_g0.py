import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sb_override as SB


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


SYS = "You are a careful SQL expert."
USER = ("Database schema: CREATE TABLE singer(id INTEGER, name TEXT, age INTEGER);\n"
        "Question: How many singers are older than 30? Think it through, then give the single SQL "
        "query in the form \\boxed{answer}.")


def wait_ready(port, served, limit_s=900):
    import time
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if served in r.read().decode():
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def main():
    res, ok = {}, True
    for alias in [SB.ALIAS[b] for b in SB.BACKBONES]:
        port = SB.PORTS[alias]
        served = SB.SERVED[alias][0]
        extra = SB.SERVED[alias][4] or {}
        wait_ready(port, served)
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
        b = post(port, "/v1/chat/completions", dict(model=served, messages=msgs, temperature=0.7,
                                                     top_p=1.0, max_tokens=512, logprobs=True,
                                                     top_logprobs=1, **extra))
        ch = b["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        lp = (ch.get("logprobs") or {}).get("content") or []
        tok = post(port, "/tokenize", dict(model=served, messages=msgs, add_generation_prompt=True))
        det = post(port, "/detokenize", dict(model=served, tokens=tok["tokens"]))
        rendered = det.get("prompt", "")
        checks = {
            "no_think_tag_in_content": "<think>" not in content and "</think>" not in content,
            "reasoning_content_empty": rc == "",
            "logprobs_nonempty": len(lp) > 0 and all("logprob" in e for e in lp),
            "system_prompt_rendered": SYS in rendered,
            "boxed_answer_present": "\\boxed" in content,
        }
        if "gpt-oss" in served:
            checks["reasoning_content_empty"] = True
            checks["reasoning_separated_from_content"] = "<|channel|>" not in content
        if "qwen3" in served or "glm" in served or "nemotron" in served:
            checks["info_template_closes_think"] = "</think>" in rendered[-200:]
        soft = {"no_think_tag_in_content", "reasoning_content_empty"} if os.environ.get("SB_G0_SOFT") else set()
        gate = {k: v for k, v in checks.items()
                if k != "boxed_answer_present" and not k.startswith("info_") and k not in soft}
        toks = [e.get("token", "") for e in lp]
        checks["info_logprobs_include_reasoning_channel"] = any("<|channel|>" in t for t in toks)
        ok = ok and all(gate.values())
        res[alias] = dict(served=served, checks=checks, finish_reason=ch.get("finish_reason"),
                          usage=b.get("usage"), content_head=content[:600],
                          rendered_prompt_tail=rendered[-400:], n_logprob_tokens=len(lp),
                          request_extra=extra, reasoning_head=rc[:400])
    if os.environ.get("SB_HARMONY_SLICE") or os.environ.get("SB_THINK_SLICE"):
        import sb_harmony as HS
        import ext_unit_driver as EUD
        units = SB.D.frozen("sql_bird_v2")["unit_ids"][:20]
        for alias in [SB.ALIAS[b] for b in SB.BACKBONES]:
            port, served, extra = SB.PORTS[alias], SB.SERVED[alias][0], SB.SERVED[alias][4] or {}
            rows = []
            for u in units:
                q = EUD.sql_task(u)["question"]
                b = post(port, "/v1/chat/completions", dict(
                    model=served, messages=[{"role": "user", "content": q + "\n\nPut the final SQL in \\boxed{}."}],
                    temperature=1.0, top_p=1.0, max_tokens=4096, logprobs=True, top_logprobs=1, **extra))
                ch = b["choices"][0]
                fn = HS.check_think if os.environ.get("SB_THINK_SLICE") else HS.check
                rows.append(fn((ch.get("logprobs") or {}).get("content") or [], ch["message"].get("content")))
            agg = {k: sum(1 for r in rows if r[k]) for k in ("marker", "exact", "ws_equal", "nonempty")}
            passed = agg["marker"] == len(rows) and agg["ws_equal"] == len(rows) and agg["nonempty"] == len(rows)
            res[alias]["harmony_slice_probe"] = dict(n=len(rows), **agg, passed=passed, rows=rows)
            ok = ok and passed
    res["pass"] = ok
    (SB.OUT / ("G0_%s.json" % "_".join(SB.BACKBONES))).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
