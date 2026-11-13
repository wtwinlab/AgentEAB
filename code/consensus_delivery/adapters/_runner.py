r"""The block runner every Batch 2 adapter shares. One place, so the guarantees are one place.

`du_mad.py` and `debate_or_vote.py` each carry their own copy of this loop, written before there
were enough adapters to justify factoring it. The four that follow use this instead, so the
properties below hold for all of them by construction rather than by four careful transcriptions:

* **a collected run is evidence** — the runner refuses to overwrite an existing output;
* **atomic writes** — records go to a temporary name and are moved, so a reader never sees half a
  file;
* **failures are named, not dropped** — a task that raises is recorded with its error and counted in
  the manifest, and `records + n_failed` is asserted against the assigned count;
* **the accounting is in the manifest** — `unexplained_absent` is computed, not left implicit;
* **the configuration is in the manifest** — endpoint identity, revision, template hash, serving
  image, block digest, adapter digest, and whatever the mechanism adds.

The per-task callable gets the question already assembled for the benchmark, and returns the record.
Nothing here parses or scores: that is the adapter's and the analysis modules' business.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..providers import TransportFailure
from ..registry_live import REGISTRY


def question_for(benchmark: str, task: Dict[str, Any]) -> str:
    """The task as the mechanism should see it: passage prepended, options lettered.

    Shared so two adapters cannot disagree about what "the question" was for the same task id,
    which would make their contrast meaningless while looking fine in both records.
    """
    q = task["inputs"][0]
    meta = task.get("metadata") or {}
    if benchmark == "squad_v2":
        ctx = (task.get("context") or [""])[0] if isinstance(task.get("context"), list) else ""
        return f"{ctx}\n\n{q}" if ctx else q
    if benchmark in ("mmlu_pro", "gpqa"):
        opts = meta.get("options") or []
        if opts:
            return q + "\n\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
    return q


def run_block(*, block_path: str, out_path: str, adapter_file: str, mechanism: str,
              aliases: Sequence[str], limit: int,
              per_task: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
              manifest_extra: Optional[Dict[str, Any]] = None,
              upstream: str = "") -> int:
    """Collect one cell. `per_task(question, task_id, task)` returns the record."""
    if os.path.exists(out_path):
        print(f"  REFUSING to overwrite {out_path}; a collected run is evidence")
        return 1
    block = json.load(open(block_path, encoding="utf-8"))
    bman_path = block_path.replace(".json", "") + ".manifest.json"
    bman = json.load(open(bman_path, encoding="utf-8")) if os.path.exists(bman_path) else {}
    benchmark = bman.get("benchmark") or os.path.basename(block_path).split("_block")[0]
    tasks = block[:limit] if limit else block

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, task in enumerate(tasks, 1):
        tid = task["example_id"]
        try:
            rec = per_task(question_for(benchmark, task), tid, task)
        except (TransportFailure, Exception) as e:   # noqa: BLE001 — a lost task is named
            failures.append({"example_id": tid, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{k}/{len(tasks)}] {tid} FAILED {type(e).__name__}", flush=True)
            continue
        rec["dataset_id"] = task["dataset_id"]
        rec["references"] = task["references"]
        records.append(rec)
        if k % 10 == 0 or k == len(tasks):
            print(f"  [{k}/{len(tasks)}] {len(records)} ok, {len(failures)} failed, "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)

    eps = []
    for alias in aliases:
        e = REGISTRY.get(alias)
        eps.append({"alias": e.alias, "provider": e.provider, "served_name": e.served_name,
                    "checkpoint": e.checkpoint, "revision": e.revision,
                    "chat_template_sha": e.chat_template_sha, "serving_image": e.serving_image,
                    "base_url": e.base_url})
    manifest: Dict[str, Any] = {
        "kind": "native-run", "mechanism": mechanism, "upstream_source": upstream,
        "adapter_sha256": hashlib.sha256(open(adapter_file, "rb").read()).hexdigest(),
        "benchmark": benchmark, "block_path": block_path,
        "block_sha256": bman.get("block_sha256"),
        "roster": list(aliases),
        # **One type, always.** This used to be a dict for a single-endpoint cell and a list for a
        # roster, and the panel's `(m.get("endpoint") or {}).get("served_name")` crashed on the
        # list — a key whose type varies is exactly what breaks a reader. `endpoint` is the primary
        # endpoint; the full roster is `endpoints`.
        "endpoint": eps[0],
        "endpoints": eps,
        "endpoint_url": eps[0]["base_url"],
        "model_name": (eps[0]["served_name"] if len(eps) == 1
                       else "+".join(e["served_name"] for e in eps)),
        "num_samples": len(tasks), "records_written": len(records),
        "failed_example_ids": [f["example_id"] for f in failures], "n_failed": len(failures),
        "failure_errors": failures,
        "reason_counts": dict(collections.Counter(f["error"].split(":")[0] for f in failures)),
        # Computed, not implied. A non-zero value means the cell is not complete.
        "unexplained_absent": len(tasks) - len(records) - len(failures),
        "endpoint_tokens": {"prompt": sum(r.get("tokens_in") or 0 for r in records),
                            "generation": sum(r.get("tokens_out") or 0 for r in records)},
        "usd": round(sum(r.get("usd") or 0 for r in records), 6),
        "wall_s": round(time.time() - t0, 1),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(manifest_extra or {}),
    }
    with open(out_path.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  {os.path.basename(block_path)}: {len(records)} records, {len(failures)} failed, "
          f"{manifest['unexplained_absent']} unexplained, {manifest['wall_s']:.0f}s, "
          f"tokens={manifest['endpoint_tokens']}")
    return 0
