"""Freeze an official benchmark split into a hashed, task-ID-pinned block.

**This is a data-layer adaptation and nothing else.** It does not touch any decision protocol, any
prompt, any role order or any scorer. §2.1A requires upstream data-layer issues to be handled in our
loader layer with evidence, not by editing upstream semantics, and there are now four of them in the
frozen five:

1. **Split inconsistency.** `GSM8K.py` and `MMLUPro.py` take `shuffle_and_select("test")`;
   `SQuAD2.py` takes `("train")`; `GPQA.py` and `StrategyQA.py` make no such call.
2. **SQuAD 2.0 no-answer items degenerate to empty references** (`SQuAD2.py:33`), which is
   indistinguishable from a load failure at the `InputExample` layer.
3. **Class names do not match file names** (`SQuAD2.py` defines `EuroparlDownloader`,
   `StrategyQA.py` defines `StrategyGADownloader`).
4. **Found this round, and it is a hard blocker rather than a semantics question:** `GSM8K.py` pins
   the **bare** dataset id `gsm8k`, which current `huggingface_hub` refuses —
   `HfUriError: Repository id must be 'namespace/name'`. The canonical `openai/gsm8k` resolves and
   yields 1319 test records. `StrategyQA` is the other non-namespaced path; it fetches a direct URL.

Separately, four downloaders — `BTVote`, `GPQA`, `SimpleEthicalQuestions`, `StrategyQA` — select their
subset with an **unseeded** `random.shuffle` on the global RNG (zero `random.seed` calls), while the
HF path is seeded (`data_download.py:81,84`). So a sample size does not identify a subset. This module
therefore never selects by sample size: it pins an explicit, sorted list of task IDs and records the
hash of the frozen block, so the same block is recoverable without trusting any shuffle.

Run it where `datasets` is installed; the study's own environment stays pinned and consumes only the
frozen JSON.

    $MALLM_PYTHON -m consensus_delivery.adapters.freeze_official_data \\
        --benchmark gsm8k --n 40 --out data/frozen/gsm8k_block01.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

#: Canonical dataset identifiers and splits, pinned here rather than inherited from upstream.
SOURCES: Dict[str, Dict[str, Any]] = {
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test",
              "question": "question", "answer": "answer",
              "note": "upstream pins the bare id `gsm8k`, which no longer resolves"},
    "squad_v2": {"repo": "rajpurkar/squad_v2", "config": None, "split": "validation",
                 "question": "question", "answer": "answers",
                 "note": "upstream reads the *train* split; the official evaluation split is "
                         "validation, and no-answer items must keep their marker"},
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "config": None, "split": "test",
                 "question": "question", "answer": "answer",
                 "note": "ten options; the option list is part of the prompt adaptation"},
    "gpqa": {"repo": "Idavidrein/gpqa", "config": "gpqa_main", "split": "train",
             "question": "Question", "answer": "Correct Answer",
             "note": "gated: needs an HF token with access granted. Its only split is `train`; "
                     "the dataset ships no test split, so the evaluation population is that split "
                     "and must be named as such rather than called a test set"},
}


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


#: StrategyQA is not on the Hub in the form upstream uses: `StrategyQA.py` fetches BIG-bench's
#: `task.json` directly. That file has no splits, so the whole example set is the population, and
#: upstream then shuffles it **unseeded**. Pinned here by URL and by the sha256 of the fetched bytes.
STRATEGYQA_URL = ("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/"
                  "benchmark_tasks/strategyqa/task.json")


def _freeze_strategyqa(n: int, out: str, offset: int) -> int:
    import urllib.request  # noqa: PLC0415

    with urllib.request.urlopen(STRATEGYQA_URL, timeout=60) as r:
        raw = r.read()
    source_sha = hashlib.sha256(raw).hexdigest()
    examples = json.loads(raw.decode())["examples"]
    keyed = sorted(((hashlib.sha256(str(e["input"]).encode()).hexdigest(), i)
                    for i, e in enumerate(examples)))
    # One task per distinct question, as in `freeze` above: the identity is the question hash, so a
    # repeated question would be two tasks sharing one ID.
    seen: set = set()
    ordered: List[int] = []
    dropped: List[Dict[str, Any]] = []
    for h, i in keyed:
        if h in seen:
            dropped.append({"source_index": i, "question_sha256": h[:16]})
        else:
            seen.add(h)
            ordered.append(i)
    chosen = ordered[offset:offset + n]
    window_hashes = {hashlib.sha256(str(examples[i]["input"]).encode()).hexdigest()[:16]
                     for i in chosen}
    dropped = [d for d in dropped if d["question_sha256"] in window_hashes]
    records: List[Dict[str, Any]] = []
    for rank, idx in enumerate(chosen):
        e = examples[idx]
        # BIG-bench scores a boolean task through `target_scores`; the gold is the key with score 1.
        scores = e.get("target_scores") or {}
        gold = [k for k, v in scores.items() if v == 1]
        records.append({
            "example_id": f"strategyqa-{hashlib.sha256(str(e['input']).encode()).hexdigest()[:16]}",
            "dataset_id": f"big-bench/strategyqa:examples:{idx}",
            "inputs": [str(e["input"])],
            "context": [],
            "references": gold or ["<no-target-score-1>"],
            "metadata": {"source_index": idx, "block_rank": rank,
                         "target_scores": scores, "n_targets": len(scores)},
        })
    manifest = {
        "benchmark": "strategyqa", "repo": STRATEGYQA_URL, "config": None,
        "split": "examples (the file has no splits)", "split_size": len(examples),
        "source_sha256": source_sha,
        "selection": "contiguous window of the whole example set ordered by sha256(input), one "
                     "task per distinct question; no RNG",
        "duplicate_stem_rows_dropped": dropped,
        "offset": offset, "n": len(records),
        "task_ids": [x["example_id"] for x in records],
        "block_sha256": digest(records),
        "upstream_note": "upstream fetches this URL and then shuffles unseeded; pinned by URL and "
                         "by the sha256 of the fetched bytes instead",
        "scope": "data-layer adaptation only",
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1); f.write("\n")
    with open(out.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1); f.write("\n")
    print(f"  strategyqa: {len(records)} of {len(examples)} from BIG-bench task.json")
    print(f"  source sha256 {source_sha[:16]}, block sha256 {manifest['block_sha256'][:16]}")
    print(f"  wrote {out} and its manifest")
    return 0


def freeze(benchmark: str, n: int, out: str, *, offset: int = 0) -> int:
    if benchmark == "strategyqa":
        return _freeze_strategyqa(n, out, offset)
    from datasets import load_dataset  # noqa: PLC0415

    # A gated dataset needs a token. It is read from `.env` programmatically and placed in the
    # process environment only; it is never printed, written to a manifest or committed.
    if not os.environ.get("HF_TOKEN"):
        for cand in (".env", "../.env", "../../.env"):
            p = os.path.join(os.path.dirname(__file__), "..", "..", "..", cand)
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    if line.strip().startswith("HF_TOKEN="):
                        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
                        break
            if os.environ.get("HF_TOKEN"):
                break

    spec = SOURCES[benchmark]
    ds = (load_dataset(spec["repo"], spec["config"], split=spec["split"]) if spec["config"]
          else load_dataset(spec["repo"], split=spec["split"]))

    # Deterministic selection without a shuffle: a stable hash of the question text orders the
    # whole split, and the block is a contiguous window of that order. Reproducible from the
    # identifier alone, with no RNG and no sample-size ambiguity.
    keyed = sorted(((hashlib.sha256(str(r[spec["question"]]).encode()).hexdigest(), i)
                    for i, r in enumerate(ds)))

    # One task per distinct question stem. MMLU-Pro's `test` split repeats stems: four pairs fall in
    # the 200-row window at offset 40, each appearing once with 8 or 9 options and again with 10, so
    # the gold *letter* differs between them while the gold *text* is the same. The identity here is
    # the question hash, so a repeated stem would be two tasks with one ID and the paired design
    # would have nothing to pair. The later occurrence is dropped and recorded, and the window is
    # over distinct stems rather than over rows.
    seen: set = set()
    ordered: List[int] = []
    dropped: List[Dict[str, Any]] = []
    for h, i in keyed:
        if h in seen:
            dropped.append({"source_index": i, "question_sha256": h[:16]})
            continue
        seen.add(h)
        ordered.append(i)
    chosen = ordered[offset:offset + n]
    # Only the duplicates inside the window matter for this block; the rest belong to other windows.
    window_hashes = {hashlib.sha256(str(ds[i][spec["question"]]).encode()).hexdigest()[:16]
                     for i in chosen}
    dropped = [d for d in dropped if d["question_sha256"] in window_hashes]

    records: List[Dict[str, Any]] = []
    for rank, idx in enumerate(chosen):
        r = ds[idx]
        rec: Dict[str, Any] = {
            "example_id": f"{benchmark}-{hashlib.sha256(str(r[spec['question']]).encode()).hexdigest()[:16]}",
            "dataset_id": f"{spec['repo']}:{spec['split']}:{idx}",
            "inputs": [str(r[spec["question"]])],
            "context": [],
            "references": [str(r[spec["answer"]])],
            "metadata": {"source_index": idx, "block_rank": rank},
        }
        # Per-benchmark fields the generic shape drops. A first pass froze MMLU-Pro without its
        # `options`, which makes the question unanswerable, and SQuAD 2.0 without its `context`,
        # which makes it unanswerable too. Both were deleted rather than used.
        if benchmark == "gpqa":
            # The dataset stores the correct answer in a fixed field, so presenting the options in
            # field order would put the answer at A every time. The order is therefore a permutation
            # derived from the task's own id hash — deterministic, recorded, and reproducible — and
            # the resulting gold letter is recorded with it. This is an adapter decision and it is
            # named rather than left implicit.
            opts = [str(r["Correct Answer"]), str(r["Incorrect Answer 1"]),
                    str(r["Incorrect Answer 2"]), str(r["Incorrect Answer 3"])]
            seed = int(hashlib.sha256(rec["example_id"].encode()).hexdigest()[:8], 16)
            order = list(range(len(opts)))
            for i in range(len(order) - 1, 0, -1):          # deterministic Fisher-Yates
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                j = seed % (i + 1)
                order[i], order[j] = order[j], order[i]
            shown = [opts[k] for k in order]
            letters = [chr(ord("A") + i) for i in range(len(shown))]
            gold_letter = letters[order.index(0)]
            rec["inputs"] = [str(r[spec["question"]]) + "\n\n"
                             + "\n".join(f"{c}. {o}" for c, o in zip(letters, shown))]
            rec["references"] = [gold_letter]
            rec["metadata"].update({"n_options": len(shown), "options_shown": shown,
                                    "presentation_order": order, "answer_letter": gold_letter,
                                    "correct_answer_text": str(r["Correct Answer"]),
                                    "subdomain": str(r.get("Subdomain") or ""),
                                    "high_level_domain": str(r.get("High-level domain") or ""),
                                    "order_note": "deterministic permutation seeded by sha256 of the "
                                                  "example_id; field order would place the answer at "
                                                  "A for every task"})
        elif benchmark == "mmlu_pro":
            opts = list(r["options"])
            letters = [chr(ord("A") + i) for i in range(len(opts))]
            rec["inputs"] = [str(r["question"]) + "\n\n"
                             + "\n".join(f"{c}. {o}" for c, o in zip(letters, opts))]
            rec["references"] = [str(r["answer"])]
            rec["metadata"].update({"n_options": len(opts), "answer_index": r["answer_index"],
                                    "category": r["category"], "options": opts,
                                    "answer_letter": str(r["answer"])})
        elif benchmark == "squad_v2":
            texts = list(r["answers"]["text"])
            rec["context"] = [str(r["context"])]
            # §2.1A-2: an unanswerable item must keep its marker. `references=[]` is
            # indistinguishable from a load failure, so the marker is explicit and the reference
            # list is never silently empty.
            rec["references"] = texts or ["<unanswerable>"]
            rec["metadata"].update({"unanswerable": not texts,
                                    "answer_starts": list(r["answers"]["answer_start"]),
                                    "n_reference_spans": len(texts),
                                    "title": str(r["title"]), "squad_id": str(r["id"])})
        records.append(rec)
    if len({x["example_id"] for x in records}) != len(records):
        raise AssertionError("duplicate example_id after freezing; the key is not unique")

    extra: Dict[str, Any] = {}
    if benchmark == "squad_v2":
        n_un = sum(1 for x in records if x["metadata"]["unanswerable"])
        extra["unanswerable"] = n_un
        extra["answerable"] = len(records) - n_un
    if benchmark == "mmlu_pro":
        extra["option_counts"] = sorted({x["metadata"]["n_options"] for x in records})
        extra["categories"] = sorted({x["metadata"]["category"] for x in records})
    manifest = {
        "benchmark": benchmark, **extra,
        "repo": spec["repo"], "config": spec["config"], "split": spec["split"],
        "split_size": len(ds),
        "selection": "contiguous window of the whole split ordered by sha256(question), one task "
                     "per distinct question stem; no RNG",
        "duplicate_stem_rows_dropped": dropped,
        "offset": offset, "n": len(records),
        "task_ids": [x["example_id"] for x in records],
        "block_sha256": digest(records),
        "upstream_note": spec["note"],
        "scope": "data-layer adaptation only; no decision protocol, prompt, role order or scorer "
                 "is altered by this module",
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(out.replace(".json", "") + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"  {benchmark}: {len(records)} of {len(ds)} from {spec['repo']}:{spec['split']}")
    print(f"  block sha256 {manifest['block_sha256'][:16]}")
    print(f"  wrote {out} and its manifest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=sorted(set(SOURCES) | {"strategyqa"}))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"  REFUSING to overwrite {a.out}; a frozen block is evidence")
        return 1
    return freeze(a.benchmark, a.n, a.out, offset=a.offset)


if __name__ == "__main__":
    sys.exit(main())
