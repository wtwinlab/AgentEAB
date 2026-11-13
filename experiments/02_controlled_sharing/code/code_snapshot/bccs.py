r"""BCCS — "Belief-Calibrated Multi-Agent Consensus Seeking", as released.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..providers import TransportFailure, chat
from ..registry_live import REGISTRY

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM = "_external/mechanism_screening_2026-10-08/repos/bccs @ 5af8e7c"

#: `run_math.py:219`, verbatim, minus the chat-template markers the server adds itself.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
#: `construct_message`'s two tails, verbatim.
TAIL_TRUSTABLE = ("\n\n Selecting and using the trustable solutions from current collaboration as "
                  "additional information, can you provide your answer to the problem? \n {}. ")
TAIL_LEADING = ("\n\n Selecting and using the leading solutions from current collaboration as "
                "additional information, can you provide your answer to the problem? \n {}. ")
PREFIX = "These are the solutions to the problem from other agents: "

#: sklearn's `CountVectorizer` default token pattern.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
ANSWER_KIND = {"gsm8k": "numeric", "mmlu_pro": "mcq", "gpqa": "mcq", "strategyqa": "boolean"}
#: Its released datasets are MATH and MMLU. SQuAD 2.0 is excluded in the freeze; the other two are
#: labelled adaptations of the `\boxed{}` convention.
ADAPTED = {"gsm8k", "gpqa", "strategyqa"}


# --------------------------------------------------------------------------- beliefs
def beliefs(token_logprobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Both readings of `clustering.py:preprocess_data`, side by side.
    """
    words: List[str] = []
    sentence_products: List[float] = []
    prob = 1.0
    empty_token_contributions = 0
    for e in token_logprobs or []:
        lp = e.get("logprob")
        if lp is not None:
            prob *= math.exp(lp)
        tok = str(e.get("token") or "").strip()
        if len(tok) == 0:
            # The released code updates the product and *then* skips the token, so whitespace
            # contributes to the belief while being absent from the clustered corpus.
            empty_token_contributions += 1
            continue
        words.append(tok)
        if tok in (".", ")."):
            sentence_products.append(prob)
            prob = 1.0
    released_list = list(sentence_products) + [prob]
    return {
        "released": released_list[-1] if released_list else 0.0,
        "intended": (sentence_products[-1] if sentence_products else (prob if words else 0.0)),
        "n_sentences": len(sentence_products),
        "ended_on_sentence_boundary": bool(words) and words[-1] in (".", ")."),
        "empty_token_contributions": empty_token_contributions,
        "corpus": " ".join(words),
    }


# --------------------------------------------------------------------------- tf-idf and k-means
def count_matrix(corpus: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """`CountVectorizer()` defaults: lowercase, `\\b\\w\\w+\\b`, vocabulary sorted."""
    docs = [TOKEN_RE.findall((c or "No answer").lower()) for c in corpus]
    vocab = sorted({t for d in docs for t in d})
    index = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(docs), len(vocab)), dtype=float)
    for r, d in enumerate(docs):
        for t in d:
            m[r, index[t]] += 1.0
    return m, vocab


def tfidf(counts: np.ndarray) -> np.ndarray:
    """`TfidfTransformer()` defaults: `smooth_idf=True`, `use_idf=True`, `norm='l2'`."""
    if counts.size == 0:
        return counts
    n = counts.shape[0]
    df = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    w = counts * idf
    norms = np.sqrt((w * w).sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return w / norms


def kmeans(weights: np.ndarray, n_clusters: int, seed: int,
           n_init: int = 10, max_iter: int = 300) -> List[int]:
    """Lloyd's algorithm with k-means++ init. Seeded, because the released call is not."""
    n = weights.shape[0]
    if n == 0:
        return []
    k = max(1, min(n_clusters, n))
    rng = np.random.default_rng(seed)
    best_labels: List[int] = [0] * n
    best_inertia = math.inf
    for _ in range(n_init):
        # k-means++ initialisation.
        centres = [weights[rng.integers(n)]]
        for _c in range(1, k):
            d2 = np.min(((weights[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2),
                        axis=1)
            total = d2.sum()
            probs = (d2 / total) if total > 0 else np.full(n, 1.0 / n)
            centres.append(weights[rng.choice(n, p=probs)])
        C = np.array(centres)
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            d2 = ((weights[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            if _it and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = weights[labels == j]
                if len(members):
                    C[j] = members.mean(axis=0)
        inertia = float(((weights - C[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = [int(x) for x in labels]
    return best_labels


def clusters_from_labels(labels: Sequence[int]) -> List[List[int]]:
    """`kmeans`'s return shape: a list per label, in first-appearance order of the label."""
    out: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(int(lab), []).append(i)
    return [out[k] for k in out]


# --------------------------------------------------------------------------- the estimators
def list_mean(lst: Sequence[float]) -> float:
    """`run_math.py:98`: an empty list means 0, which is what makes some divisions undefined."""
    return 0.0 if len(lst) == 0 else float(np.mean(lst))


def calculate_entropy(lst: Sequence[float]) -> float:
    """`run_math.py:136`: `sum(-a log a)/len`, with a bare except that treats failures as 0."""
    entropy = 0.0
    for a in lst:
        try:
            entropy += -a * math.log(a)
        except Exception:                         # noqa: BLE001 — upstream's bare except, kept
            entropy += 0.0
    return entropy / len(lst) if len(lst) else 0.0


def estimate_uncertainty(clusters: Sequence[Sequence[int]], probs: Sequence[float]) -> List[float]:
    return [calculate_entropy([probs[i] for i in c]) for c in clusters]


def _strip_period(s: str) -> str:
    return s[:-1] if s and s[-1] == "." else s


def estimate_conflict(clusters: Sequence[Sequence[int]], probs: Sequence[float],
                      support: Sequence[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """`run_math.py:104`, transcribed. Undefined divisions are **recorded**, not hidden."""
    n = len(clusters)
    scores = [[0.0 for _ in range(n)] for _ in range(n)]
    undefined: List[Dict[str, Any]] = []
    stripped = [_strip_period(s) for s in support]
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            set_a = {stripped[t] for t in clusters[i]}
            set_b = {stripped[t] for t in clusters[j]}
            inter, union = set_a & set_b, set_a | set_b
            a_i = [probs[c] for c in clusters[i] if support[c] in inter]
            a_u = [probs[c] for c in clusters[i] if support[c] in union - inter]
            b_i = [probs[c] for c in clusters[j] if support[c] in inter]
            b_u = [probs[c] for c in clusters[j] if support[c] in union - inter]
            try:
                if len(inter) != 0 and len(a_u + b_u) > 0:
                    score = ((list_mean(a_u + b_u) / list_mean(a_u + b_u + a_i + b_i))
                             * (abs(list_mean(a_i) - list_mean(b_i))
                                / abs(list_mean(a_u) - list_mean(b_u))))
                elif len(inter) != 0 and len(a_u + b_u) == 0:
                    score = abs(list_mean(a_i) - list_mean(b_i))
                else:
                    score = ((list_mean(a_u + b_u) / list_mean(a_u + b_u + a_i + b_i))
                             / abs(list_mean(a_u) - list_mean(b_u)))
            except ZeroDivisionError:
                # The released formula is undefined here. With equal beliefs — which is exactly what
                # the released `probs[-1]` produces — this is the common case, not the edge case.
                undefined.append({"cluster_i": i, "cluster_j": j,
                                  "reason": "the released conflict formula divides by zero",
                                  "mean_a_i": list_mean(a_i), "mean_b_i": list_mean(b_i),
                                  "mean_a_u": list_mean(a_u), "mean_b_u": list_mean(b_u)})
                score = 0.0
            scores[j][i] = scores[i][j] = score
    return scores, undefined


def estimate_consensus(support: Sequence[str], probs: Sequence[float]) -> Dict[str, Any]:
    """`run_math.py:151`, transcribed, with the two thresholds and the belief mass exposed."""
    srs = [_strip_period(s if s != "" else "None") for s in support]
    if not srs:
        return {"verdict": "no consensus", "frequency_share": 0.0, "consensus_score": 0.0,
                "modal_answer": None}
    modal = max(srs, key=srs.count)
    freq = srs.count(modal)
    score = sum(p for s, p in zip(srs, probs) if s == modal)
    total = sum(probs)
    consensus_score = (score / total) if total else 1.0
    share = freq / len(srs)
    if share > (2 / 3) and consensus_score > 0.8:
        verdict = "full consensus"
    elif share >= (2 / len(srs)) and consensus_score > 0.5:
        verdict = "partial consensus"
    else:
        verdict = "no consensus"
    return {"verdict": verdict, "frequency_share": round(share, 4),
            "consensus_score": round(consensus_score, 4), "modal_answer": modal,
            "modal_frequency": freq}


def extract_boxed(text: str, kind: str) -> str:
    r"""`solve_math_problems`' target: the content of the last `\boxed{...}`. `""` when absent."""
    hits = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    raw = hits[-1].strip() if hits else ""
    if not raw:
        return ""
    if kind == "numeric":
        m = re.findall(r"[-+]?\d*\.?\d+", raw.replace(",", "").replace("$", ""))
        if not m:
            return ""
        try:
            return str(round(float(m[-1]), 4))
        except ValueError:
            return ""
    if kind == "mcq":
        m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", raw.upper())
        return m.group(1) if m else ""
    if kind == "boolean":
        low = raw.lower()
        return "yes" if low.startswith("y") else "no" if low.startswith("n") else ""
    return " ".join(raw.split())


def find_top_indices(lst: Sequence[float], top_k: int) -> List[int]:
    """`run_math.py:84`: the indices of the `top_k` largest values, descending."""
    return [i for i, _v in sorted(enumerate(lst), key=lambda kv: kv[1], reverse=True)[:top_k]]


def construct_message(current_agent: int, responses: Sequence[str], question: str,
                      probs: Sequence[float], clusters: Sequence[Sequence[int]],
                      conflict: Sequence[Sequence[float]], uncertainties: Sequence[float],
                      T: int, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    """`construct_message`, transcribed. Returns the prompt and which peers it showed, and why."""
    prefix = PREFIX
    shown: List[Dict[str, Any]] = []
    cluster_idx = -1
    cluster_i: Sequence[int] = []
    for idx0, cluster in enumerate(clusters):
        if current_agent in cluster:
            cluster_i = cluster
            cluster_idx = idx0
    cluster_i_probs = [probs[i] for i in cluster_i]

    if T == 0 and cluster_idx >= 0:
        if uncertainties and uncertainties[cluster_idx] == max(uncertainties):
            if cluster_i_probs and probs[current_agent] == min(cluster_i_probs):
                # The agent least confident inside the most uncertain cluster is shown the
                # *conflicting* clusters' leaders.
                for c in range(len(conflict[cluster_idx])):
                    if conflict[cluster_idx][c] > 2:
                        for agent_idx in range(len(responses)):
                            if agent_idx not in clusters[c]:
                                continue
                            if probs[agent_idx] != max(probs[j] for j in clusters[c]):
                                continue
                            prefix += "\n\n One conflicting agent solution: ```{}```".format(
                                responses[agent_idx])
                            shown.append({"agent": agent_idx, "role": "conflicting",
                                          "cluster": c})
            else:
                for c in range(len(conflict[cluster_idx])):
                    if 0 <= conflict[cluster_idx][c] <= 2:
                        for agent_idx in range(len(responses)):
                            if agent_idx not in clusters[c]:
                                continue
                            if probs[agent_idx] != max(probs[j] for j in clusters[c]):
                                continue
                            prefix += "\n\n One supporting agent solution: ```{}```".format(
                                responses[agent_idx])
                            shown.append({"agent": agent_idx, "role": "supporting", "cluster": c})
        else:
            max_unc_idx = list(uncertainties).index(max(uncertainties))
            max_unc_probs = [probs[i] for i in clusters[max_unc_idx]]
            min_pos = max_unc_probs.index(min(max_unc_probs))
            for c in range(len(conflict[cluster_idx])):
                if 0 <= conflict[cluster_idx][c] <= 2:
                    for agent_idx in range(len(responses)):
                        if agent_idx not in clusters[c]:
                            continue
                        if c == max_unc_idx and agent_idx == clusters[max_unc_idx][min_pos]:
                            continue
                        if probs[agent_idx] != max(probs[j] for j in clusters[c]):
                            continue
                        prefix += "\n\n One supporting agent solution: ```{}```".format(
                            responses[agent_idx])
                        shown.append({"agent": agent_idx, "role": "supporting", "cluster": c})
        prefix += TAIL_TRUSTABLE.format(question)
    else:
        for pos in find_top_indices(cluster_i_probs, top_k):
            agent_idx = cluster_i[pos]
            prefix += "\n\n One leader solution: ```{}```".format(responses[agent_idx])
            shown.append({"agent": agent_idx, "role": "leader", "cluster": cluster_idx})
        prefix += TAIL_LEADING.format(question)
    return prefix, shown


def run_one(question: str, *, task_id: str, benchmark: str, alias: str, agents: int, rounds: int,
            n_clusters: int, top_k: int, max_tokens: int, timeout_s: float, retries: int,
            temperature: float, belief: str) -> Dict[str, Any]:
    kind = ANSWER_KIND[benchmark]
    calls: List[Dict[str, Any]] = []
    order: List[Dict[str, Any]] = []
    per_round: List[Dict[str, Any]] = []
    responses: List[str] = [""] * agents
    t0 = time.time()
    seq = 0
    T = 0
    clusters: List[List[int]] = []
    conflict: List[List[float]] = []
    uncs: List[float] = []
    probs: List[float] = []
    stop_reason = f"fixed-round-budget-exhausted after {rounds} rounds"
    stopped_at: Optional[int] = None
    undefined_divisions: List[Dict[str, Any]] = []

    for rnd in range(rounds):
        belief_rows: List[Dict[str, Any]] = []
        for i in range(agents):
            if rnd == 0:
                user = question
            else:
                user, shown = construct_message(i, responses, question, probs, clusters,
                                                conflict, uncs, T, top_k)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            text, rec = chat(registry=REGISTRY, alias=alias, messages=messages,
                             temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                             retries=retries, timeout_s=timeout_s, logprobs=True)
            responses[i] = text
            b = beliefs(rec.token_logprobs or [])
            belief_rows.append(b)
            calls.append({
                "seq": seq, "round": rnd, "agent_index": i, "attempt": rec.attempt,
                "messages_sent": [dict(m) for m in messages],
                "peers_shown": (shown if rnd > 0 else []),
                "raw_reply": text, "extracted_answer": extract_boxed(text, kind),
                "belief_released": b["released"], "belief_intended": b["intended"],
                "n_sentences": b["n_sentences"],
                "ended_on_sentence_boundary": b["ended_on_sentence_boundary"],
                "empty_token_contributions": b["empty_token_contributions"],
                "n_logprob_tokens": len(rec.token_logprobs or []),
                "request_sha": rec.request_sha, "served_model": rec.served_model,
                "fingerprint": rec.fingerprint, "finish_reason": rec.finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                "latency_s": rec.latency_s, "usd": rec.usd,
            })
            order.append({"seq": seq, "round": rnd, "agent_index": i})
            seq += 1
        support = [extract_boxed(r, kind) for r in responses]
        corpus = [b["corpus"] for b in belief_rows]
        probs_released = [b["released"] for b in belief_rows]
        probs_intended = [b["intended"] for b in belief_rows]
        probs = probs_intended if belief == "intended" else probs_released
        labels = kmeans(tfidf(count_matrix(corpus)[0]), n_clusters,
                        seed=int.from_bytes(task_id.encode()[-4:], "little") + rnd)
        clusters = clusters_from_labels(labels)
        cons = estimate_consensus(support, probs)
        cons_released = estimate_consensus(support, probs_released)
        per_round.append({
            "round": rnd, "support": support, "clusters": clusters,
            "beliefs_released": probs_released, "beliefs_intended": probs_intended,
            "consensus": cons, "consensus_under_released_belief": cons_released,
        })
        if cons["verdict"] == "full consensus":
            stop_reason = "full consensus"
            stopped_at = rnd
            break
        T = 1 if cons["verdict"] == "no consensus" else 0
        conflict, undef = estimate_conflict(clusters, probs, support)
        undefined_divisions.extend([dict(u, round=rnd) for u in undef])
        uncs = estimate_uncertainty(clusters, probs)
        per_round[-1].update({"conflict": conflict, "uncertainties": uncs, "T": T,
                              "undefined_divisions": undef})

    final = per_round[-1]
    delivered = bool(final["consensus"]["modal_answer"]) and final["consensus"]["modal_answer"] not in ("", "None")
    return {
        "run_id": f"bccs-{alias}-{task_id}-r0",
        "example_id": task_id, "mechanism": "bccs", "model_alias": alias, "repeat": 0,
        "question": question, "benchmark": benchmark, "answer_kind": kind,
        "is_task_adaptation": benchmark in ADAPTED,
        "adaptation_note": ("its released datasets are MATH and MMLU; this benchmark keeps the "
                            "\\boxed{} convention and is a labelled adaptation"
                            if benchmark in ADAPTED else None),
        "agents": agents, "rounds_configured": rounds, "rounds_run": len(per_round),
        "n_clusters": n_clusters, "top_k": top_k, "temperature": temperature,
        "belief_used": belief,
        "per_agent_final_answer": [s if s else None for s in final["support"]],
        "n_distinct_final_answers": len({s for s in final["support"] if s}),
        "majority_answer": final["consensus"]["modal_answer"] if delivered else None,
        # The stop rule is the mechanism's own and it does fire.
        "rule_satisfied": stop_reason == "full consensus",
        "rule_note": ("estimate_consensus: frequency share > 2/3 AND belief-weighted consensus "
                      "score > 0.8, and the released loop DOES break on it"),
        "stopped_at_round": stopped_at,
        "native_stop_reason": stop_reason,
        "consensus_per_round": [r["consensus"]["verdict"] for r in per_round],
        "consensus_per_round_under_released_belief":
            [r["consensus_under_released_belief"]["verdict"] for r in per_round],
        "accepted_delivery": "not-measured",
        "accepted_delivery_note": ("BCCS judges consensus from answer frequency and a belief "
                                   "probability. It solicits no acceptance of a delivered "
                                   "artifact; a belief is not a stance and absence of a verdict is "
                                   "not a refusal"),
        "absence_cause": (None if delivered else
                          "no agent produced a readable \\boxed{} answer"),
        # A degeneracy flag that floating point can defeat is worse than the number it was hiding.
        "released_belief_min": min((b for r in per_round for b in r["beliefs_released"]),
                                   default=None),
        "released_belief_max": max((b for r in per_round for b in r["beliefs_released"]),
                                   default=None),
        "released_belief_spread": (
            round(max(b for r in per_round for b in r["beliefs_released"])
                  - min(b for r in per_round for b in r["beliefs_released"]), 12)
            if per_round and per_round[0]["beliefs_released"] else None),
        "intended_belief_spread": (
            round(max(b for r in per_round for b in r["beliefs_intended"])
                  - min(b for r in per_round for b in r["beliefs_intended"]), 12)
            if per_round and per_round[0]["beliefs_intended"] else None),
        "released_belief_is_degenerate": all(
            abs(b - 1.0) < 1e-6 for r in per_round for b in r["beliefs_released"]),
        "undefined_conflict_divisions": undefined_divisions,
        "rounds_detail": per_round,
        "calls": calls, "order": order,
        "tokens_in": sum(c["prompt_tokens"] or 0 for c in calls),
        "tokens_out": sum(c["completion_tokens"] or 0 for c in calls),
        "usd": round(sum(c["usd"] or 0 for c in calls), 6),
        "wall_s": round(time.time() - t0, 2),
    }


def _assert_transcription() -> None:
    assert SYSTEM_PROMPT == ("Please reason step by step, and put your final answer within "
                            "\\boxed{}."), "system prompt drifted"
    assert PREFIX == "These are the solutions to the problem from other agents: "
    assert TAIL_TRUSTABLE.startswith("\n\n Selecting and using the trustable solutions")
    assert TAIL_LEADING.startswith("\n\n Selecting and using the leading solutions")
    # The belief issue, demonstrated rather than described: a generation ending in "." gets 1.0.
    toks = [{"token": "Four", "logprob": -0.5}, {"token": ".", "logprob": -0.1}]
    b = beliefs(toks)
    assert b["released"] == 1.0, b
    assert abs(b["intended"] - math.exp(-0.6)) < 1e-9, b
    assert b["ended_on_sentence_boundary"] is True
    # And one that does not end on a boundary: both readings agree.
    b2 = beliefs([{"token": "Four", "logprob": -0.5}])
    assert abs(b2["released"] - b2["intended"]) < 1e-12, b2
    # The consensus thresholds.
    c = estimate_consensus(["4", "4", "4", "4", "4", "4", "9"], [1.0] * 7)
    assert c["verdict"] == "full consensus", c
    c2 = estimate_consensus(["4", "4", "4", "4", "9", "9", "9"], [1.0] * 7)
    assert c2["verdict"] == "partial consensus", c2
    c3 = estimate_consensus(["1", "2", "3", "4", "5", "6", "7"], [1.0] * 7)
    assert c3["verdict"] == "no consensus", c3
    assert abs(c["consensus_score"] - c["frequency_share"]) < 1e-9, (c,)
    # The conflict formula is undefined with equal beliefs.
    _s, undef = estimate_conflict([[0, 1], [2, 3]], [1.0, 1.0, 1.0, 1.0], ["a", "b", "a", "c"])
    assert undef, "equal beliefs must make the released conflict formula undefined"
    # Uncertainty collapses to zero with unit beliefs.
    assert estimate_uncertainty([[0, 1]], [1.0, 1.0]) == [0.0]
    # tf-idf on a hand-checkable corpus: two docs sharing one term.
    counts, vocab = count_matrix(["aa bb", "aa cc"])
    assert vocab == ["aa", "bb", "cc"], vocab
    w = tfidf(counts)
    assert abs(float((w ** 2).sum(axis=1)[0]) - 1.0) < 1e-9, "rows must be l2-normalised"
    idf_shared = math.log((1 + 2) / (1 + 2)) + 1.0
    idf_unique = math.log((1 + 2) / (1 + 1)) + 1.0
    expected = idf_shared / math.sqrt(idf_shared ** 2 + idf_unique ** 2)
    assert abs(float(w[0, 0]) - expected) < 1e-9, (float(w[0, 0]), expected)
    assert extract_boxed(r"so \boxed{42}.", "numeric") == "42.0"
    assert extract_boxed("no box here", "numeric") == ""
    assert find_top_indices([0.1, 0.9, 0.5], 2) == [1, 2]


_assert_transcription()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", type=int, default=7)        # run_math.py:220
    ap.add_argument("--rounds", type=int, default=3)        # run_math.py:221
    ap.add_argument("--clusters", type=int, default=3)      # kmeans(..., n_clusters=3)
    ap.add_argument("--top-k", type=int, default=2)         # run_math.py:222
    ap.add_argument("--temperature", type=float, default=0.7)   # SamplingParams(temperature=0.7)
    ap.add_argument("--belief", default="intended", choices=("intended", "released"),
                    help="which reading of the belief drives the mechanism; `released` is "
                         "degenerate and makes the conflict estimator undefined")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    bman = a.block.replace(".json", "") + ".manifest.json"
    bench = (json.load(open(bman, encoding="utf-8")).get("benchmark")
             if os.path.exists(bman) else os.path.basename(a.block).split("_block")[0])
    if bench not in ANSWER_KIND:
        print(f"  REFUSING: BCCS has no declared answer kind for benchmark {bench!r}")
        return 1

    from ._runner import run_block  # noqa: PLC0415
    return run_block(
        block_path=a.block, out_path=a.out, adapter_file=os.path.abspath(__file__),
        mechanism="bccs", aliases=[a.alias], limit=a.limit, upstream=UPSTREAM,
        per_task=lambda q, tid, task: run_one(
            q, task_id=tid, benchmark=bench, alias=a.alias, agents=a.agents, rounds=a.rounds,
            n_clusters=a.clusters, top_k=a.top_k, max_tokens=a.max_tokens,
            timeout_s=a.timeout_s, retries=a.retries, temperature=a.temperature,
            belief=a.belief),
        manifest_extra={
            "agents": a.agents, "rounds": a.rounds, "n_clusters": a.clusters, "top_k": a.top_k,
            "temperature": a.temperature, "max_tokens": a.max_tokens,
            "belief_used": a.belief,
            "released_defaults": "run_math.py: 7 agents, 3 rounds, top_k 2, temperature 0.7, "
                                 "logprobs on, 3 clusters, Qwen/Qwen2.5-7B-Instruct",
            "stop_rule": "estimate_consensus == 'full consensus' breaks the loop (run_math.py:256)",
            "belief_audit": (
                "the released belief is `probs[-1]` after a post-loop append of a freshly reset "
                "product, so any generation ending in '.' scores exactly 1.0. Under that, the "
                "belief-weighted consensus score equals the frequency share, cluster uncertainty is "
                "0, and estimate_conflict divides by zero. Both readings are recorded per call; "
                "`belief_used` says which drove the mechanism"),
            "clustering_note": (
                "the released KMeans leaves random_state=None, so upstream's collaborator selection "
                "is not reproducible between runs. Here k-means is seeded per task id and that is "
                "recorded. The released stopword list is loaded and never applied"),
            "logprobs_requested": True,
            "is_task_adaptation": bench in ADAPTED,
        })


if __name__ == "__main__":
    sys.exit(main())
