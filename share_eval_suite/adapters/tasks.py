"""Task adapters: question answering, text-to-SQL and entity matching.

A task adapter builds the input a workflow receives and scores the answer it returns. Every function
here works on in-memory values: answer texts, references, query result rows, pair decisions. Scoring
a SQL query against the database is done by executing it elsewhere; here two result row lists are
compared, the step the execution scorer applies after execution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

QUESTION_ANSWERING = "question answering"
TEXT_TO_SQL = "text-to-SQL"
ENTITY_MATCHING = "entity matching"

#: answer kind per QA benchmark (the kinds `qa_answer` reads)
QA_KINDS = {"gsm8k": "numeric", "strategyqa": "boolean", "mmlu_pro": "mcq", "gpqa": "mcq", "squad_v2": "span"}


# ------------------------------------------------------------------ benchmark scorers
#: a number with optional sign, thousands separators and fractional part
_NUM = re.compile(r"[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?")
#: GSM8K references give the rationale first and the answer after `####`
_GSM8K_GOLD = re.compile(r"####\s*([-+]?\d(?:[\d,]*\d)?(?:\.\d+)?)")
#: a standalone uppercase option letter, optionally in parentheses; a letter inside a formula
#: (after `\ { _ ^` or before `=`) or inside a word does not count
_LETTER = re.compile(r"(?<![A-Za-z\\{_^])\(?([A-J])\)?(?![A-Za-z]|\s*=)")
_YES = ("yes", "true", "correct")
_NO = ("no", "false", "incorrect")
_NO_ANSWER = ("", "no answer", "unanswerable", "none", "not answerable", "cannot be answered")


def _to_number(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", "").replace("+", ""))
    except ValueError:
        return None


def _last_number(text: Any) -> Optional[float]:
    """The last numeric token (currency signs and bold markup removed); no characters are stripped
    from the token itself."""
    if text is None:
        return None
    s = str(text).replace("$", "").replace("*", "").replace("−", "-")
    for tok in reversed(_NUM.findall(s)):
        n = _to_number(tok)
        if n is not None:
            return n
    return None


def _gsm8k_gold(reference: Any) -> Optional[float]:
    """The number after `####` in a GSM8K reference; `None` when the marker is absent."""
    if reference is None:
        return None
    m = _GSM8K_GOLD.search(str(reference))
    return _to_number(m.group(1)) if m else None


def _gsm8k_correct(delivered: Any, reference: Any, tol: float = 1e-6) -> Optional[bool]:
    gold, pred = _gsm8k_gold(reference), _last_number(delivered)
    if gold is None or pred is None:
        return None
    return abs(gold - pred) <= tol


def _letter_answer(text: Any, n_options: int = 10) -> Optional[str]:
    """The last standalone option letter within the question's `n_options` options."""
    if text is None:
        return None
    allowed = {chr(ord("A") + i) for i in range(max(1, min(10, n_options)))}
    found = [m.group(1) for m in _LETTER.finditer(str(text)) if m.group(1) in allowed]
    return found[-1] if found else None


def _choice_correct(delivered: Any, gold_letter: Any, n_options: int = 10) -> Optional[bool]:
    pred = _letter_answer(delivered, n_options)
    gold = str(gold_letter).strip().upper() if gold_letter is not None else None
    if pred is None or not gold or gold not in {chr(ord("A") + i) for i in range(10)}:
        return None
    return pred == gold


def _boolean_answer(text: Any) -> Optional[bool]:
    """The last standalone yes/no word; `None` when the reply commits to neither."""
    if text is None:
        return None
    for tok in reversed(re.findall(r"[A-Za-z]+", str(text).lower())):
        if tok in _YES:
            return True
        if tok in _NO:
            return False
    return None


def _strategyqa_correct(delivered: Any, reference: Any) -> Optional[bool]:
    pred = _boolean_answer(delivered)
    g = str(reference).strip().lower()
    gold = True if g in _YES else False if g in _NO else None
    if pred is None or gold is None:
        return None
    return pred is gold


def _squad_norm(s: str) -> str:
    """SQuAD normalisation: lowercase, articles, punctuation and extra whitespace removed."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


# ------------------------------------------------------------------ SQL and matching readers
#: last code fence of a reply (language tag optional)
_SQL_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.S)
#: first code fence of a reply, any letter case of the tag
_STATEMENT_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)
#: polarity words, longest first so that "non-match" is read before "match"
_POLAR = re.compile(
    r"\b(non-?match(?:es)?|match(?:es|ed|ing)?|same|identical|equivalent|duplicates?|"
    r"different|distinct|unrelated)\b", re.I)
_NEG_CUE = re.compile(
    r"\b(?:not|n't|no|never|cannot|can't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"nor|neither|without|fails? to|unlikely to|rather than)\b", re.I)
_NEGATIVE_WORDS = {"different", "distinct", "unrelated"}
_LOOKBACK_WORDS = 5


def _coerce(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, float):
        return round(v, 6)
    return v


def _norm_result(rows: Sequence[Sequence[Any]]) -> set:
    return {tuple(_coerce(x) for x in r) for r in rows}


# ------------------------------------------------------------------ question answering
def qa_question(benchmark: str, question: str, context: Optional[str] = None) -> str:
    """The question as the agents receive it; SQuAD 2.0 questions carry their passage first."""
    if benchmark == "squad_v2" and context:
        return f"{context}\n\nQuestion: {question}"
    return question


def qa_answer(text: Optional[str], kind: str) -> Optional[str]:
    """The answer read from a reply (`numeric`, `mcq`, `boolean` or `span`): the text after `Answer:`,
    normalised per kind (a span is the text with its whitespace collapsed); `None` when there is
    nothing to read."""
    t = text or ""
    tail = t.split("Answer:")[-1] if "Answer:" in t else t
    if kind == "numeric":
        m = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", tail)
        if not m:
            return None
        try:
            return str(round(float(m[-1].replace(",", "")), 4))
        except ValueError:
            return None
    if kind == "mcq":
        m = re.search(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", tail.upper())
        return m.group(1) if m else None
    if kind == "boolean":
        for tok in re.findall(r"[a-z]+", tail.strip().lower()):
            if tok in ("yes", "true"):
                return "yes"
            if tok in ("no", "false"):
                return "no"
        return None
    s = " ".join(tail.split())
    return s or None


def qa_correct(benchmark: str, text: Optional[str], reference: Any, n_options: int = 10) -> int:
    """1 when the answer read from `text` matches the benchmark reference by the benchmark's own rule,
    else 0. An unreadable answer scores 0.

    `reference` is the GSM8K reference string (`... #### N`), the StrategyQA target (`yes`/`no`,
    `true`/`false`), the option letter (MMLU-Pro, GPQA, with `n_options` options) or the list of SQuAD
    2.0 reference answers (empty or `["<unanswerable>"]` for an unanswerable question). A SQuAD 2.0
    answer is correct on exact match (`squad_scores`); an unanswerable question is answered by
    abstaining.
    """
    kind = QA_KINDS.get(benchmark, "mcq")
    ans = qa_answer(text, kind)
    if benchmark == "squad_v2":
        refs = [str(x) for x in (reference or [])]
        unanswerable = not [x for x in refs if x != "<unanswerable>"]
        s = squad_scores(ans or "", refs, unanswerable)
        return int(bool(s and s["exact"] == 1.0))
    if ans is None or not str(ans).strip():
        return 0
    if benchmark == "gsm8k":
        return int(bool(_gsm8k_correct(ans, reference)))
    if benchmark == "strategyqa":
        return int(bool(_strategyqa_correct(ans, reference)))
    return int(bool(_choice_correct(ans, reference, int(n_options or 10))))


def qa_official_correct(benchmark: str, delivered: Any, reference: Any,
                        n_options: int = 10) -> Optional[bool]:
    """The benchmark's own scorer on a delivered answer: `True`/`False`, or `None` when the comparison
    is undefined (no reference, or no answer in the text)."""
    if benchmark == "gsm8k":
        return _gsm8k_correct(delivered, reference)
    if benchmark == "strategyqa":
        return _strategyqa_correct(delivered, reference)
    return _choice_correct(delivered, reference, n_options)


def squad_scores(delivered: Any, references: Sequence[str], unanswerable: bool) -> Optional[Dict[str, Any]]:
    """SQuAD 2.0 exact match and token F1, with an unanswerable question answered by abstaining."""
    if delivered is None:
        return None
    pred = str(delivered).strip()
    said_none = _squad_norm(pred) in _NO_ANSWER
    if unanswerable:
        return {"exact": float(said_none), "f1": float(said_none), "mode": "no-answer"}
    golds = [str(x) for x in (references or []) if str(x) != "<unanswerable>"]
    if not golds:
        return None
    pn = _squad_norm(pred)
    best = {"exact": 0.0, "f1": 0.0, "mode": "span"}
    for g in golds:
        gn = _squad_norm(g)
        exact = float(pn == gn)
        pt, gt = pn.split(), gn.split()
        # tokens shared by answer and reference, each reference token used once
        common, pool = 0, list(gt)
        for t in pt:
            if t in pool:
                pool.remove(t)
                common += 1
        f1 = 0.0 if not common else (2 * (common / len(pt)) * (common / len(gt))
                                     / ((common / len(pt)) + (common / len(gt))))
        if (exact, f1) > (best["exact"], best["f1"]):
            best = {"exact": exact, "f1": f1, "mode": "span"}
    return best


# ------------------------------------------------------------------ text-to-SQL
def sql_question(question: str, schema: str, evidence: str = "") -> str:
    """The question with its schema, as the agents receive it (no reference query, no answer format)."""
    ev = f"\n\nAdditional context: {evidence}" if evidence else ""
    return ("Write a single SQL query that answers the question for the given database schema.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}{ev}\n\n"
            "The answer is one SQL query for this schema.")


def sql_answer(text: Any) -> str:
    """The query in a reply: last code fence, language tag, trailing semicolon and extra whitespace
    removed; case kept."""
    t = str(text or "").strip()
    if not t:
        return ""
    m = _SQL_FENCE.findall(t)
    if m:
        t = m[-1]
    t = re.sub(r"^(?:sql|SQL)\s*[\n:]", "", t.strip()).strip()
    t = t.rstrip(";").strip()
    return " ".join(t.split())


def sql_statement(text: str) -> Optional[str]:
    """One SELECT/WITH statement read from a reply, or `None` when nothing SQL-like is present."""
    if not text:
        return None
    m = _STATEMENT_FENCE.search(text)
    s = (m.group(1) if m else text).strip()
    s = re.sub(r"^(?:sql|sqlite)\s*[:\n]", "", s, flags=re.I).strip()
    i = s.lower().find("select")
    if i < 0:
        i = s.lower().find("with")
    if i < 0:
        return None
    s = s[i:].strip().rstrip(";").strip()
    return s or None


def sql_results_match(predicted_rows: Iterable[Sequence[Any]], reference_rows: Iterable[Sequence[Any]]) -> bool:
    """Execution match: the two result row lists are equal as sets of tuples (floats rounded to six
    decimals, binary values by length)."""
    return _norm_result(list(predicted_rows)) == _norm_result(list(reference_rows))


def sql_execution_vote(candidates: Sequence[str],
                       results: Sequence[Tuple[str, Optional[list]]]) -> Optional[str]:
    """Self-consistency over candidate queries: group the executed queries by their result rows (a
    timed-out query counts as an empty result) and return the first query of the largest group;
    queries that end in an error do not vote."""
    groups: Dict[str, List[str]] = {}
    for sql, (status, rows) in zip(candidates, results):
        if status == "error":
            continue
        # a timed-out query groups with the queries that return no rows
        key = repr(rows if status == "ok" else [])
        groups.setdefault(key, []).append(sql)
    if not groups:
        return None
    largest = max(groups, key=lambda k: len(groups[k]))
    return groups[largest][0]


def sql_first_valid(candidates: Sequence[str], results: Sequence[Tuple[str, Optional[list]]],
                    nonempty: bool = False) -> Optional[str]:
    """The first candidate that executes (and, with `nonempty`, returns at least one row)."""
    for sql, (status, rows) in zip(candidates, results):
        if status == "ok" and (rows or not nonempty):
            return sql
    return None


# ------------------------------------------------------------------ entity matching
def matching_question(task: str, record_a: Mapping[str, Any], record_b: Mapping[str, Any],
                      fields: Optional[Sequence[str]] = None) -> str:
    """The candidate pair as the agents see it: the task line and the two records' fields."""
    def block(tag: str, r: Mapping[str, Any]) -> str:
        keys = fields if fields else [k for k in r if r.get(k) not in (None, "")]
        lines = [f"  {k}: {r.get(k, '')}" for k in keys if r.get(k) not in (None, "")]
        return f"Record {tag}:\n" + "\n".join(lines)

    return (f"{task}\n\n{block('A', dict(record_a))}\n\n{block('B', dict(record_b))}\n\n"
            "The answer is either match or non-match.")


def matching_label(text: Any) -> str:
    """`match`, `non-match`, or `""` when the reply commits to neither (last polarity word, with a
    negation window before it)."""
    t = str(text if text is not None else "").strip()
    if not t:
        return ""
    hits = list(_POLAR.finditer(t))
    if not hits:
        return ""
    m = hits[-1]
    word = m.group(1).lower()
    positive = not (word.startswith("non") or word in _NEGATIVE_WORDS)
    # a negation among the five words before the polarity word reverses it
    before = t[max(0, m.start() - 120):m.start()]
    if _NEG_CUE.search(" ".join(before.split()[-_LOOKBACK_WORDS:])):
        positive = not positive
    return "match" if positive else "non-match"


def pair_scores(predictions: Sequence[Optional[str]], gold: Sequence[str]) -> Dict[str, float]:
    """Pair F1 on the match class (percent), recall (percent) and the counts tp, fp, fn, n.

    `predictions` and `gold` are aligned labels; an unreadable prediction counts as a non-match.
    """
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold differ in length")
    pred = [p or "" for p in predictions]
    tp = sum(1 for p, g in zip(pred, gold) if p == "match" and g == "match")
    fp = sum(1 for p, g in zip(pred, gold) if p == "match" and g != "match")
    fn = sum(1 for p, g in zip(pred, gold) if p != "match" and g == "match")
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=100 * (2 * p * rc / (p + rc) if p + rc else 0.0), recall=100 * rc,
                tp=tp, fp=fp, fn=fn, n=len(pred))


def cluster_f1(true_relations: int, predicted_relations: int, gold_relations: int) -> float:
    """Cluster F1 from relation counts, as a fraction: 2T / (N + G), with T the predicted relations
    that are true, N the predicted relations and G the true relations."""
    T, N, G = true_relations, predicted_relations, gold_relations
    return (2 * T / (N + G)) if (N + G) else 0.0


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class TaskAdapter:
    """One task family: its benchmarks, how its input is built, how an answer is read and scored."""
    name: str
    benchmarks: Tuple[str, ...]
    build_input: Any
    read_answer: Any
    score: Any
    sources: Tuple[str, ...]


TASKS: Dict[str, TaskAdapter] = {
    QUESTION_ANSWERING: TaskAdapter(
        QUESTION_ANSWERING, ("gsm8k", "strategyqa", "mmlu_pro", "gpqa", "squad_v2"),
        qa_question, qa_answer, qa_correct,
        ("code/consensus_delivery/adapters/official_scorers.py", "code/consensus_delivery/adapters/mad_m2.py",
         "experiments/15_stronger_peers/code/w3_score.py")),
    TEXT_TO_SQL: TaskAdapter(
        TEXT_TO_SQL, ("bird", "spider"), sql_question, sql_answer, sql_results_match,
        ("experiments/02_controlled_sharing/code/sql_tasks.py", "experiments/06_cost_resources/code/sql_common.py",
         "experiments/common/code/adapters/chess.py")),
    ENTITY_MATCHING: TaskAdapter(
        ENTITY_MATCHING, ("wdc_products", "geographic_settlements", "musicbrainz20k", "amazon_google", "dblp_acm"),
        matching_question, matching_label, pair_scores,
        ("experiments/02_controlled_sharing/code/matching_tasks.py",
         "experiments/02_controlled_sharing/code/answer_extract.py", "experiments/common/code/core.py")),
}


def list_tasks() -> List[TaskAdapter]:
    """The registered task families."""
    return list(TASKS.values())


def get_task(name: str) -> TaskAdapter:
    """The task family called `name`."""
    for k, t in TASKS.items():
        if name.strip().lower() in (k.lower(), k.lower().replace(" ", "_"), k.lower().replace("-", "_")):
            return t
    raise KeyError(f"unknown task {name!r}; known: {', '.join(TASKS)}")
