"""
Judges. Pointwise (bottom of the file): every pooled document is graded once, 0-5, and models are ranked by nDCG
over the grades. Pairwise: each pair is judged in both presentation orders and scored
fractionally: score_A = mean over the two orders of {win: 1, tie: 0.5, loss: 0}.
A split across orders is 0.5, a soft tie that still carries magnitude into
Bradley-Terry. Flipping cancels position bias in the score; the record reports
it as a_first_rate (0.5 is unbiased).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .queries import extract_json
from .retrieval import Ranked

logger = logging.getLogger(__name__)

_ROLE = "You compare two retrieval systems.\n"
_TASK = "The retrieval task is: {task}\n"
_BODY = (
    "Given a query and two ranked result sets (System A and System B), decide which set "
    "better satisfies the query, judging relevance, coverage of the information need, and "
    "ranking quality. "
)
_TAIL = (
    'Answer "tie" when the two sets are equally useful. '
    "Reply with strict JSON, reasoning first: "
    '{"reasoning": "one sentence", "winner": "A"|"B"|"tie"}'
)


def task_prompt(prompt) -> str | None:
    """The query prompt from mteb TaskMetadata.prompt, which is a string or a dict of them."""
    return (prompt if isinstance(prompt, str) else (prompt or {}).get("query") or "").strip() or None


def judge_system(instruction: str | None = None) -> str:
    """The judge's instruction. With and without a task description the wording is identical
    apart from the description itself, so a run with one can be compared with a run without."""
    task = _TASK.format(task=instruction) if instruction else ""
    return _ROLE + task + _BODY + _TAIL


_OUTCOME = {"A": 1.0, "tie": 0.5, "B": 0.0}


@dataclass
class Verdict:
    qid: str
    query: str
    model_a: str
    model_b: str
    score_a: float  # 0..1 fractional, after position averaging
    raw: list[str] = field(default_factory=list)  # per-order winners, for audit
    reasoning: str = ""
    parsed_ok: list[bool] = field(default_factory=list)  # per order; empty = no judge call
    # which run it belongs to, so a published verdict is identifiable without its file path
    task: str = ""
    judge: str = ""
    query_set: str = ""


def _cut(text: str, n: int) -> str:
    """The first `n` characters, ending at a word boundary and marked, or the whole text if it fits."""
    if len(text) <= n:
        return text
    head = text[:n]
    return (head.rsplit(None, 1)[0] if " " in head else head) + "..."


def _format(r: Ranked, doc_chars: int) -> str:
    """The list as the judge sees it: each document cut to `doc_chars`."""
    return "\n".join(f"  {i + 1}. {_cut(t, doc_chars)}" for i, t in enumerate(r.doc_texts))


def _parse(raw: str) -> tuple[str, str, bool]:
    """(winner, reasoning, parsed_ok). An unparseable answer reads as a tie and is flagged; rate()
    leaves its comparison out."""
    out = extract_json(raw)
    # read as written, up to case and trim: "a", "Tie", "B." and "System A" all mean what they say
    said = str(out.get("winner", "")).strip().rstrip(".").lower().removeprefix("system ").strip()
    winner = {"a": "A", "b": "B", "tie": "tie"}.get(said)
    return (winner or "tie"), out.get("reasoning", ""), winner is not None


class Judge:
    # Dead-endpoint guards, checked once after the first EARLY_FAIL_WINDOW verdicts.
    EARLY_FAIL_WINDOW = 30
    MAX_EARLY_PARSE_FAIL = 0.5  # above this fraction unparseable: dead judge or bad API key
    MAX_EARLY_IDENTICAL = 0.95  # above this fraction identical lists: empty or mis-loaded corpus

    def __init__(self, client, instruction: str | None = None, workers: int = 1, doc_chars: int = 2000):
        self.client = client
        self.system = judge_system(instruction)
        self.workers = max(1, workers)
        self.doc_chars = doc_chars  # characters of each document shown to the judge

    def _ask(self, query: str, first: Ranked, second: Ranked) -> tuple[str, str, bool]:
        msg = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": f"Query: {query}\n\nSystem A results:\n{_format(first, self.doc_chars)}\n\n"
                f"System B results:\n{_format(second, self.doc_chars)}\n\nReply as JSON.",
            },
        ]
        return _parse(self.client.chat(msg))

    def judge_pair(self, ra: Ranked, rb: Ranked, model_a: str, model_b: str) -> Verdict:
        """Both presentation orders, averaged to a fractional score for A. Identical
        result sets are scored 0.5 without spending judge calls."""
        if ra.doc_ids == rb.doc_ids and ra.doc_texts == rb.doc_texts:
            return Verdict(
                ra.qid,
                ra.query,
                model_a,
                model_b,
                0.5,
                raw=["identical"],
                reasoning="identical result sets; tied without judging",
            )
        w1, why1, ok1 = self._ask(ra.query, ra, rb)
        w2, why2, ok2 = self._ask(ra.query, rb, ra)
        score = (_OUTCOME[w1] + 1.0 - _OUTCOME[w2]) / 2
        return Verdict(
            ra.qid,
            ra.query,
            model_a,
            model_b,
            score,
            raw=[w1, w2],
            reasoning=" | ".join(w for w in (why1, why2) if w),
            parsed_ok=[ok1, ok2],
        )

    def _early_failure_guard(self, n_done: int, n_failed: int, n_identical: int) -> None:
        """Abort loudly instead of scoring a whole run as ties."""
        if n_done != self.EARLY_FAIL_WINDOW:
            return
        if n_failed / n_done > self.MAX_EARLY_PARSE_FAIL:
            raise RuntimeError(
                f"judge parse-failure rate {n_failed / n_done:.0%} over the first {n_done} "
                "verdicts: the judge endpoint is likely dead or the API key invalid."
            )
        if n_identical / n_done > self.MAX_EARLY_IDENTICAL:
            raise RuntimeError(
                f"{n_identical / n_done:.0%} of the first {n_done} pairs had identical retrieved "
                "lists, so the judge was never called: the corpus is likely empty, "
                "mis-loaded, or smaller than top_k."
            )

    def judge_all(
        self, ranked_a: list[Ranked], ranked_b: list[Ranked], model_a: str, model_b: str, on_verdict=None
    ) -> list[Verdict]:
        """Judge every shared query; `on_verdict` is called per verdict as it completes."""
        by_qid = {r.qid: r for r in ranked_b}
        pairs = [(ra, by_qid[ra.qid]) for ra in ranked_a if ra.qid in by_qid]
        guard_lock = threading.Lock()
        counts = {"done": 0, "failed": 0, "identical": 0}

        def one(pair):
            v = self.judge_pair(*pair, model_a, model_b)
            with guard_lock:
                counts["done"] += 1
                counts["failed"] += bool(v.parsed_ok) and not any(v.parsed_ok)
                counts["identical"] += v.raw == ["identical"]
                self._early_failure_guard(counts["done"], counts["failed"], counts["identical"])
            if on_verdict is not None:
                on_verdict(v)
            return v

        def collect(results):
            out = []
            for i, v in enumerate(results, 1):
                out.append(v)
                if i % 25 == 0:
                    logger.info("%s vs %s: %d/%d queries judged", model_a, model_b, i, len(pairs))
            return out

        if self.workers <= 1 or len(pairs) <= 1:
            return collect(map(one, pairs))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:  # map keeps input order
            return collect(pool.map(one, pairs))


# ------------------------------------------------------------------------------------------- pointwise
# Each document in the pool (every model's top k for a query) is graded once, 0-5, and each model is ranked by nDCG@k
# over the grades. A decision model answers the grade question in one forward pass with a calibrated probability per
# grade; a chat LLM writes a short reason and a grade.

GRADE_QUESTION = (
    "How relevant is the document to the query, for this retrieval task? 0 = not relevant at all, "
    "5 = perfectly relevant: it fully satisfies the information need."
)
_GRADE_ROLE = "You grade how relevant a document is to a search query.\n"
_GRADE_TAIL = (
    "Given a query and one document, grade the document from 0 (not relevant at all) to 5 (perfectly relevant: it "
    'fully satisfies the information need). Reply with strict JSON, reasoning first: {"reasoning": "one sentence", '
    '"grade": 0-5}'
)


@dataclass
class Grade:
    qid: str
    doc_id: str
    grade: float  # 0..5; the expected grade under a decision model's probabilities
    probabilities: dict[str, float] | None = None  # decision model only
    reasoning: str = ""  # chat LLM only
    parsed_ok: bool = True  # an unparseable answer is left out of the ranking
    # which run it belongs to, as on a Verdict
    task: str = ""
    judge: str = ""
    query_set: str = ""


def is_decision_model(client) -> bool:
    return callable(getattr(client, "decide", None)) and not callable(getattr(client, "chat", None))


def grade_state(query: str, document: str, instruction: str | None, doc_chars: int) -> str:
    task = f"Retrieval task: {instruction}\n" if instruction else ""
    return f"{task}Query: {query}\nDocument: {_cut(document, doc_chars)}"


class PointwiseJudge:
    """Grades one (query, document) at a time. Uses `client.decide` when the client is a decision model, else the
    client's chat endpoint with a JSON answer."""

    def __init__(
        self,
        client,
        instruction: str | None = None,
        workers: int = 8,
        doc_chars: int = 2000,
        decision: bool | None = None,
    ):
        self.client = client
        self.instruction = instruction
        self.workers = max(1, workers)
        self.doc_chars = doc_chars
        # a client with decide() and no chat() is a decision model; `decision` overrides, e.g. for MockLLM
        self.decision = is_decision_model(client) if decision is None else decision
        task = _TASK.format(task=instruction) if instruction else ""
        # what the judge is told; part of the grades' cache identity
        self.system = GRADE_QUESTION if self.decision else _GRADE_ROLE + task + _GRADE_TAIL

    def grade(self, qid: str, query: str, doc_id: str, document: str) -> Grade:
        if self.decision:
            p = self.client.decide(
                "score", grade_state(query, document, self.instruction, self.doc_chars), GRADE_QUESTION
            )
            return Grade(qid, doc_id, sum(int(k) * v for k, v in p.items()), probabilities=p)
        msg = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument:\n{_cut(document, self.doc_chars)}\n\nReply as JSON.",
            },
        ]
        out = extract_json(self.client.chat(msg))
        try:
            g = float(out["grade"])
            ok = 0 <= g <= 5
        except (KeyError, TypeError, ValueError):
            g, ok = 0.0, False
        return Grade(qid, doc_id, g, reasoning=out.get("reasoning", ""), parsed_ok=ok)

    def grade_all(self, items: list[tuple[str, str, str, str]], on_grade=None) -> list[Grade]:
        """`items` are (qid, query, doc_id, document); `on_grade` is called per grade as it completes."""

        def one(item):
            g = self.grade(*item)
            if on_grade is not None:
                on_grade(g)
            return g

        if self.workers <= 1 or len(items) <= 1:
            return list(map(one, items))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, items))
