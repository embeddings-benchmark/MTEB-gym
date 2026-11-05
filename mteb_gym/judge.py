"""
Judges. Pointwise: every pooled document is graded once, 0-5, and models are ranked by nDCG over the grades. Pairwise:
each pair of result lists is judged in both presentation orders and scored fractionally, score_A = mean over the two
orders of {win: 1, tie: 0.5, loss: 0}. Flipping cancels position bias in the score; the record reports it as
a_first_rate (0.5 is unbiased).
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
from openai import APITimeoutError

from .queries import extract_json
from .retrieval import Ranked

if TYPE_CHECKING:
    from .llm import ChatClient, DecisionClient

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


def _reply(client: ChatClient, messages: list[dict]) -> str | None:
    """The judge's answer, or None once a call has timed out through its retries. One stuck call costs one comparison
    or grade, not the run."""
    try:
        return client.chat(messages)
    except (TimeoutError, APITimeoutError, httpx.TimeoutException):
        logger.warning("judge call timed out after its retries; it is left out")
        return None


def judge_system(instruction: str) -> str:
    return _ROLE + _TASK.format(task=instruction) + _BODY + _TAIL


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
    words = text[:n].rsplit(None, 1)
    return (words[0] if len(words) > 1 else text[:n]) + "..."


def _format(r: Ranked, doc_chars: int) -> str:
    """The list as the judge sees it: each document cut to `doc_chars`."""
    return "\n".join(f"  {i + 1}. {_cut(t, doc_chars)}" for i, t in enumerate(r.doc_texts))


def _reasoning(answer: dict) -> str:
    reasoning = answer.get("reasoning", "")
    return reasoning if isinstance(reasoning, str) else ""


def _parse(raw: str) -> tuple[str, str, bool]:
    """(winner, reasoning, parsed_ok). An unparseable answer reads as a tie and is flagged; rate()
    leaves its comparison out."""
    out = extract_json(raw)
    # read as written, up to case and trim: "a", "Tie", "B." and "System A" all mean what they say
    said = str(out.get("winner", "")).strip().rstrip(".").lower().removeprefix("system ").strip()
    winner = {"a": "A", "b": "B", "tie": "tie"}.get(said)
    return (winner or "tie"), _reasoning(out), winner is not None


class PairwiseJudge:
    """Compares two models' result lists for a query, in both presentation orders."""

    EARLY_FAIL_WINDOW = 30
    MAX_EARLY_PARSE_FAIL = 0.5

    def __init__(self, client: ChatClient, instruction: str, workers: int = 1, doc_chars: int = 2000):
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
        reply = _reply(self.client, msg)
        return ("tie", "", False) if reply is None else _parse(reply)

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

    def _early_failure_guard(self, n_done: int, n_failed: int) -> None:
        """Stop before a dead endpoint or a bad key costs a whole run."""
        if n_done == self.EARLY_FAIL_WINDOW and n_failed / n_done > self.MAX_EARLY_PARSE_FAIL:
            raise RuntimeError(
                f"{n_failed / n_done:.0%} of the first {n_done} comparisons had no usable answer: "
                "the judge endpoint is likely dead or the API key invalid."
            )

    def judge_all(
        self, ranked_a: list[Ranked], ranked_b: list[Ranked], model_a: str, model_b: str, on_verdict=None
    ) -> list[Verdict]:
        """Judge every shared query; `on_verdict` is called per verdict as it completes."""
        by_qid = {r.qid: r for r in ranked_b}
        pairs = [(ra, by_qid[ra.qid]) for ra in ranked_a if ra.qid in by_qid]
        guard_lock = threading.Lock()
        counts = {"done": 0, "failed": 0}

        def one(pair):
            v = self.judge_pair(*pair, model_a, model_b)
            with guard_lock:
                counts["done"] += 1
                counts["failed"] += bool(v.parsed_ok) and not any(v.parsed_ok)
                self._early_failure_guard(counts["done"], counts["failed"])
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


def _valid_probabilities(p: dict) -> bool:
    return (
        bool(p)
        and set(p) <= {"0", "1", "2", "3", "4", "5"}
        and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in p.values())
        and math.isclose(sum(p.values()), 1.0, abs_tol=0.01)
    )


def is_decision_model(client) -> bool:
    """A client with decide() and no chat()."""
    return callable(getattr(client, "decide", None)) and not callable(getattr(client, "chat", None))


def grade_state(query: str, document: str, instruction: str, doc_chars: int) -> str:
    return f"Retrieval task: {instruction}\nQuery: {query}\nDocument: {_cut(document, doc_chars)}"


class PointwiseJudge:
    """Grades one (query, document) at a time. A decision model answers in one forward pass with a probability per
    grade; a chat LLM writes a short reason and a grade."""

    def __init__(
        self,
        client: ChatClient | DecisionClient,
        instruction: str,
        workers: int = 8,
        doc_chars: int = 2000,
    ):
        self.client = client
        self.instruction = instruction
        self.workers = max(1, workers)
        self.doc_chars = doc_chars
        self.decision = is_decision_model(client)
        # what the judge is told; part of the grades' cache identity
        self.system = GRADE_QUESTION if self.decision else _GRADE_ROLE + _TASK.format(task=instruction) + _GRADE_TAIL

    def grade(self, qid: str, query: str, doc_id: str, document: str) -> Grade:
        if self.decision:
            p = self.client.decide(
                "score", grade_state(query, document, self.instruction, self.doc_chars), GRADE_QUESTION
            )
            if not _valid_probabilities(p):
                return Grade(qid, doc_id, 0.0, parsed_ok=False)
            return Grade(qid, doc_id, sum(int(k) * v for k, v in p.items()), probabilities=p)
        msg = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument:\n{_cut(document, self.doc_chars)}\n\nReply as JSON.",
            },
        ]
        reply = _reply(self.client, msg)
        if reply is None:
            return Grade(qid, doc_id, 0.0, parsed_ok=False)
        out = extract_json(reply)
        try:
            g = float(out["grade"])
            ok = 0 <= g <= 5
        except (KeyError, TypeError, ValueError):
            g, ok = 0.0, False
        return Grade(qid, doc_id, g, reasoning=_reasoning(out), parsed_ok=ok)

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
