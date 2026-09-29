"""
Pairwise judge. Each pair is judged in both presentation orders and scored
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
_WINNERS = ("A", "B", "tie")
_VERDICT_SCHEMA = {  # field order is generation order: the judge reasons before it commits
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "winner": {"type": "string", "enum": list(_WINNERS)}},
    "required": ["reasoning", "winner"],
    "additionalProperties": False,
}


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
    """(winner, reasoning, parsed_ok). An unparseable answer scores as a tie but is flagged."""
    out = extract_json(raw)
    winner = out.get("winner")
    ok = winner in _WINNERS
    return (winner if ok else "tie"), out.get("reasoning", ""), ok


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
        return _parse(self.client.chat(msg, schema=_VERDICT_SCHEMA))

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
