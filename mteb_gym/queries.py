"""
Synthetic queries. The generator sees a few corpus documents and writes one
realistic query answerable from the corpus but not a restatement of a shown
document. Filtering is the biggest lever on agreement with ground truth; three
gates, cheapest first: length/degeneracy heuristics, an LLM quality score
(keep >= min_score), near-duplicate removal by token overlap.
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .llm import llm_settings

logger = logging.getLogger(__name__)


@dataclass
class Query:
    qid: str
    text: str
    seed_doc_ids: list[str]  # documents the query was generated from
    quality: int | None = None  # LLM quality score 1-5, None if unfiltered


_GEN_ROLE = "You write realistic search queries for evaluating retrieval systems.\n"
_GEN_TASK = "The retrieval task is: {task}\nEvery query must express that kind of need.\n"
_GEN_BODY = (
    "Given a few documents from a corpus, produce ONE natural query a real user "
    "might type that is answerable using the corpus but is NOT a restatement of "
    "any single shown document. Vary phrasing and specificity. "
    'Reply with strict JSON: {"query": "..."}'
)

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}
_FILTER_SYSTEM = (
    "You rate the quality of search queries for benchmarking retrieval models. "
    "A 5 is a clear, specific, genuinely answerable information need that would "
    "separate good retrievers from bad ones. A 1 is vague, trivial, malformed, "
    "or answerable by exact keyword match. "
    'Reply with strict JSON: {"score": 1-5, "reason": "..."}'
)


def extract_json(text: str) -> dict:
    """Last JSON object in a response (models add prose, fences, reasoning)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    decoder = json.JSONDecoder()
    found: dict = {}
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found = obj
        i = text.find("{", end)
    return found


class QueryGenerator:
    def __init__(
        self,
        client,
        *,
        task_description: str | None = None,
        n_queries: int = 100,
        seed: int = 0,
        filter: bool = True,
        workers: int = 16,
        docs_per_query: int = 3,
        overshoot: float = 1.6,
        min_chars: int = 15,
        max_chars: int = 240,
        min_score: int = 3,
        dedup: float = 0.8,
    ):
        self.client = client
        self.params = dict(
            task_description=task_description,
            n_queries=n_queries,
            seed=seed,
            filter=filter,
            docs_per_query=docs_per_query,
            overshoot=overshoot,
            min_chars=min_chars,
            max_chars=max_chars,
            min_score=min_score,
            dedup=dedup,
        )  # everything that changes the query set
        task = _GEN_TASK.format(task=task_description) if task_description else ""
        self.system = _GEN_ROLE + task + _GEN_BODY
        self.workers = max(1, workers)
        self.n_generated: int | None = None  # pre-filter count, for the record
        self.settings: dict | None = None  # what the generator ran with, for the record

    def run(self, docs: dict[str, str]) -> list[Query]:
        raw = self.generate(docs)
        self.settings = llm_settings(self.client)  # generation's settings, before the filter reuses the client at 0
        self.n_generated = len(raw)
        return self.filter(raw)

    # ---------------------------------------------------------------- generate
    def _one(self, doc_ids: list[str], docs: dict[str, str], idx: int) -> Query | None:
        snippet = "\n\n".join(f"[{i + 1}] {docs[d][:600]}" for i, d in enumerate(doc_ids))
        msg = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Documents:\n{snippet}\n\nWrite one query as JSON."},
        ]
        # A client error propagates (the client has already retried): a dead endpoint or a bad
        # key fails on the first call, not after minutes of empty waves. An unparseable answer
        # yields None and the next wave retries with fresh documents.
        text = (extract_json(self.client.chat(msg, schema=_QUERY_SCHEMA)).get("query") or "").strip()
        return Query(qid=f"q{idx}", text=text, seed_doc_ids=doc_ids) if text else None

    def generate(self, docs: dict[str, str]) -> list[Query]:
        """Order-preserving waves: the seeded RNG draws document samples in the main
        thread and acceptance runs in attempt order, so the query set is identical
        for any worker count."""
        p = self.params
        rng = random.Random(p["seed"])
        doc_ids = list(docs)
        target = int(p["n_queries"] * (p["overshoot"] if p["filter"] else 1.0))
        cap = target * 3
        out: list[Query] = []
        attempts = 0
        while len(out) < target and attempts < cap:
            wave = min(target - len(out), cap - attempts)  # failed attempts are retried by the next wave
            seeds = [rng.sample(doc_ids, min(p["docs_per_query"], len(doc_ids))) for _ in range(wave)]
            base, attempts = attempts, attempts + wave
            if self.workers <= 1 or wave <= 1:
                results = [self._one(s, docs, base + k) for k, s in enumerate(seeds)]
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    results = list(pool.map(lambda ks: self._one(ks[1], docs, base + ks[0]), enumerate(seeds)))
            for q in results:
                if q and self._heuristic_ok(q.text):
                    q.qid = f"q{len(out)}"  # qid = acceptance order
                    out.append(q)
            logger.info("query generation: %d/%d kept (%d attempts)", len(out), target, attempts)
        return out

    # ----------------------------------------------------------------- filters
    def _heuristic_ok(self, text: str) -> bool:
        p = self.params
        return (
            p["min_chars"] <= len(text) <= p["max_chars"]
            and text.count(" ") >= 2  # one or two words: too trivial
            and re.search(r"[A-Za-z]", text) is not None
        )

    def _llm_quality(self, queries: list[Query]) -> None:
        def score(q: Query) -> int:
            """Score in place; return 1 if the score had to be defaulted."""
            msg = [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": f"Query: {q.text}\nReply as JSON."},
            ]
            try:
                out = extract_json(self.client.chat(msg, schema=_SCORE_SCHEMA))
                q.quality = int(out.get("score", 3))
                return 0 if "score" in out else 1
            except (ValueError, TypeError):  # non-numeric score
                q.quality = 3
                return 1

        if self.workers <= 1 or len(queries) <= 1:
            failures = sum(score(q) for q in queries)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                failures = sum(pool.map(score, queries))
        if failures:
            logger.warning(
                "quality filter: %d/%d scores defaulted to 3 (parse/call failure); "
                "a misbehaving filter model can silently disable filtering",
                failures,
                len(queries),
            )

    def _dedup(self, queries: list[Query]) -> list[Query]:
        """Drop a query whose token set overlaps an earlier one at Jaccard >= dedup."""

        def tokens(q):
            return set(re.findall(r"[a-z0-9]+", q.text.lower()))

        kept: list[Query] = []
        sets: list[set] = []
        for q, t in zip(queries, map(tokens, queries)):
            if all(len(t & s) / len(t | s) < self.params["dedup"] for s in sets if t | s):
                kept.append(q)
                sets.append(t)
        return kept

    def filter(self, queries: list[Query]) -> list[Query]:
        n = self.params["n_queries"]
        if not self.params["filter"]:
            return queries[:n]
        self._llm_quality(queries)
        good = self._dedup([q for q in queries if (q.quality or 0) >= self.params["min_score"]])[:n]
        for k, q in enumerate(good):  # re-id in kept order
            q.qid = f"q{k}"
        return good
