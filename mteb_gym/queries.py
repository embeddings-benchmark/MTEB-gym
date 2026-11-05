"""
Synthetic queries. The generator sees a few corpus documents and writes one query of the kind the task describes,
answerable from the corpus but not a restatement of a shown document. Three filters, cheapest first: length, an LLM
quality score (keep >= min_score), and near-duplicate removal by token overlap.
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .llm import ChatClient, llm_settings

logger = logging.getLogger(__name__)


@dataclass
class Query:
    qid: str
    text: str
    seed_doc_ids: list[str]  # documents the query was generated from
    quality: int | None = None  # LLM quality score 1-5, None if unfiltered


_GEN_ROLE = "You write realistic search queries for evaluating retrieval systems.\n"
_GEN_TASK = "The retrieval task is: {task}\n"
_GEN_BODY = (
    "Given a few documents from a corpus, produce ONE query of the kind the retrieval task describes, in the form it "
    "describes: a claim if it gives claims, a topic if it gives topics, a question if it gives questions, a full "
    "problem or case description if it gives those. The query must be answerable using the corpus but must NOT restate "
    "any single shown document, and must not mention the documents. Vary phrasing and specificity. "
    'Reply with strict JSON: {"query": "..."}'
)

_GEN_USER = "Documents:\n{documents}\n\nWrite one query as JSON."
_GEN_DOC_CHARS = 16000  # characters of each document shown to the generator

_FILTER_SYSTEM = (
    "You rate the quality of queries for benchmarking retrieval models. The retrieval task is: {task}\n"
    "A 5 is a clear, specific query of the kind the task describes that could be answered from a corpus and would "
    "separate good retrievers from bad ones. A 1 is vague, trivial, malformed, or refers to shown documents. "
    'Reply with strict JSON: {{"score": 1-5, "reason": "..."}}'
)
_FILTER_USER = "Query: {query}\nReply as JSON."


def extract_json(text: str) -> dict:
    """Last JSON object in a response, ignoring reasoning in think tags. {} if there is none, or the reasoning never
    closed: an answer cut off there has no answer in it, only drafts."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "<think>" in text:
        return {}
    text = re.sub(r"```(?:json)?", "", text)
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
    # A \u escape for half a character decodes to a lone surrogate, which UTF-8 rejects; each becomes U+FFFD.
    text = json.dumps(found, ensure_ascii=False)
    return json.loads(text.encode("utf-16", "surrogatepass").decode("utf-16", "replace"))


class QueryGenerator:
    def __init__(
        self,
        client: ChatClient,
        *,
        task_instruction: str,
        n_queries: int = 100,
        seed: int = 0,
        filter_queries: bool = True,
        workers: int = 16,
        docs_per_query: int = 3,
        overshoot: float = 1.6,
        min_chars: int = 3,
        max_chars: int = 4000,
        min_score: int = 3,
        dedup: float = 0.8,
    ):
        self.client = client
        self.task_instruction = task_instruction
        self.n_queries = n_queries
        self.seed = seed
        self.filter_queries = filter_queries
        self.docs_per_query = docs_per_query
        self.overshoot = overshoot
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_score = min_score
        self.dedup = dedup
        self.system = _GEN_ROLE + _GEN_TASK.format(task=task_instruction) + _GEN_BODY
        self.workers = max(1, workers)
        self.n_generated: int | None = None  # before filtering
        self.settings: dict | None = None  # what the generator ran with

    @property
    def params(self) -> dict:
        """Everything that changes the query set."""
        return {
            "task_instruction": self.task_instruction,
            "n_queries": self.n_queries,
            "seed": self.seed,
            "filter": self.filter_queries,
            "docs_per_query": self.docs_per_query,
            "overshoot": self.overshoot,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_score": self.min_score,
            "dedup": self.dedup,
        }

    @property
    def prompts(self) -> tuple:
        """What the generator and the filter are told: part of the query set's identity."""
        return (self.system, _GEN_USER, _GEN_DOC_CHARS, _FILTER_SYSTEM, _FILTER_USER)

    def run(self, docs: dict[str, str]) -> list[Query]:
        raw = self.generate(docs)
        self.settings = llm_settings(self.client)  # stored with the query set
        self.n_generated = len(raw)
        return self.apply_filters(raw)

    def _one(self, doc_ids: list[str], docs: dict[str, str], idx: int) -> Query | None:
        snippet = "\n\n".join(f"[{i + 1}] {docs[d][:_GEN_DOC_CHARS]}" for i, d in enumerate(doc_ids))
        msg = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": _GEN_USER.format(documents=snippet)},
        ]
        # A client error propagates: a dead endpoint or a bad key fails on the first call. An unusable answer gives
        # None, and the next wave tries fresh documents.
        text = extract_json(self.client.chat(msg)).get("query")
        text = text.strip() if isinstance(text, str) else ""
        return Query(qid=f"q{idx}", text=text, seed_doc_ids=doc_ids) if text else None

    def generate(self, docs: dict[str, str]) -> list[Query]:
        """Order-preserving waves: the seeded RNG draws document samples in the main
        thread and acceptance runs in attempt order, so the query set is identical
        for any worker count."""
        rng = random.Random(self.seed)
        doc_ids = list(docs)
        target = int(self.n_queries * (self.overshoot if self.filter_queries else 1.0))
        cap = target * 3
        out: list[Query] = []
        attempts = 0
        while len(out) < target and attempts < cap:
            wave = min(target - len(out), cap - attempts)  # failed attempts are retried by the next wave
            seeds = [rng.sample(doc_ids, min(self.docs_per_query, len(doc_ids))) for _ in range(wave)]
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

    def _heuristic_ok(self, text: str) -> bool:
        return self.min_chars <= len(text) <= self.max_chars and any(c.isalpha() for c in text)

    def _llm_quality(self, queries: list[Query]) -> None:
        def score(q: Query) -> int:
            """Score in place; 1 if the score had to be defaulted, else 0."""
            msg = [
                {"role": "system", "content": _FILTER_SYSTEM.format(task=self.task_instruction)},
                {"role": "user", "content": _FILTER_USER.format(query=q.text)},
            ]
            try:
                out = extract_json(self.client.chat(msg))
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

    def _deduplicate(self, queries: list[Query]) -> list[Query]:
        """Drop a query whose token set overlaps an earlier one at Jaccard >= dedup."""

        def tokens(q):
            return set(re.findall(r"\w+", q.text.lower()))

        kept: list[Query] = []
        sets: list[set] = []
        for q, t in zip(queries, map(tokens, queries)):
            if all(len(t & s) / len(t | s) < self.dedup for s in sets if t | s):
                kept.append(q)
                sets.append(t)
        return kept

    def apply_filters(self, queries: list[Query]) -> list[Query]:
        if not self.filter_queries:
            return queries[: self.n_queries]
        self._llm_quality(queries)
        good = self._deduplicate([q for q in queries if (q.quality or 0) >= self.min_score])[: self.n_queries]
        for k, q in enumerate(good):  # re-id in kept order
            q.qid = f"q{k}"
        return good
