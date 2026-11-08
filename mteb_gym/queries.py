"""
Synthetic queries. The generator sees a few corpus documents and writes one query, answerable from the corpus but not a
restatement of a shown document. A length check always applies. An optional filter adds an LLM quality score (keep >=
min_score) and near-duplicate removal by token overlap.
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .llm import ChatClient, llm_settings
from .progress import progress_bar

logger = logging.getLogger(__name__)


@dataclass
class Query:
    qid: str
    text: str
    seed_doc_ids: list[str]  # documents the query was generated from
    quality: int | None = None  # LLM quality score 1-5, None if unfiltered


DOC_CHARS = 16000  # characters of each document shown to the generator and, by default, to a pointwise judge

_GEN_ROLE = "You write realistic search queries for evaluating retrieval systems.\n"
_GEN_BODY = (
    "Given a few documents from a corpus, produce ONE natural query a real user might type that is answerable using "
    "the corpus but is NOT a restatement of any single shown document. Vary phrasing and specificity. "
    'Reply with strict JSON: {"query": "..."}'
)

_GEN_BODY_TASK = (
    "Given a few documents from a corpus, produce ONE query of the kind the retrieval task describes. The query must "
    "be answerable using the corpus but must NOT restate any single shown document, and must not mention the "
    'documents. Vary phrasing and specificity. Reply with strict JSON: {"query": "..."}'
)

_GEN_USER = "Documents:\n{documents}\n\nWrite one query as JSON."

_FILTER_ROLE = "You rate the quality of search queries for benchmarking retrieval models.\n"
_FILTER_BODY = (
    "A 5 is a clear, specific, genuinely answerable information need that would separate good retrievers from bad "
    "ones. A 1 is vague, trivial, malformed, or answerable by exact keyword match. "
    'Reply with strict JSON: {"score": 1-5, "reason": "..."}'
)
_FILTER_USER = "Query: {query}\nReply as JSON."


def task_line(instruction: str | None) -> str:
    """The line of a system prompt that names the task; nothing when there is no task sentence."""
    return f"The retrieval task is: {instruction}\n" if instruction else ""


def extract_json(text: str) -> dict:
    """The last JSON object of an answer, read only after any reasoning. {} if there is none, if the reasoning never
    closed, or if the answer was cut off after its last complete object."""
    text = re.split(r"</think(?:ing)?>", text or "")[-1]
    if re.search(r"<think(?:ing)?>", text):
        return {}
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    found: dict = {}
    cut_off = False
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            cut_off = True
            i = text.find("{", i + 1)
            continue
        cut_off = False
        if isinstance(obj, dict):
            found = obj
        i = text.find("{", end)
    if cut_off:
        return {}
    # A \u escape for half a character decodes to a lone surrogate, which UTF-8 rejects; each becomes U+FFFD.
    text = json.dumps(found, ensure_ascii=False)
    return json.loads(text.encode("utf-16", "surrogatepass").decode("utf-16", "replace"))


class QueryGenerator:
    def __init__(
        self,
        client: ChatClient,
        *,
        task_instruction: str | None = None,
        n_queries: int = 100,
        seed: int = 0,
        filter_queries: bool = False,
        workers: int = 16,
        docs_per_query: int = 3,
        overshoot: float = 1.6,
        min_chars: int = 3,
        max_chars: int = 4000,
        min_score: int = 3,
        dedup: float = 0.8,
        doc_chars: int = DOC_CHARS,
        show_progress_bar: bool = False,
    ):
        self.client = client
        self.show_progress_bar = show_progress_bar  # not part of the query set
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
        self.doc_chars = doc_chars
        self.system = _GEN_ROLE + (task_line(task_instruction) + _GEN_BODY_TASK if task_instruction else _GEN_BODY)
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
            "doc_chars": self.doc_chars,
        }

    @property
    def prompts(self) -> tuple:
        """What the generator and the filter are told: part of the query set's identity."""
        return (self.system, _GEN_USER, self._filter_system, _FILTER_USER)

    @property
    def _filter_system(self) -> str:
        return _FILTER_ROLE + task_line(self.task_instruction) + _FILTER_BODY

    def run(self, docs: dict[str, str]) -> list[Query]:
        raw = self.generate(docs)
        self.settings = llm_settings(self.client)  # stored with the query set
        self.n_generated = len(raw)
        return self.apply_filters(raw)

    def _one(self, doc_ids: list[str], docs: dict[str, str], idx: int) -> Query | None:
        snippet = "\n\n".join(f"[{i + 1}] {docs[d][: self.doc_chars]}" for i, d in enumerate(doc_ids))
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
        bar = progress_bar("Writing queries", target, self.show_progress_bar, unit="query")
        while len(out) < target and attempts < cap:
            wave = min(target - len(out), cap - attempts)  # failed attempts are retried by the next wave
            seeds = [rng.sample(doc_ids, min(self.docs_per_query, len(doc_ids))) for _ in range(wave)]
            base, attempts = attempts, attempts + wave
            with ThreadPoolExecutor(max_workers=self.workers if wave > 1 else 1) as pool:
                for q in pool.map(lambda ks: self._one(ks[1], docs, base + ks[0]), enumerate(seeds)):  # in order
                    if q and self._heuristic_ok(q.text):
                        q.qid = f"q{len(out)}"  # qid = acceptance order
                        out.append(q)
                        bar.update()
            logger.info("query generation: %d/%d kept (%d attempts)", len(out), target, attempts)
        bar.close()
        return out

    def _heuristic_ok(self, text: str) -> bool:
        return self.min_chars <= len(text) <= self.max_chars and any(c.isalpha() for c in text)

    def _llm_quality(self, queries: list[Query]) -> None:
        def score(q: Query) -> int:
            """Set the query's quality; 1 if it had to be defaulted, else 0."""
            msg = [
                {"role": "system", "content": self._filter_system},
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
