"""A benchmark: a fixed set of tasks judged one way."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    """A fixed set of tasks and one judge setup, as mteb's `Benchmark` is a fixed set of tasks. A record belongs to it
    when its task is one of `tasks` and its configuration matches every judge setup field.

    Args:
        name: The name of the benchmark.
        tasks: The names of its tasks.
        judge: The judge's model id, as records store it.
        judging: "pointwise" or "pairwise".
        query_source: "original" or "synthetic".
        n_queries: The number of queries. A task with fewer original queries uses all of them.
        generator: The generator's model id for synthetic queries, as records store it: the model and the settings that
            change its queries.
        judge_revision: The judge's revision. A record matches only if it records the same one.
        seed: The seed queries were sampled or generated with.
        description: What the benchmark measures.
        reference: A link to more about it.
        citation: A BibTeX citation.
    """

    name: str
    tasks: tuple[str, ...]
    judge: str
    judging: str
    query_source: str
    n_queries: int
    generator: str | None = None
    judge_revision: str | None = None
    seed: int = 0
    description: str | None = None
    reference: str | None = None
    citation: str | None = None

    def matches(self, record: dict) -> bool:
        config = record["config"]
        judge = (record.get("llms") or {}).get("judge") or {}
        n = config["n_queries"]
        return (
            record["task_name"] in self.tasks
            and config["judge_model"] == self.judge
            and config["judging"] == self.judging
            and config["query_source"] == self.query_source
            and config.get("generator_model") == self.generator
            and config["seed"] == self.seed
            and (n == self.n_queries or (self.query_source == "original" and n < self.n_queries))
            and (self.judge_revision is None or judge.get("revision") == self.judge_revision)
        )
