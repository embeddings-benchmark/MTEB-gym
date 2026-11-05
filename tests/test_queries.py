import tempfile
import types
from pathlib import Path

from mteb_gym.llm import MockLLM
from mteb_gym.queries import QueryGenerator, extract_json
from mteb_gym.run import resolve_queries
from tests.helpers import make_corpus


def test_extract_json():
    from mteb_gym.judge import _parse

    for said, want in [("A", "A"), ("a", "A"), ("Tie", "tie"), ("B.", "B"), ("System A", "A")]:
        assert _parse(f'{{"reasoning": "r", "winner": "{said}"}}')[::2] == (want, True)
    assert _parse('{"winner": "C"}')[2] is False and _parse("")[2] is False
    # a server without a reasoning parser returns the thinking inline, before the answer
    inline = (
        '<think>A looks better, maybe {"winner": "B"}? No: A covers more.</think>\n{"reasoning": "r", "winner": "A"}'
    )
    assert extract_json(inline)["winner"] == "A"
    assert extract_json('```json\n{"winner": "A"}\n```')["winner"] == "A"
    assert extract_json('blah {"score": 4} trailing')["score"] == 4
    assert extract_json("not json") == {}
    assert extract_json('Let me think {"winner": "B"} ... final {"winner": "A"}')["winner"] == "A"  # last object wins
    assert extract_json('{"reasoning": "set {x} wins", "winner": "B"}')["winner"] == "B"
    # a lone surrogate: escapes for half a character
    assert extract_json(r'{"reasoning": "derive \u00a2\udc80\udd6a(n)"}')["reasoning"] == "derive ¢\ufffd\ufffd(n)"


def test_query_generation():
    corpus = make_corpus()
    gen = QueryGenerator(MockLLM(), task_instruction="t", n_queries=8, filter_queries=True, workers=1)
    kept = gen.run(corpus)
    assert gen.n_generated >= 8 and len(kept) <= 8 and all(q.quality is not None for q in kept)

    # worker count must never change the query set, including under flaky calls
    class Flaky(MockLLM):
        def chat(self, messages, **kw):
            prompt = " ".join(m.get("content", "") for m in messages)
            return "no json here" if self._hash(prompt) % 3 == 0 else super().chat(messages, **kw)

    def gen_with(workers, client):
        return QueryGenerator(
            client, task_instruction="t", n_queries=12, filter_queries=False, workers=workers
        ).generate(corpus)

    for client in (MockLLM, Flaky):
        seq, par = gen_with(1, client()), gen_with(4, client())
        assert len(seq) == 12
        assert [(q.qid, q.text, tuple(q.seed_doc_ids)) for q in par] == [
            (q.qid, q.text, tuple(q.seed_doc_ids)) for q in seq
        ]


def test_a_changed_prompt_is_a_new_query_set():
    """The prompts are part of the query set's identity, so a changed prompt generates new queries
    instead of reusing the cached ones."""
    corp = types.SimpleNamespace(id="c", name="c", docs=make_corpus())
    with tempfile.TemporaryDirectory() as tmp:

        def qset(gen):
            return resolve_queries(corp, Path(tmp), "synthetic", gen, 4, 0).id

        gen = QueryGenerator(MockLLM(), task_instruction="t", n_queries=4, filter_queries=False, workers=1)
        assert qset(gen) == qset(
            QueryGenerator(MockLLM(), task_instruction="t", n_queries=4, filter_queries=False, workers=1)
        )
        gen.system += " One passage, one query."
        assert qset(gen) != qset(
            QueryGenerator(MockLLM(), task_instruction="t", n_queries=4, filter_queries=False, workers=1)
        )
