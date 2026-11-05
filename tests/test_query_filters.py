"""Which generated queries are kept, and how answers are read."""

import json

import pytest

from mteb_gym.queries import Query, QueryGenerator, extract_json


class Answers:
    model = "answers"

    def __init__(self, generated=(), scores=None):
        self.generated, self.scores, self.calls = list(generated), scores or {}, []

    def chat(self, messages):
        self.calls.append(messages)
        system = messages[0]["content"]
        if "You rate the quality" in system:
            for text, score in self.scores.items():
                if text in messages[1]["content"]:
                    return json.dumps({"score": score})
            return json.dumps({"score": 5})
        return json.dumps({"query": self.generated.pop(0) if self.generated else "enough other query"})


def generator(client, **kw):
    return QueryGenerator(client, task_instruction="Given a claim, find evidence", workers=1, seed=0, **kw)


DOCS = {f"d{i}": f"document {i}" for i in range(9)}


@pytest.mark.parametrize(
    "text, keep",
    [
        ("walnut oil", True),
        ("Phoenix is a swamp.", True),
        ("ab", False),
        ("x" * 4001, False),
        ("x" * 4000, True),
        ("1234", False),
        ("Феникс это болото", True),
        ("東京都", True),
    ],
)
def test_length_and_letters(text, keep):
    assert generator(Answers())._heuristic_ok(text) is keep


def test_filter_keeps_only_scores_at_or_above_the_minimum():
    client = Answers(scores={"low": 2, "edge": 3, "top": 5})
    gen = generator(client, n_queries=9, min_score=3)
    kept = gen.apply_filters([Query(f"q{i}", t, []) for i, t in enumerate(["low query", "edge query", "top query"])])
    assert [q.text for q in kept] == ["edge query", "top query"] and [q.qid for q in kept] == ["q0", "q1"]
    assert [q.quality for q in kept] == [3, 5]


def test_filter_can_be_switched_off():
    client = Answers(scores={"low": 1})
    gen = generator(client, n_queries=1, filter_queries=False)
    assert [q.text for q in gen.apply_filters([Query("q0", "low query", []), Query("q1", "two", [])])] == ["low query"]
    assert not client.calls


def test_filter_is_told_the_task():
    client = Answers()
    generator(client, n_queries=1).apply_filters([Query("q0", "a claim", [])])
    assert "Given a claim, find evidence" in client.calls[0][0]["content"]


def test_near_duplicates_are_removed_in_any_script():
    client = Answers()
    gen = generator(client, n_queries=9, dedup=0.8)
    same = ["alpha beta gamma delta epsilon", "ALPHA beta gamma delta epsilon", "alpha beta gamma delta zeta"]
    kept = gen.apply_filters([Query(f"q{i}", t, []) for i, t in enumerate(same)])
    assert [q.text for q in kept] == [same[0], same[2]]
    russian = ["Феникс это болото", "Феникс это болото", "Токио большой город"]
    kept = gen.apply_filters([Query(f"q{i}", t, []) for i, t in enumerate(russian)])
    assert [q.text for q in kept] == [russian[0], russian[2]]


def test_overshoot_generates_more_than_the_target_only_when_filtering():
    assert len(generator(Answers(), n_queries=10, overshoot=1.5).generate(DOCS)) == 15
    assert len(generator(Answers(), n_queries=10, overshoot=1.5, filter_queries=False).generate(DOCS)) == 10


def test_unusable_answers_are_retried_with_fresh_documents():
    class Flaky(Answers):
        def chat(self, messages):
            self.calls.append(messages)
            return {1: "no json", 2: '{"query": 5}'}.get(len(self.calls), '{"query": "a usable query"}')

    client = Flaky()
    assert [q.text for q in generator(client, n_queries=1, filter_queries=False).generate(DOCS)] == ["a usable query"]
    assert len(client.calls) == 3


def test_the_generator_sees_whole_documents_and_the_task():
    client = Answers()
    long = {"d": "start " + "word " * 1000 + "the end"}
    generator(client, n_queries=1, filter_queries=False).generate(long)
    assert "the end" in client.calls[0][1]["content"]
    assert "Given a claim, find evidence" in client.calls[0][0]["content"]


def test_seed_documents_are_recorded():
    q = generator(Answers(), n_queries=1, filter_queries=False, docs_per_query=2).generate(DOCS)[0]
    assert len(q.seed_doc_ids) == 2 and set(q.seed_doc_ids) <= set(DOCS)


@pytest.mark.parametrize(
    "text, parsed",
    [
        ('<think>maybe {"winner": "B"}</think>{"winner": "A"}', {"winner": "A"}),
        ('<think>maybe {"winner": "B"}</think>{"reasoning": "cut off', {}),
        ('<think>still going {"winner": "B"}', {}),
        ('draft {"winner": "A"} hmm</think>{"reasoning": "cut off', {}),
        ('<thinking>{"winner": "B"}</thinking>{"winner":', {}),
        ('Maybe {"winner": "B"}? Actually the answer is {"winner":', {}),
        ('draft {"winner": "A"}</think>{"winner": "B"}', {"winner": "B"}),
    ],
)
def test_drafts_in_reasoning_are_never_the_answer(text, parsed):
    assert extract_json(text) == parsed


def test_generate_rejects_what_fails_the_length_rule():
    client = Answers(generated=["x" * 5000, "a usable query"])
    assert [q.text for q in generator(client, n_queries=1, filter_queries=False).generate(DOCS)] == ["a usable query"]


def test_the_filter_stops_at_the_requested_number():
    queries = [Query(f"q{i}", f"distinct query {i} word{i}", []) for i in range(5)]
    assert len(generator(Answers(), n_queries=2).apply_filters(queries)) == 2


def test_the_seed_decides_which_documents_are_shown():
    many = {f"d{i}": f"document {i}" for i in range(200)}

    def seeds(seed):
        gen = QueryGenerator(Answers(), task_instruction="t", n_queries=2, seed=seed, filter_queries=False, workers=1)
        return [q.seed_doc_ids for q in gen.generate(many)]

    assert seeds(0) == seeds(0) and seeds(0) != seeds(1)


def test_a_score_it_cannot_read_is_the_neutral_three():
    class NoScore(Answers):
        def chat(self, messages):
            return json.dumps({"reason": "no score"})

    q = Query("q0", "a query", [])
    generator(NoScore())._llm_quality([q])
    assert q.quality == 3
