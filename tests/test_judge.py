import pytest

from mteb_gym import results
from mteb_gym.judge import PairwiseJudge, PointwiseJudge
from mteb_gym.llm import MockLLM
from mteb_gym.queries import Query
from mteb_gym.retrieval import Ranked
from tests.helpers import fake_ranked


def test_doc_chars_reaches_the_judge():

    assert PairwiseJudge(MockLLM(), "t", doc_chars=300).doc_chars == 300
    assert (
        "x" * 301 not in PairwiseJudge(MockLLM(), "t", doc_chars=300).system
    )  # the setting is the judge's, not the prompt's


def test_judge():
    from mteb_gym.judge import _format

    shown = _format(Ranked("q", "query", ["d1", "d2"], ["one two three four five", "short"]), doc_chars=12)
    assert shown.splitlines() == ["  1. one two...", "  2. short"]  # cut at a word boundary and marked, or whole
    queries = [Query(f"q{i}", f"query {i} about statins", ["D0"]) for i in range(30)]
    ra, rb = fake_ranked("a", queries), fake_ranked("b", queries)
    seq = PairwiseJudge(MockLLM(seed=1), "t", workers=1).judge_all(ra, rb, "m_a", "m_b")
    par = PairwiseJudge(MockLLM(seed=1), "t", workers=8).judge_all(ra, rb, "m_a", "m_b")
    assert all(0.0 <= v.score_a <= 1.0 for v in seq)
    assert [(v.qid, v.score_a) for v in par] == [(v.qid, v.score_a) for v in seq]

    class Exploding:
        def chat(self, messages, **kw):
            raise AssertionError("judge must not be called for identical result sets")

    v = PairwiseJudge(Exploding(), "t").judge_all(ra[:1], fake_ranked("a", queries[:1]), "m_a", "m_b")[0]
    assert v.score_a == 0.5 and v.raw == ["identical"]

    class Garbage:
        def chat(self, messages, **kw):
            return "I refuse to answer in the requested format."

    verdicts = PairwiseJudge(Garbage(), "t").judge_all(ra[:5], rb[:5], "m_a", "m_b")
    assert all(v.score_a == 0.5 for v in verdicts)
    assert results.verdict_diagnostics(verdicts)["parse_failure_rate"] == 1.0


def test_a_timed_out_judge_call_is_left_out():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")

    class Slow(MockLLM):
        def chat(self, messages, **kw):
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://judge"))

    queries = [Query("q0", "query", ["D0"])]
    v = PairwiseJudge(Slow(), "t", workers=1).judge_pair(
        fake_ranked("a", queries)[0], fake_ranked("b", queries)[0], "a", "b"
    )
    assert v.parsed_ok == [False, False]

    assert not PointwiseJudge(Slow(), "t", workers=1).grade("q0", "query", "D0", "text").parsed_ok

    class Broken(MockLLM):
        def chat(self, messages, **kw):
            raise ValueError("bad key")

    with pytest.raises(ValueError):
        PairwiseJudge(Broken(), "t", workers=1).judge_pair(
            fake_ranked("a", queries)[0], fake_ranked("b", queries)[0], "a", "b"
        )
