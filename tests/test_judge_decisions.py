"""What the judges decide and how answers are read."""

import json

import httpx
import pytest

from mteb_gym.judge import PairwiseJudge, PointwiseJudge, _cut, grade_state
from mteb_gym.llm import MockLLM
from mteb_gym.retrieval import Ranked


class Answering:
    model = "answering"

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def chat(self, messages):
        self.prompts.append(messages)
        return self.answer(messages)


def ranked(qid, text):
    return Ranked(qid, "query", [text], [text])


def prefers(word):
    """Picks whichever system's list contains `word`, wherever it is shown."""

    def answer(messages):
        a, b = messages[-1]["content"].split("System B results:")
        return json.dumps({"winner": "A" if word in a else "B" if word in b else "tie"})

    return answer


def test_a_judge_that_prefers_the_better_list_scores_it_1_in_either_position():
    judge = PairwiseJudge(Answering(prefers("good")), "t")
    assert judge.judge_pair(ranked("q", "good"), ranked("q", "bad"), "a", "b").score_a == 1.0
    assert judge.judge_pair(ranked("q", "bad"), ranked("q", "good"), "a", "b").score_a == 0.0


def test_position_bias_cancels():
    """A judge that always picks whichever list is shown first has told us nothing: both orders average to 0.5."""
    judge = PairwiseJudge(Answering(lambda m: json.dumps({"winner": "A"})), "t")
    v = judge.judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    assert v.score_a == 0.5 and v.raw == ["A", "A"]


def test_a_win_and_a_tie_score_three_quarters():
    answers = iter(['{"winner": "A"}', '{"winner": "tie"}'])
    v = PairwiseJudge(Answering(lambda m: next(answers)), "t").judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    assert v.score_a == 0.75


def test_the_second_order_really_swaps_the_lists():
    client = Answering(lambda m: '{"winner": "tie"}')
    PairwiseJudge(client, "t").judge_pair(ranked("q", "first"), ranked("q", "second"), "a", "b")
    one, two = client.prompts[0][-1]["content"], client.prompts[1][-1]["content"]
    assert one.index("first") < one.index("second") and two.index("second") < two.index("first")


def test_identical_lists_tie_without_asking():
    client = Answering(lambda m: pytest.fail("the judge should not be called"))
    v = PairwiseJudge(client, "t").judge_pair(ranked("q", "same"), ranked("q", "same"), "a", "b")
    assert v.score_a == 0.5 and v.parsed_ok == []


def test_the_instruction_reaches_each_judge():
    client = Answering(lambda m: '{"winner": "tie", "grade": 1}')
    PairwiseJudge(client, "find refutations").judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    PointwiseJudge(client, "find refutations", workers=1).grade("q", "query", "d", "text")
    assert all("find refutations" in p[0]["content"] for p in client.prompts)
    assert "find refutations" in grade_state("query", "text", "find refutations", 10)


def test_documents_are_cut_for_the_judge():
    client = Answering(lambda m: '{"winner": "tie", "grade": 1}')
    long = "word " * 100
    PairwiseJudge(client, "t", doc_chars=20).judge_pair(ranked("q", long), ranked("q", "y"), "a", "b")
    PointwiseJudge(client, "t", workers=1, doc_chars=20).grade("q", "query", "d", long)
    assert all(long not in p[-1]["content"] and "word word" in p[-1]["content"] for p in client.prompts)


def test_cut_handles_text_without_a_word_boundary():
    assert _cut("   ", 2) == "  ..."
    assert _cut("abcdef", 3) == "abc..."
    assert _cut("ab cdef", 5) == "ab..."
    assert _cut("short", 10) == "short"


@pytest.mark.parametrize(
    "answer, grade, ok",
    [
        ('{"grade": 4}', 4.0, True),
        ('{"grade": 0}', 0.0, True),
        ('{"grade": 6}', 6.0, False),
        ('{"grade": -1}', -1.0, False),
        ('{"grade": "high"}', 0.0, False),
        ('{"reasoning": "no grade"}', 0.0, False),
        ("not json", 0.0, False),
    ],
)
def test_a_chat_judges_grade_is_read_or_flagged(answer, grade, ok):
    g = PointwiseJudge(Answering(lambda m: answer), "t", workers=1).grade("q", "query", "d", "text")
    assert (g.grade, g.parsed_ok) == (grade, ok)


def test_answers_with_odd_types_do_not_stop_a_run():
    v = PairwiseJudge(Answering(lambda m: '{"winner": "A", "reasoning": 3}'), "t")
    assert v.judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b").reasoning == ""
    g = PointwiseJudge(Answering(lambda m: '{"grade": 2, "reasoning": ["x"]}'), "t", workers=1)
    assert g.grade("q", "query", "d", "text").reasoning == ""


def test_a_decision_model_grade_is_the_expected_grade():
    class Decider:
        model = "decider"

        def decide(self, kind, state, question, options=None):
            assert kind == "score" and "Retrieval task: t" in state and "Query: query" in state
            return {"1": 0.5, "5": 0.5}

    g = PointwiseJudge(Decider(), "t", workers=1).grade("q", "query", "d", "text")
    assert g.grade == 3.0 and g.parsed_ok and g.probabilities == {"1": 0.5, "5": 0.5}


def test_the_mock_can_grade():
    g = PointwiseJudge(MockLLM(), "t", workers=1).grade("q", "query", "d", "text")
    assert g.parsed_ok and 0 <= g.grade <= 5


@pytest.mark.parametrize("error", [TimeoutError("slow"), httpx.ReadTimeout("slow")])
def test_every_kind_of_timeout_costs_one_answer(error):
    def times_out(messages):
        raise error

    assert PairwiseJudge(Answering(times_out), "t").judge_pair(
        ranked("q", "x"), ranked("q", "y"), "a", "b"
    ).parsed_ok == [
        False,
        False,
    ]
    assert not PointwiseJudge(Answering(times_out), "t", workers=1).grade("q", "query", "d", "text").parsed_ok


def test_a_dead_judge_stops_early():
    a = [ranked(f"q{i}", "x") for i in range(40)]
    b = [ranked(f"q{i}", "y") for i in range(40)]
    client = Answering(lambda m: "not json")
    with pytest.raises(RuntimeError, match="parse-failure"):
        PairwiseJudge(client, "t").judge_all(a, b, "a", "b")
    assert len(client.prompts) == 60  # the first 30 comparisons, each in both orders


def test_identical_retrieval_everywhere_stops_early():
    a = [ranked(f"q{i}", "x") for i in range(40)]
    with pytest.raises(RuntimeError, match="identical"):
        PairwiseJudge(Answering(lambda m: pytest.fail("not asked")), "t").judge_all(a, a, "a", "b")


def test_a_decision_model_sees_the_cut_document():
    seen = []

    class Decider:
        model = "decider"

        def decide(self, kind, state, question, options=None):
            seen.append(state)
            return {"1": 1.0}

    PointwiseJudge(Decider(), "t", workers=1, doc_chars=20).grade("q", "query", "d", "word " * 100)
    assert "word " * 100 not in seen[0] and "word word" in seen[0]
