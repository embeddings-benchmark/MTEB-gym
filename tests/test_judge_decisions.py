"""What the judges decide and how answers are read."""

import json

import httpx
import pytest

from mteb_gym.judge import PairwiseJudge, PointwiseJudge, _cut, grade_state
from mteb_gym.llm import InputTooLong
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
    with pytest.raises(RuntimeError, match="no usable answer"):
        PairwiseJudge(client, "t").judge_all(a, b, "a", "b")
    assert len(client.prompts) == 60  # the first 30 comparisons, each in both orders


def test_a_decision_model_sees_the_cut_document():
    seen = []

    class Decider:
        model = "decider"

        def decide(self, kind, state, question, options=None):
            seen.append(state)
            return {"1": 1.0}

    PointwiseJudge(Decider(), "t", workers=1, doc_chars=20).grade("q", "query", "d", "word " * 100)
    assert "word " * 100 not in seen[0] and "word word" in seen[0]


@pytest.mark.parametrize(
    "probabilities",
    [{}, {"0": float("nan"), "5": 1.0}, {"0": 50.0, "5": 50.0}, {"0": 0.5}, {"high": 1.0}, {"0": -1.0, "5": 2.0}],
)
def test_an_invalid_decision_is_flagged(probabilities):
    class Decider:
        model = "decider"

        def decide(self, kind, state, question, options=None):
            return probabilities

    g = PointwiseJudge(Decider(), "t", workers=1).grade("q", "query", "d", "text")
    assert not g.parsed_ok and g.grade == 0.0


class Deciding:
    """Answers each order from `answers`, and keeps what it was asked."""

    model = "deciding"

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def decide(self, kind, state, question, options=None):
        self.asked.append((kind, state, question, options))
        return self.answers.pop(0)


def outcome(a, b, tie):
    return {"System A": a, "System B": b, "Tie": tie}


def test_a_decision_model_scores_a_pair_by_its_probabilities():
    # first order: .9 + .08 / 2 = .94 for a. Second order, a shown second: .1 + .2 / 2 = .2 for b, so .8 for a.
    client = Deciding(outcome(0.9, 0.02, 0.08), outcome(0.1, 0.7, 0.2))
    v = PairwiseJudge(client, "t").judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    assert v.score_a == pytest.approx(0.87)
    assert v.raw == ["A", "B"] and v.parsed_ok == [True, True]
    assert v.probabilities == [outcome(0.9, 0.02, 0.08), outcome(0.1, 0.7, 0.2)]


def test_a_decision_models_position_bias_cancels():
    # it favours whichever list is shown first by the same amount: (.8 + 1 - .8) / 2
    client = Deciding(outcome(0.7, 0.1, 0.2), outcome(0.7, 0.1, 0.2))
    assert PairwiseJudge(client, "t").judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b").score_a == pytest.approx(
        0.5
    )


def test_a_decision_model_sees_the_task_the_query_and_both_orders():
    client = Deciding(outcome(1, 0, 0), outcome(1, 0, 0))
    PairwiseJudge(client, "find refutations", doc_chars=8).judge_pair(
        ranked("q", "first document"), ranked("q", "second document"), "a", "b"
    )
    (kind, one, question, options), (_, two, _, _) = client.asked
    assert kind == "choice" and options == ["System A", "System B", "Tie"] and "Which result list" in question
    assert one == (
        "Retrieval task: find refutations\nQuery: query\n\nSystem A results:\n  1. first...\n\n"
        "System B results:\n  1. second..."
    )
    assert two.index("second...") < two.index("first...")


def test_the_instruction_is_part_of_what_a_decision_model_pair_judge_is_told():
    assert "find refutations" in PairwiseJudge(Deciding(), "find refutations").system
    assert PairwiseJudge(Deciding(), "a").system != PairwiseJudge(Deciding(), "b").system


def test_identical_lists_tie_without_asking_a_decision_model():
    client = Deciding()
    v = PairwiseJudge(client, "t").judge_pair(ranked("q", "same"), ranked("q", "same"), "a", "b")
    assert v.score_a == 0.5 and v.parsed_ok == [] and client.asked == []


@pytest.mark.parametrize(
    "bad",
    [{}, {"0": 1.0}, outcome(0.9, 0.9, 0.9), outcome(float("nan"), 0.5, 0.5), outcome(-0.5, 1.0, 0.5)],
)
def test_a_decision_models_unusable_answer_is_flagged(bad):
    client = Deciding(bad, outcome(1, 0, 0))
    v = PairwiseJudge(client, "t").judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    assert v.parsed_ok == [False, True] and v.raw == ["tie", "A"] and v.probabilities is None


class Grading:
    model = "grading"

    def __init__(self):
        self.asked = []

    def decide(self, kind, state, question, options=None):
        self.asked.append((state, question))
        return {"3": 1.0}


def test_without_a_sentence_the_judges_ask_about_plain_relevance():
    grader = Grading()
    PointwiseJudge(grader, workers=1, doc_chars=5).grade("q", "query", "d", "text")
    assert grader.asked == [
        (
            "Query: query\nDocument: text",
            "How relevant is the document to the query? 0 = not relevant at all, 5 = perfectly relevant: it fully "
            "satisfies the information need.",
        )
    ]
    pair = Deciding(outcome(1, 0, 0), outcome(1, 0, 0))
    PairwiseJudge(pair).judge_pair(ranked("q", "x"), ranked("q", "y"), "a", "b")
    assert pair.asked[0][1].startswith("Query: query\n")


def test_the_sentence_is_part_of_what_each_judge_is_told():
    for judge in (PointwiseJudge, PairwiseJudge):
        for client in (Grading(), Answering(lambda m: "{}")):
            with_sentence = judge(client, "find refutations")
            if judge is PointwiseJudge and with_sentence.decision:
                continue  # the sentence reaches a decision model in the state, and is its own part of the grades' key
            assert judge(client).system != with_sentence.system
    grader = Grading()
    PointwiseJudge(grader, "find refutations", workers=1).grade("q", "query", "d", "text")
    state, question = grader.asked[0]
    assert state.startswith("Retrieval task: find refutations\n") and "for this retrieval task" in question


def test_default_document_limits():
    long = "word " * 1000
    assert PointwiseJudge(Grading()).doc_chars == 16000
    assert PairwiseJudge(Grading()).doc_chars == 4000
    client = Deciding(outcome(1, 0, 0), outcome(1, 0, 0))
    PairwiseJudge(client).judge_pair(ranked("q", long), ranked("q", "y"), "a", "b")
    assert long not in client.asked[0][1] and "word " * 700 in client.asked[0][1]


class Limited:
    """A decision model with a context limit: an input over it does not fit."""

    model = "limited"

    def __init__(self, limit):
        self.limit, self.states = limit, []

    def decide(self, kind, state, question, options=None):
        self.states.append(state)
        if len(state) > self.limit:
            raise InputTooLong("does not fit")
        return outcome(1, 0, 0) if kind == "choice" else {"3": 1.0}


def lists(text):
    return Ranked("q", "query", ["d"], [text])


def test_a_comparison_that_does_not_fit_is_cut_shorter_in_both_orders():
    # 800 characters per side does not fit in 1000; the first shorter cut, 400 per side, does.
    client = Limited(1000)
    v = PairwiseJudge(client, "t", doc_chars=800).judge_pair(lists("a " * 500), lists("b " * 500), "a", "b")
    assert v.doc_chars == 400 and v.parsed_ok == [True, True]
    assert v.score_a == pytest.approx(0.5)  # it picked whichever came first, so the orders cancel
    assert all(len(st) <= 1000 for st in client.states[-2:]) and len(client.states) == 3


def test_a_comparison_that_fits_records_no_shorter_cut():
    v = PairwiseJudge(Limited(10**6), "t", doc_chars=800).judge_pair(lists("a"), lists("b"), "a", "b")
    assert v.doc_chars is None


def test_a_comparison_that_never_fits_is_left_out_and_does_not_stop_the_run():
    v = PairwiseJudge(Limited(5), "t", doc_chars=800).judge_pair(lists("a " * 500), lists("b " * 500), "a", "b")
    assert v.parsed_ok == [False, False] and v.score_a == 0.5


def test_a_document_that_does_not_fit_is_graded_on_a_shorter_cut():
    g = PointwiseJudge(Limited(500), "t", workers=1, doc_chars=800).grade("q", "query", "d", "a " * 500)
    assert g.doc_chars == 400 and g.parsed_ok and g.grade == 3.0
    assert not PointwiseJudge(Limited(5), "t", workers=1, doc_chars=800).grade("q", "query", "d", "a " * 500).parsed_ok


def test_a_chat_judge_is_cut_shorter_too():
    class Chat:
        model = "chat"

        def chat(self, messages):
            if len(messages[1]["content"]) > 1000:
                raise InputTooLong("does not fit")
            return '{"winner": "A", "grade": 2}'

    v = PairwiseJudge(Chat(), "t", doc_chars=800).judge_pair(lists("a " * 500), lists("b " * 500), "a", "b")
    assert v.doc_chars == 400
    assert PointwiseJudge(Chat(), "t", workers=1, doc_chars=1600).grade("q", "query", "d", "a " * 900).doc_chars == 800
