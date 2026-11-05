"""The maths behind a ranking, against answers worked out by hand."""

import math

import numpy as np
import pytest

from mteb_gym.judge import Verdict
from mteb_gym.rank import BASE, SCALE, _bradley_terry, ndcg_per_query, rate, rate_pointwise
from mteb_gym.retrieval import Ranked


def lists(model_docs):
    return {m: [Ranked(q, "query", d, list(d)) for q, d in queries.items()] for m, queries in model_docs.items()}


def test_ndcg_uses_linear_gain_and_log_discount():
    grades = {"a": 3.0, "b": 1.0}
    expected_dcg = 1.0 / math.log2(2) + 3.0 / math.log2(3)
    ideal = 3.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_per_query(["b", "a"], grades, [3.0, 1.0], 2) == pytest.approx(expected_dcg / ideal)


def test_ndcg_ideal_comes_from_the_pool_not_the_models_own_list():
    """A model that found only a weak document cannot score 1 when the pool holds a better one."""
    ideal = 5.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_per_query(["weak"], {"weak": 1.0, "strong": 5.0}, [1.0, 5.0], 10) == pytest.approx(1.0 / ideal)


def test_ndcg_counts_only_the_top_k_of_both_the_list_and_the_ideal():
    grades = {"a": 0.0, "b": 5.0, "c": 5.0, "d": 5.0}
    pool = list(grades.values())
    ideal_at_1 = 5.0
    assert ndcg_per_query(["a", "b"], grades, pool, 1) == 0.0
    assert ndcg_per_query(["b", "a"], grades, pool, 1) == pytest.approx(5.0 / ideal_at_1)


def test_ndcg_of_an_all_zero_pool_is_zero():
    assert ndcg_per_query(["a"], {"a": 0.0}, [0.0], 10) == 0.0


def test_pointwise_rating_is_mean_ndcg_times_100():
    ranked = lists({"good": {"q0": ["d1"], "q1": ["d3"]}, "bad": {"q0": ["d2"], "q1": ["d4"]}})
    grades = {("q0", "d1"): 5.0, ("q0", "d2"): 0.0, ("q1", "d3"): 5.0, ("q1", "d4"): 5.0}
    by_name = {r.name: r for r in rate_pointwise(grades, ranked, k=1, bootstrap=0)}
    assert by_name["good"].rating == pytest.approx(100.0)
    assert by_name["bad"].rating == pytest.approx(50.0)
    assert (by_name["good"].wins, by_name["good"].ties, by_name["bad"].losses) == (1.0, 1.0, 1.0)


def test_pointwise_uses_k():
    ranked = lists({"m": {"q0": ["bad", "good"]}})
    grades = {("q0", "bad"): 0.0, ("q0", "good"): 5.0}
    assert rate_pointwise(grades, ranked, k=1, bootstrap=0)[0].rating == 0.0
    assert rate_pointwise(grades, ranked, k=2, bootstrap=0)[0].rating == pytest.approx(100 / math.log2(3))


def test_pointwise_interval_reflects_variation_across_queries():
    n = 40
    ranked = lists({"m": {f"q{i}": [f"d{i}"] for i in range(n)}})
    grades = {(f"q{i}", f"d{i}"): 5.0 for i in range(n)}
    grades |= {(f"q{i}", f"other{i}"): 5.0 if i % 2 else 0.0 for i in range(n)}  # half the queries: a tied better doc
    ranked["other"] = [Ranked(f"q{i}", "query", [f"other{i}"], [f"other{i}"]) for i in range(n)]
    r = {x.name: x for x in rate_pointwise(grades, ranked, k=1, bootstrap=500, seed=0)}
    assert r["m"].ci_low == pytest.approx(r["m"].rating) == pytest.approx(r["m"].ci_high)  # always 100: no variation
    assert r["other"].rating == pytest.approx(50.0)
    assert 30 < r["other"].ci_low < 50 < r["other"].ci_high < 70


def verdict(a, b, score, qid="q0"):
    return Verdict(qid, "query", a, b, score, raw=["A", "A"], parsed_ok=[True, True])


def test_bradley_terry_two_models_closed_form():
    """Two models: the strength ratio is exactly the ratio of wins, after smoothing."""
    p = _bradley_terry(np.array([[0.0, 3.0], [1.0, 0.0]]))
    assert p[0] / p[1] == pytest.approx(3.1 / 1.1)
    assert p[0] * p[1] == pytest.approx(1.0)


def test_bradley_terry_converges_on_lopsided_matches():
    """A fixed-point solution: each model's wins equal its expected wins. Cutting iterations short breaks this."""
    W = np.array([[0.0, 95.0, 99.0, 99.5], [5.0, 0.0, 95.0, 99.0], [1.0, 5.0, 0.0, 95.0], [0.5, 1.0, 5.0, 0.0]])
    p = _bradley_terry(W)
    S = W + 0.1 * (1 - np.eye(4))
    games = S + S.T
    expected = (games * (p[:, None] / (p[:, None] + p[None, :]))).sum(axis=1)
    assert expected == pytest.approx(S.sum(axis=1), rel=1e-6)
    assert list(np.argsort(-p)) == [0, 1, 2, 3]


def test_rate_scale_and_order():
    verdicts = [verdict("a", "b", 1.0, f"q{i}") for i in range(3)] + [verdict("a", "b", 0.0, "q3")]
    r = {x.name: x for x in rate(verdicts, bootstrap=0)}
    gap = SCALE * math.log10(3.1 / 1.1)
    assert r["a"].rating - r["b"].rating == pytest.approx(gap)
    assert (r["a"].rating + r["b"].rating) / 2 == pytest.approx(BASE)
    assert (r["a"].wins, r["a"].losses, r["b"].wins, r["b"].losses) == (3.0, 1.0, 1.0, 3.0)


def test_rate_counts_a_half_as_a_tie_for_both():
    r = {x.name: x for x in rate([verdict("a", "b", 0.5)], bootstrap=0)}
    assert r["a"].ties == r["b"].ties == 1.0 and r["a"].rating == pytest.approx(r["b"].rating)


def test_a_winless_model_ends_at_the_bottom():
    verdicts = [verdict("a", "b", 1.0), verdict("a", "c", 1.0), verdict("b", "c", 1.0)]
    assert [x.name for x in rate(verdicts, bootstrap=0)] == ["a", "b", "c"]


def test_rate_bootstraps_over_queries_not_verdicts():
    """Every query has three correlated verdicts, all won by whichever side that query favours. Resampling verdicts
    would treat them as independent and give too narrow an interval."""
    verdicts = []
    for i in range(12):
        score = 1.0 if i % 2 else 0.0
        verdicts += [verdict("a", "b", score, f"q{i}") for _ in range(1)]
        verdicts += [
            Verdict(f"q{i}", "query", "a", "b", score, raw=["A", "A"], parsed_ok=[True, True]) for _ in range(2)
        ]
    r = {x.name: x for x in rate(verdicts, bootstrap=300, seed=0)}
    half_width = (r["a"].ci_high - r["a"].ci_low) / 2
    per_verdict = {
        x.name: x
        for x in rate([verdict("a", "b", v.score_a, f"v{i}") for i, v in enumerate(verdicts)], bootstrap=300, seed=0)
    }
    assert half_width > (per_verdict["a"].ci_high - per_verdict["a"].ci_low) / 2 * 1.3
    assert r["a"].ci_low < r["a"].rating < r["a"].ci_high
