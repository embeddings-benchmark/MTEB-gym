from mteb_gym.judge import Verdict
from mteb_gym.rank import format_leaderboard, rate, rate_pointwise
from mteb_gym.retrieval import Ranked


def test_failed_comparisons_are_left_out():
    """A comparison with an unparseable order counts for nothing, rather than as a tie."""
    won = [Verdict(f"q{i}", "q", "a", "b", 1.0, parsed_ok=[True, True]) for i in range(6)]
    failed = [Verdict(f"f{i}", "q", "a", "b", 0.5, parsed_ok=[True, False]) for i in range(6)]
    assert rate(won + failed, bootstrap=0) == rate(won, bootstrap=0)


def test_rank():
    def v(qid, a, b, score):
        return Verdict(qid=qid, query="q", model_a=a, model_b=b, score_a=score)

    verdicts = []
    for i in range(20):
        verdicts += [
            v(f"q{i}", "winner", "mid", 0.75),
            v(f"q{i}", "mid", "loser", 1.0),
            v(f"q{i}", "winner", "loser", 1.0),
        ]
    ratings = rate(verdicts, bootstrap=100)
    assert [r.name for r in ratings] == ["winner", "mid", "loser"]
    assert ratings[-1].rating < ratings[1].rating - 50, "a model that loses every verdict must sink"
    assert (
        format_leaderboard(
            [
                {
                    "model": r.name,
                    **{k: getattr(r, k) for k in ("rating", "ci_low", "ci_high", "wins", "losses", "ties")},
                }
                for r in ratings
            ]
        ).count("\n")
        == 6
    )


def test_rate_pointwise():
    from mteb_gym.rank import ndcg_per_query

    assert ndcg_per_query(["d1", "d2"], {"d1": 5, "d2": 0}, [5, 0], k=2) == 1.0
    assert ndcg_per_query(["d2", "d1"], {"d1": 5, "d2": 0}, [5, 0], k=2) < 1.0
    qids = [f"q{i}" for i in range(20)]
    good = [Ranked(q, q, ["rel", "x"], ["", ""]) for q in qids]
    bad = [Ranked(q, q, ["x", "rel"], ["", ""]) for q in qids]
    grades = {(q, "rel"): 5.0 for q in qids} | {(q, "x"): 0.0 for q in qids}
    top, second = rate_pointwise(grades, {"good": good, "bad": bad}, k=2)
    assert top.name == "good" and top.rating == 100.0 and second.rating < 100.0
    assert top.ci_low <= top.rating <= top.ci_high and top.n == 20 and top.wins == 20
