"""Tests for analysis.arena: hand-built arena rows through the mock judge, a kappa case worked by hand, then
battles built from the mock fixtures under ANALYSIS_FIXTURES when that variable is set. No mteb, no network."""

import itertools
import json
import os
from pathlib import Path

import pytest

from analysis import arena
from mteb_gym import reliability as rel
from mteb_gym.llm import MockLLM

FIXTURES = os.environ.get("ANALYSIS_FIXTURES")


def row(i: int, vote_type: str, doc_a: str | None = None, doc_b: str | None = None) -> dict:
    """One retrieval_battle row with the dataset's field names; each side shows one document."""
    query = f"query {i}"
    return {
        "tstamp": 1751193963.0 + i,
        "task_type": "retrieval",
        "type": vote_type,
        "models": ["", ""],
        "ip": "",
        "0_conv_id": f"left{i:02d}",
        "0_model_name": "m/left",
        "0_prompt": query,
        "0_output": [[query, doc_a if doc_a is not None else f"document a{i}"]],
        "0_corpus": "wikipedia",
        "1_conv_id": f"right{i:02d}",
        "1_model_name": "m/right",
        "1_prompt": query,
        "1_output": [[query, doc_b if doc_b is not None else f"document b{i}"]],
        "1_corpus": "wikipedia",
    }


ROWS = [
    row(0, "leftvote"),
    row(1, "rightvote"),
    row(2, "tievote"),
    row(3, "bothbadvote"),
    row(4, "leftvote", "same text", "same text"),  # identical sides: tied without a judge call
    row(5, "share"),  # a share click, not a vote
    row(6, "rightvote"),
]


def test_load_battles_maps_the_arena_fields(tmp_path):
    battles = arena.load_battles(ROWS)
    assert [b["battle_id"] for b in battles] == [f"left{i:02d}__right{i:02d}" for i in (0, 1, 2, 3, 4, 6)]
    assert [b["vote"] for b in battles] == ["A", "B", "tie", "both_bad", "A", "B"]
    b = battles[1]
    assert b["query"] == "query 1" and b["model_a"] == "m/left" and b["model_b"] == "m/right"
    assert b["docs_a"] == ["document a1"] and b["docs_b"] == ["document b1"]  # the query is dropped from output
    # the same rows as a datasets-server page, a .json file and a .jsonl file
    page = {"rows": [{"row_idx": i, "row": r} for i, r in enumerate(ROWS)]}
    assert arena.load_battles(page) == battles
    (tmp_path / "b.json").write_text(json.dumps(ROWS), encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    assert arena.load_battles(tmp_path / "b.json") == battles and arena.load_battles(tmp_path / "b.jsonl") == battles
    assert arena.battle_of({"type": "share"}, 0) is None
    assert arena.battle_of(row(9, "leftvote") | {"0_conv_id": ""}, 9)["battle_id"] == "row9"
    with pytest.raises(arena.ArenaError):
        arena.load_battles([row(0, "leftvote", "see ${DOC}")])


def test_unknown_row_types_and_empty_sides_are_refused_not_skipped():
    with pytest.raises(arena.ArenaError, match="type 'upvote'"):
        arena.load_battles([row(0, "upvote")])
    with pytest.raises(arena.ArenaError, match="type None"):  # a page row that was never unwrapped
        arena.load_battles([{"row_idx": 0}])
    for missing in ({"1_output": [["query 0"]]}, {"1_output": None}, {"0_output": []}):
        with pytest.raises(arena.ArenaError, match="shows no document"):
            arena.load_battles([row(0, "leftvote") | missing])
    assert arena.battle_of(row(1, "share") | {"0_output": None, "1_output": None}, 1) is None  # share: never judged
    with pytest.raises(arena.ArenaError, match="top_k"):
        arena.judge_battles(MockLLM(), arena.load_battles(ROWS[:1]), top_k=0)


class Counting(MockLLM):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def chat(self, messages, temperature=0.0):
        self.calls += 1
        return super().chat(messages, temperature)


def test_judge_battles_on_the_mock_judge():
    battles = arena.load_battles(ROWS)
    assert len(battles) == 6
    client = Counting()
    verdicts = arena.judge_battles(client, battles, top_k=10, workers=1)
    assert [v["battle_id"] for v in verdicts] == [b["battle_id"] for b in battles]
    assert [v["vote"] for v in verdicts] == [b["vote"] for b in battles]
    assert all(v["score_a"] in (0.0, 0.25, 0.5, 0.75, 1.0) for v in verdicts)
    assert client.calls == 2 * 5  # both orders for every battle except the identical one
    identical = verdicts[4]
    assert identical["score_a"] == 0.5 and identical["raw"] == ["identical"] and identical["parsed_ok"] == []
    assert all(v["parsed_ok"] == [True, True] for v in verdicts if v is not identical)
    assert all(v["model_a"] == "m/left" and v["model_b"] == "m/right" for v in verdicts)
    # workers change nothing: same verdicts in the same order
    assert arena.judge_battles(MockLLM(), battles, workers=4) == verdicts


def test_top_k_truncates_each_side():
    long = row(0, "leftvote") | {"0_output": [["query 0", "a1", "a2", "a3"]]}
    b = arena.load_battles([long])[0]
    assert b["docs_a"] == ["a1", "a2", "a3"]
    assert arena.ranked(b, "a", 2).doc_texts == ["a1", "a2"] and arena.ranked(b, "b", 2).doc_texts == ["document b0"]
    assert arena.ranked(b, "a", 2).doc_ids == [arena._doc_id("a1"), arena._doc_id("a2")]


def verdict(i: int, vote: str, score_a: float) -> dict:
    return {"battle_id": f"b{i}", "model_a": "x", "model_b": "y", "vote": vote, "score_a": score_a}


# 8 decisive votes (5 left, 3 right), 2 ties, 1 both bad. The judge: right on 5 (3 of the left votes at
# 1.0, 2 of the right votes at 0.0), wrong on 1 left vote (0.0), abstains (0.5) on 1 left and 1 right;
# tied on 1 of the 2 tie votes and on the both-bad vote.
#   committed 6, p = 5/6, S = 2p - 1 = 2/3, cells AA 3, AB 1, BA 0, BB 2
#   Cohen: p_o = 5/6, p_e = (4/6)(3/6) + (2/6)(3/6) = 1/2, kappa = (5/6 - 1/2) / (1/2) = 2/3
HAND = [
    verdict(0, "A", 1.0),
    verdict(1, "A", 1.0),
    verdict(2, "A", 0.75),
    verdict(3, "A", 0.0),
    verdict(4, "A", 0.5),
    verdict(5, "B", 0.0),
    verdict(6, "B", 0.25),
    verdict(7, "B", 0.5),
    verdict(8, "tie", 0.5),
    verdict(9, "tie", 1.0),
    verdict(10, "both_bad", 0.5),
]


def test_kappa_by_hand():
    r = arena.kappa(HAND, bootstrap=300, seed=0)
    assert r["n_battles"] == 11 and r["n_decisive"] == 8 and r["n_committed"] == 6 and r["n_abstain"] == 2
    assert r["n_tie_votes"] == 2 and r["n_both_bad_votes"] == 1
    assert r["committed_agreement"] == pytest.approx(5 / 6)
    assert r["s_committed"] == pytest.approx(2 / 3) and r["kappa_committed"] == r["s_committed"]
    assert r["cohen_kappa_committed"] == pytest.approx(2 / 3)
    assert r["cells"] == {"AA": 3, "AB": 1, "BA": 0, "BB": 2}
    assert r["human_ties_judge_tied"] == pytest.approx(2 / 3)
    lo, hi = r["s_committed_ci95"]
    assert -1.0 <= lo <= r["s_committed"] <= hi <= 1.0
    assert r["tier"] == rel.tier_of(lo) and r["tier_of_point_estimate"] == "A" and r["tier_basis"] == "ci_lower_bound"
    assert arena.kappa(HAND, bootstrap=300, seed=0)["s_committed_ci95"] == [lo, hi]  # seeded: reproducible
    assert "kappa (S = 2p - 1) 0.667" in arena.format_readout(r)


def test_kappa_with_nothing_committed():
    r = arena.kappa([verdict(0, "A", 0.5), verdict(1, "tie", 0.5)], bootstrap=10)
    assert r["n_decisive"] == 1 and r["n_committed"] == 0 and r["n_abstain"] == 1
    assert r["committed_agreement"] is None and r["s_committed"] is None and r["s_committed_ci95"] == [None, None]
    assert r["tier"] is None and r["human_ties_judge_tied"] == 1.0
    assert "n/a" in arena.format_readout(r)


def test_kappa_refuses_an_unknown_vote_label():
    with pytest.raises(arena.ArenaError, match="vote 'left'"):
        arena.kappa([verdict(0, "left", 1.0)], bootstrap=5)
    with pytest.raises(arena.ArenaError, match="vote 'share'"):
        arena.kappa([verdict(0, "A", 1.0), verdict(1, "share", 0.5)], bootstrap=5)


def test_parse_failures_are_counted_not_hidden_in_abstain():
    assert arena.kappa(HAND, bootstrap=5)["n_parse_failed"] == 0  # hand verdicts carry no parsed_ok
    verdicts = [
        verdict(0, "A", 0.5) | {"parsed_ok": [False, False]},  # both orders unparsed: an abstain
        verdict(1, "A", 0.75) | {"parsed_ok": [True, False]},  # one order unparsed, still committed
        verdict(2, "tie", 0.5) | {"parsed_ok": [True, True]},
        verdict(3, "A", 0.5) | {"parsed_ok": []},  # identical sides, no judge call: not a failure
    ]
    r = arena.kappa(verdicts, bootstrap=5)
    assert r["n_parse_failed"] == 2 and r["n_abstain"] == 2 and r["n_committed"] == 1
    assert "did not parse (scored tie) 2" in arena.format_readout(r)


class Unparseable(MockLLM):
    def chat(self, messages, temperature=0.0):
        return "not json"


def test_judge_battles_flags_unparseable_answers(caplog):
    battles = arena.load_battles(ROWS[:2])
    with caplog.at_level("WARNING", logger="analysis.arena"):
        verdicts = arena.judge_battles(Unparseable(), battles, workers=1)
    assert all(v["score_a"] == 0.5 and v["parsed_ok"] == [False, False] for v in verdicts)
    assert "2/2 verdicts have a judge answer that did not parse" in caplog.text
    assert arena.kappa(verdicts, bootstrap=5)["n_parse_failed"] == 2


def test_main_dry_run(tmp_path, capsys):
    battles = tmp_path / "battles.json"
    battles.write_text(json.dumps(ROWS), encoding="utf-8")
    out = tmp_path / "arena.json"
    arena.main(["--battles", str(battles), "--judge-model", "mock", "--out", str(out), "--bootstrap", "20"])
    saved = json.loads(out.read_text())
    assert saved["judge_model"] == "mock" and len(saved["verdicts"]) == 6 and saved["readout"]["n_battles"] == 6
    assert "battles 6: decisive 4" in capsys.readouterr().out
    with pytest.raises(arena.ArenaError):
        arena.main(["--battles", str(battles), "--judge-model", "mock", "--out", str(out), "--limit", "0"])


def fixture_battles(out: Path, record: dict) -> tuple[list[dict], list[dict]]:
    """Battles from a record's own pairwise verdicts: the query and the top-k document ids of each side (no corpus
    text offline, so the id stands in for the text), voted the way the record's judge scored the pair. The second
    list re-uses the record's scores as verdicts, so kappa() against these votes must come out at 1."""
    c = record["config"]
    battles, verdicts = [], []
    for a, b in itertools.combinations(c["models"], 2):
        path = rel.verdict_file(out, record, a, b)
        if not path.exists():  # the run ordered the pair the other way round
            a, b = b, a
            path = rel.verdict_file(out, record, a, b)
        hits = {m: json.loads(rel.prediction_file(out, record, m).read_text())["default"]["test"] for m in (a, b)}
        for v in json.loads(path.read_text()):
            docs = {m: sorted(hits[m][v["qid"]], key=hits[m][v["qid"]].get, reverse=True)[: c["top_k"]] for m in (a, b)}
            bid = f"{v['qid']}__{rel.slug(a)}__{rel.slug(b)}"
            vote = rel.judge_winner(float(v["score_a"]))
            battles.append(
                {
                    "battle_id": bid,
                    "query": v["query"],
                    "model_a": a,
                    "model_b": b,
                    "docs_a": [f"doc {d}" for d in docs[a]],
                    "docs_b": [f"doc {d}" for d in docs[b]],
                    "vote": vote,
                }
            )
            verdicts.append({"battle_id": bid, "model_a": a, "model_b": b, "vote": vote, "score_a": v["score_a"]})
    return battles, verdicts


@pytest.mark.skipif(not FIXTURES, reason="ANALYSIS_FIXTURES not set")
def test_fixture_record_as_battles():
    out = Path(FIXTURES)
    records = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    record = next(r for r in records if r["config"]["arm"] == "original")
    battles, own = fixture_battles(out, record)
    n_pairs = len(record["config"]["models"]) * (len(record["config"]["models"]) - 1) // 2
    assert len(battles) == n_pairs * record["config"]["n_queries"]
    assert all(len(b["docs_a"]) <= record["config"]["top_k"] and b["docs_a"] for b in battles)

    # the record's own scores against votes derived from them: perfect agreement, counts from the files
    n_committed = sum(v["score_a"] != 0.5 for v in own)
    r = arena.kappa(own, bootstrap=100, seed=0)
    assert r["n_battles"] == len(battles) and r["n_decisive"] == n_committed == r["n_committed"]
    assert r["n_abstain"] == 0 and r["n_tie_votes"] == len(battles) - n_committed and r["n_both_bad_votes"] == 0
    assert r["s_committed"] == 1.0 and r["committed_agreement"] == 1.0 and r["human_ties_judge_tied"] == 1.0

    # a fresh mock judge over the same battles: one verdict per battle, in order, scores on the judge's scale
    verdicts = arena.judge_battles(MockLLM(), battles, top_k=record["config"]["top_k"], workers=4)
    assert [v["battle_id"] for v in verdicts] == [b["battle_id"] for b in battles]
    assert all(v["score_a"] in (0.0, 0.25, 0.5, 0.75, 1.0) and v["parsed_ok"] == [True, True] for v in verdicts)
    fresh = arena.kappa(verdicts, bootstrap=100, seed=0)
    assert fresh["n_decisive"] == n_committed and fresh["n_committed"] + fresh["n_abstain"] == n_committed
    assert sum(fresh["cells"].values()) == fresh["n_committed"]
