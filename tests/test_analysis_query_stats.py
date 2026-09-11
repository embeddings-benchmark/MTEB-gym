"""Tests for analysis.query_stats: hand-built strings and a toy run folder (no mteb, no network), plus the
mock run under $ANALYSIS_FIXTURES end to end when that variable is set."""

import json
import os
from pathlib import Path

import pytest

from analysis import query_stats as qs
from mteb_gym import reliability as rel

FIXTURES = os.environ.get("ANALYSIS_FIXTURES")
MODELS = ["m/a", "m/b"]


# ----------------------------------------------------------------------------- hand-built strings
def test_word_count_and_question_flag():
    cases = [
        ("What is the role of p53 in apoptosis?", 8, True),
        ("role of p53 in apoptosis", 5, False),
        ("Does aspirin reduce stroke risk", 5, True),  # question word, no question mark
        ("aspirin and stroke risk?", 4, True),  # question mark, no question word
        ("Whatever happened to cold fusion", 5, False),  # "whatever" is not "what"
        ("  How  ", 1, True),
        ("Which?", 1, True),
        ("", 0, False),
    ]
    for text, n, flag in cases:
        assert qs.word_count(text) == n, text
        assert qs.is_question(text) is flag, text


def test_describe():
    texts = {
        "q0": "What is the role of p53 in apoptosis?",
        "q1": "role of p53 in apoptosis",
        "q2": "Does aspirin reduce stroke risk",
        "q3": "aspirin and stroke risk?",
    }
    d = qs.describe(texts)
    assert d == {"count": 4, "mean_words": 5.5, "median_words": 5, "question_share": 0.75}
    assert qs.describe({}) == {"count": 0, "mean_words": None, "median_words": None, "question_share": None}


def test_copied_share_hand_computed():
    # content words: cadherin, mediated, cell, adhesion, epithelium (in, the are stopwords)
    query = "Cadherin mediated cell adhesion in the epithelium"
    docs = ["Cadherins are calcium dependent cell adhesion molecules.", "The epithelium forms a barrier."]
    assert qs.content_words(query) == {"cadherin", "mediated", "cell", "adhesion", "epithelium"}
    assert qs.copied_share(query, docs) == pytest.approx(3 / 5)  # cell, adhesion, epithelium; cadherin != cadherins
    assert qs.copied_share("what is the", docs) is None  # stopwords only
    assert qs.copied_share("Cadherin cadherin CADHERIN", ["cadherin"]) == 1.0  # a word counts once


def test_copied_word_share_aggregates():
    docs = {"d1": "Cadherins are calcium dependent cell adhesion molecules.", "d2": "The epithelium forms a barrier."}
    docs["d3"] = "p53 tumor suppressor"
    queries = [
        {"qid": "q0", "text": "Cadherin mediated cell adhesion in the epithelium", "seed_doc_ids": ["d1", "d2"]},
        {"qid": "q1", "text": "what is the p53 role", "seed_doc_ids": ["d3"]},  # p53 in doc, role not: 1/2
        {"qid": "q2", "text": "what is the", "seed_doc_ids": ["d3"]},  # no content words
        {"qid": "q3", "text": "anything at all", "seed_doc_ids": ["d9"]},  # seed doc not in docs
    ]
    r = qs.copied_word_share(queries, docs)
    assert r["mean_per_query"] == pytest.approx((3 / 5 + 1 / 2) / 2)
    assert r["pooled"] == pytest.approx(4 / 7)
    assert (r["n_scored"], r["n_missing_seed_docs"], r["n_no_content"]) == (2, 1, 1)
    without = qs.copied_word_share(queries, None)
    assert without["mean_per_query"] is None and without["pooled"] is None  # null, never 0


def test_quality_and_filter_accounting():
    queries = [{"quality": 5}, {"quality": 3}, {"quality": 5}, {"quality": None}]
    q = qs.quality_distribution(queries)
    assert q.pop("note") == qs.QUALITY_NOTE and "defaulted" in qs.QUALITY_NOTE
    assert q == {"counts": {"3": 1, "5": 2}, "mean": pytest.approx(13 / 3), "n_scored": 3, "n_unscored": 1}
    config = {"n_queries": 10, "n_queries_generated": 16, "gen_filter": True, "gen_min_score": 3, "gen_dedup": 0.8}
    f = qs.filter_accounting(16, 10, config)
    assert (f["n_generated"], f["n_kept"], f["n_dropped"], f["drop_rate"]) == (16, 10, 6, 0.375)
    assert f["n_generated_in_record"] == 16
    assert "does not record why" in f["note"]
    unknown = qs.filter_accounting(None, 10, {})
    assert unknown["n_dropped"] is None and unknown["drop_rate"] is None
    assert qs.filter_accounting(None, 10, config)["n_generated_in_record"] == 16  # file silent, record not
    with pytest.raises(ValueError, match="rewritten after the run"):
        qs.filter_accounting(20, 10, config)  # file and record disagree


def test_load_texts_rows_and_refusals(tmp_path):
    p = tmp_path / "docs.json"
    p.write_text(json.dumps({"d1": "plain text", 7: {"title": "A title", "text": "and body"}, "d3": {"text": "only"}}))
    assert qs.load_texts(p, "docs") == {"d1": "plain text", "7": "A title and body", "d3": "only"}
    p.write_text(json.dumps({"d1": {"body": "no text key"}}))
    with pytest.raises(ValueError, match="not text"):
        qs.load_texts(p, "docs")
    p.write_text(json.dumps({"d1": ["a", "list"]}))
    with pytest.raises(ValueError, match="not text"):
        qs.load_texts(p, "docs")
    p.write_text(json.dumps([["d1", "text"]]))
    with pytest.raises(ValueError, match="JSON object"):
        qs.load_texts(p, "docs")
    p.write_text(json.dumps({"o1": "What is ${topic}?"}))
    with pytest.raises(ValueError, match="template-looking"):
        qs.load_texts(p, "original queries")
    assert qs.load_texts(p, "docs") == {"o1": "What is ${topic}?"}  # only queries are checked for templates


# ----------------------------------------------------------------------------- a toy run folder
def write_verdicts(path: Path, a: str, b: str, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"qid": q, "query": q, "model_a": a, "model_b": b, "score_a": 1.0} for q in qids]
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        path.write_text(json.dumps(rows))


def write_run(
    root: Path, queries: list[dict], n_generated: int | None, arm: str = "synthetic", models: list[str] = MODELS
) -> Path:
    """A record plus its queries, prediction and verdict files; the first pair is stored the other way round."""
    config = {
        "arm": arm,
        "query_set": "corpus-gen-abc",
        "judge_model": "judge-x",
        "generator_model": "gen-y",
        "judge_system": "You compare two retrieval systems.",
        "top_k": 10,
        "n_queries": 3,
        "n_queries_generated": n_generated,
        "gen_filter": True,
        "gen_min_score": 3,
        "gen_dedup": 0.8,
        "models": models,
        "model_revisions": {m: "rev1" for m in models},
        "config_hash": "abcd1234",
    }
    record = {
        "task_name": "ToyRetrieval",
        "source": "mteb",
        "config": config,
        "diagnostics": {},
        "ratings": [{"model": m, "rating": 1000.0, "ci_low": 990.0, "ci_high": 1010.0} for m in models],
    }
    qf = qs.queries_file(root, record)
    qf.parent.mkdir(parents=True)
    qf.write_text(json.dumps({"n_generated": n_generated, "queries": queries}))
    qids = [q["qid"] for q in queries]
    for m in models:
        p = rel.prediction_file(root, record, m)
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"default": {"test": {q: {"d1": 1.0} for q in qids}}}))
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            first, second = (b, a) if (a, b) == (models[0], models[1]) else (a, b)  # first pair reversed
            write_verdicts(rel.verdict_file(root, record, first, second), first, second, qids)
    path = root / "records" / "ToyRetrieval__judge-x__gen-y__q3-s0-abcd1234.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return path


TOY_QUERIES = [
    {"qid": "q0", "text": "What is the role of p53 in apoptosis?", "seed_doc_ids": ["d1"], "quality": 5},
    {
        "qid": "q1",
        "text": "Cadherin mediated cell adhesion in the epithelium",
        "seed_doc_ids": ["d2", "d3"],
        "quality": 3,
    },
    {"qid": "q2", "text": "aspirin and stroke risk?", "seed_doc_ids": ["d4"], "quality": 4},
]
TOY_DOCS = {
    "d1": "p53 is a tumor suppressor that triggers apoptosis.",  # p53, apoptosis of {role, p53, apoptosis}: 2/3
    "d2": "Cadherins are calcium dependent cell adhesion molecules.",
    "d3": "The epithelium forms a barrier.",  # with d2: 3/5
    "d4": "Low dose aspirin lowers stroke risk in trials.",  # aspirin, stroke, risk: 3/3
}


def test_toy_run_known_answers(tmp_path):
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5)
    s = qs.query_stats(tmp_path, path, docs=TOY_DOCS)["synthetic"]
    # 8 + 7 + 4 words; q0 and q2 are questions
    assert s["count"] == 3 and s["median_words"] == 7
    assert s["mean_words"] == pytest.approx(19 / 3) and s["question_share"] == pytest.approx(2 / 3)
    assert s["copied_word_share"]["mean_per_query"] == pytest.approx((2 / 3 + 3 / 5 + 1.0) / 3)
    assert s["copied_word_share"]["pooled"] == pytest.approx(8 / 11)
    assert s["quality"]["counts"] == {"3": 1, "4": 1, "5": 1}
    assert (s["filter"]["n_dropped"], s["filter"]["drop_rate"]) == (2, 0.4)
    cov = s["coverage"]
    assert (cov["n_qids_predicted"], cov["n_qids_judged"], cov["n_kept_not_judged"]) == (3, 3, 0)
    assert (cov["n_pairs_expected"], cov["n_pairs_found"], cov["n_kept_not_judged_in_some_pair"]) == (1, 1, 0)
    assert cov["missing_files"] == []


def test_coverage_partial_pair_missing_files_and_jsonl(tmp_path):
    models = ["m/a", "m/b", "m/c"]
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5, models=models)
    record = json.loads(path.read_text())
    qids = [q["qid"] for q in TOY_QUERIES]
    # (a, c): the run stopped early, only the .jsonl stream exists and it lacks q2
    ac = rel.verdict_file(tmp_path, record, "m/a", "m/c")
    ac.unlink()
    write_verdicts(ac.with_suffix(".jsonl"), "m/a", "m/c", qids[:2])
    # (b, c): never judged; m/c: never predicted
    rel.verdict_file(tmp_path, record, "m/b", "m/c").unlink()
    rel.prediction_file(tmp_path, record, "m/c").unlink()
    cov = qs.query_stats(tmp_path, path)["synthetic"]["coverage"]
    assert (cov["n_models"], cov["n_pairs_expected"], cov["n_pairs_found"]) == (3, 3, 2)
    assert (cov["n_qids_predicted"], cov["n_kept_not_predicted"]) == (3, 0)  # the other two models cover them
    assert (cov["n_qids_judged"], cov["n_kept_not_judged"]) == (3, 0)  # the union hides the gap
    assert cov["n_kept_not_judged_in_some_pair"] == 1  # this does not
    assert cov["missing_files"] == [
        str(rel.prediction_file(tmp_path, record, "m/c")),
        str(rel.verdict_file(tmp_path, record, "m/b", "m/c")),
    ]


def test_stale_queries_file_is_refused(tmp_path):
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5)
    qf = qs.queries_file(tmp_path, json.loads(path.read_text()))
    qf.write_text(json.dumps({"n_generated": 9, "queries": TOY_QUERIES}))
    with pytest.raises(ValueError, match="rewritten after the run"):
        qs.query_stats(tmp_path, path)


def test_toy_run_without_docs_and_with_original(tmp_path):
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5)
    original = {"o1": "Is p53 a tumor suppressor?", "o2": "cadherin function"}
    stats = qs.query_stats(tmp_path, path, original_queries=original)
    assert stats["docs_given"] is False
    assert stats["synthetic"]["copied_word_share"]["mean_per_query"] is None  # null, not 0
    assert stats["original"]["count"] == 2 and stats["original"]["question_share"] == 0.5
    assert stats["original"]["copied_word_share"]["pooled"] is None
    table = qs.markdown_table(stats)
    assert "| queries | 3 | 2 |" in table
    assert "| copied-word share, mean per query | null | null |" in table


def test_toy_run_refuses_other_arms_and_templates(tmp_path):
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5, arm="original")
    with pytest.raises(ValueError, match="synthetic-arm"):
        qs.query_stats(tmp_path, path)
    bad = [dict(TOY_QUERIES[0], text="What is ${topic}?")]
    path = write_run(tmp_path / "bad", bad, n_generated=1)
    with pytest.raises(ValueError, match="template-looking"):
        qs.query_stats(tmp_path / "bad", path)


def test_main_writes_json_and_markdown(tmp_path, capsys):
    path = write_run(tmp_path, TOY_QUERIES, n_generated=5)
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(TOY_DOCS))
    out = tmp_path / "stats" / "query_stats.json"
    qs.main(["--output-folder", str(tmp_path), "--record", str(path), "--docs", str(docs), "--out", str(out)])
    stats = json.loads(out.read_text())
    assert stats["synthetic"]["copied_word_share"]["pooled"] == pytest.approx(8 / 11)
    assert out.with_suffix(".md").read_text().startswith("| metric | synthetic (gen-y) | original |")
    assert "not given" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the mock run
def _fixture_record() -> tuple[Path, Path]:
    root = Path(FIXTURES)
    for p in sorted(root.glob("records/*.json")):
        if json.loads(p.read_text())["config"].get("arm") == "synthetic":
            return root, p
    raise AssertionError(f"no synthetic-arm record under {root}")


@pytest.mark.skipif(not FIXTURES, reason="ANALYSIS_FIXTURES not set")
def test_fixture_without_docs():
    root, record = _fixture_record()
    stats = qs.query_stats(root, record)
    s = stats["synthetic"]
    assert stats["task_name"] == "NanoSciFactRetrieval"
    assert s["count"] == 10 and s["filter"]["n_generated"] == 16
    assert (s["filter"]["n_dropped"], s["filter"]["drop_rate"]) == (6, 0.375)
    assert s["copied_word_share"]["mean_per_query"] is None
    assert s["quality"]["n_unscored"] == 0 and sum(s["quality"]["counts"].values()) == 10
    cov = s["coverage"]
    assert (cov["n_models"], cov["n_pairs_expected"], cov["n_pairs_found"]) == (3, 3, 3)
    assert cov["missing_files"] == []
    assert (cov["n_kept_not_predicted"], cov["n_kept_not_judged"], cov["n_judged_not_kept"]) == (0, 0, 0)
    assert cov["n_kept_not_judged_in_some_pair"] == 0


@pytest.mark.skipif(not FIXTURES, reason="ANALYSIS_FIXTURES not set")
def test_fixture_with_docs(tmp_path):
    root, record = _fixture_record()
    queries = json.loads(qs.queries_file(root, json.loads(record.read_text())).read_text())["queries"]
    docs: dict[str, str] = {}
    for q in queries:  # every seed document carries its query's text, so every content word is copied
        for d in q["seed_doc_ids"]:
            docs[d] = docs.get(d, "") + " " + q["text"]
    stats = qs.query_stats(root, record, docs=docs)
    share = stats["synthetic"]["copied_word_share"]
    assert share["n_scored"] == 10 and share["n_missing_seed_docs"] == 0
    assert share["mean_per_query"] == 1.0 and share["pooled"] == 1.0
    gone = queries[0]["seed_doc_ids"][0]
    affected = sum(gone in q["seed_doc_ids"] for q in queries)
    partial = qs.query_stats(root, record, docs={d: t for d, t in docs.items() if d != gone})["synthetic"]
    assert partial["copied_word_share"]["n_missing_seed_docs"] == affected >= 1
    assert partial["copied_word_share"]["n_scored"] == 10 - affected
    out = tmp_path / "q.json"
    docs_path = tmp_path / "docs.json"
    docs_path.write_text(json.dumps(docs))
    qs.main(["--output-folder", str(root), "--record", str(record), "--docs", str(docs_path), "--out", str(out)])
    assert json.loads(out.read_text())["docs_given"] is True
