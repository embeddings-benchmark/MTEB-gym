import json
import math

import pytest

import mteb_gym as gym
from mteb_gym.benchmarks import Benchmark, agreement_table, per_task_table, summary_table

BENCH = Benchmark(
    name="Test(v1)",
    tasks=("TaskA", "TaskB"),
    judge="org/judge",
    judge_revision="rev1",
    judging="pointwise",
    query_source="synthetic",
    n_queries=10,
    generator="org/gen+abc",
)


def record(task, ratings, *, labels="seed_documents", judge_revision="rev1", n=10, **config):
    base = {
        "judge_model": "org/judge",
        "judging": "pointwise",
        "query_source": "synthetic",
        "n_queries": n,
        "generator_model": "org/gen+abc",
        "seed": 0,
    }
    return {
        "task_name": task,
        "config": {**base, **config},
        "labels": labels,
        "llms": {"judge": {"model": "org/judge", "revision": judge_revision}},
        "ratings": [
            {"model": m, "rating": r, "ci_low": r - 1, "ci_high": r + 1, "ndcg_at_10": nd} for m, r, nd in ratings
        ],
    }


def write(tmp_path, records):
    for i, rec in enumerate(records):
        path = tmp_path / rec["task_name"] / f"{rec['task_name']}__judge__{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
    return gym.load_results(tmp_path)


def test_the_paper_benchmarks_are_registered():
    names = [b.name for b in gym.get_benchmarks()]
    assert names == ["Gym(eng, v1)", "Gym(eng, v1, pairwise)", "Gym(eng, v1, original queries)"]
    main = gym.get_benchmark("Gym(eng, v1)")
    assert len(main.tasks) == 28 and main.query_source == "synthetic" and main.judging == "pointwise"
    assert gym.get_benchmark("Gym(eng, v1, pairwise)").judging == "pairwise"
    assert gym.get_benchmark("Gym(eng, v1, original queries)").generator is None
    with pytest.raises(KeyError, match="Did you mean: Gym"):
        gym.get_benchmark("Gym(eng,v1)")


@pytest.mark.parametrize(
    "change",
    [
        {"judge_model": "org/other"},
        {"judging": "pairwise"},
        {"query_source": "original", "generator_model": None},
        {"generator_model": "org/gen+def"},
        {"seed": 1},
        {"n": 9},
        {"judge_revision": "rev2"},
        {"judge_revision": None},
    ],
)
def test_a_record_must_match_every_field(change):
    assert BENCH.matches(record("TaskA", []))
    assert not BENCH.matches(record("TaskA", [], **change))
    assert not BENCH.matches(record("TaskC", []))


def test_a_task_with_fewer_original_queries_uses_all_of_them():
    original = Benchmark(
        name="O", tasks=("TaskA",), judge="org/judge", judging="pointwise", query_source="original", n_queries=10
    )
    assert original.matches(record("TaskA", [], query_source="original", generator_model=None, n=7))
    assert not original.matches(record("TaskA", [], query_source="original", generator_model=None, n=11))


def test_the_summary_ranks_models_by_their_mean_and_leaves_a_missing_task_missing(tmp_path):
    results = write(
        tmp_path,
        [
            record("TaskA", [("m1", 80, 0.5), ("m2", 70, 0.4), ("m3", 60, 0.3)]),
            record("TaskB", [("m1", 60, 0.5), ("m2", 90, 0.4)]),
        ],
    )
    summary = summary_table(results, BENCH)
    assert list(summary.columns) == ["Rank", "Model", "Mean", "Tasks"]
    assert summary["Model"].tolist() == ["m2", "m1", "m3"]
    assert summary["Mean"].tolist()[:2] == [80, 70] and math.isnan(summary["Mean"].iloc[2])
    assert summary["Rank"].tolist()[:2] == [1, 2] and summary["Rank"].isna().iloc[2]
    assert summary["Tasks"].tolist() == [2, 2, 1]
    per_task = per_task_table(results, BENCH)
    assert per_task.at["m1", "TaskA"] == 80 and math.isnan(per_task.at["m3", "TaskB"])
    assert per_task_table(results, BENCH, value="ci_low").at["m2", "TaskB"] == 89
    assert results.get_benchmark_result(BENCH).equals(summary)


def test_a_task_without_a_record_leaves_every_mean_missing(tmp_path):
    results = write(tmp_path, [record("TaskA", [("m1", 80, 0.5), ("m2", 70, 0.4)])])
    summary = summary_table(results, BENCH)
    assert summary["Mean"].isna().all() and summary["Tasks"].tolist() == [1, 1]
    assert list(per_task_table(results, BENCH).columns) == ["TaskA", "TaskB"]


def test_two_records_for_one_task_are_an_error(tmp_path):
    results = write(tmp_path, [record("TaskA", [("m1", 80, 0.5)]), record("TaskA", [("m1", 70, 0.5)])])
    with pytest.raises(ValueError, match="TaskA has 2 matching records"):
        summary_table(results, BENCH)


def test_of_original_records_the_one_with_the_most_queries_is_used(tmp_path):
    original = Benchmark(
        name="O", tasks=("TaskA",), judge="org/judge", judging="pointwise", query_source="original", n_queries=10
    )
    results = write(
        tmp_path,
        [
            record("TaskA", [("m1", 50, 0.5)], query_source="original", generator_model=None, n=5),
            record("TaskA", [("m1", 70, 0.5)], query_source="original", generator_model=None, n=8),
        ],
    )
    assert per_task_table(results, original).at["m1", "TaskA"] == 70


def test_agreement_compares_ratings_with_labelled_ndcg(tmp_path):
    original = Benchmark(
        name="O",
        tasks=("TaskA", "TaskB"),
        judge="org/judge",
        judging="pointwise",
        query_source="original",
        n_queries=10,
    )
    ms = [("m1", 90, 0.9), ("m2", 80, 0.8), ("m3", 70, 0.7), ("m4", 60, 0.6)]
    results = write(
        tmp_path,
        [
            record("TaskA", ms, labels="dataset", query_source="original", generator_model=None),
            record("TaskB", ms, labels=None, query_source="original", generator_model=None),
        ],
    )
    table = agreement_table(results, original, bootstrap=50)
    assert table["Task"].tolist() == ["TaskA"]
    assert table["Spearman"].iloc[0] == pytest.approx(1.0) and table["Models"].iloc[0] == 4
