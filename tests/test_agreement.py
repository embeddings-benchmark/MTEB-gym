import json
import tempfile
from pathlib import Path

from mteb_gym import Result, agreement, load_results


def test_correlate():
    g = {f"m{i}": float(i) for i in range(5)}
    res = agreement.correlate(g, dict(g), bootstrap=50)
    assert abs(res["spearman_rho"] - 1.0) < 1e-9
    import numpy as np

    truth = {f"m{i}": float(i) for i in range(25)}
    top = list(range(15, 25))
    np.random.default_rng(0).shuffle(top)
    gym = {f"m{i}": float(v) for i, v in zip(range(15, 25), top)} | {f"m{i}": float(i) for i in range(15)}
    out = agreement.correlate(gym, truth, bootstrap=0)
    assert out["spearman_rho"] > 0.85 and abs(out["spearman_top10"]) < 0.6  # strong models shuffled among themselves
    a = np.arange(10, dtype=float)
    assert agreement._tau_ap(a, a) == 1.0 and agreement._tau_ap(-a, a) == -1.0


def test_agreement():
    original = agreement.fetch_truth
    agreement.fetch_truth = lambda models, task, **kw: (
        {"model_a": 30.0, "model_b": 20.0, "model_c": 10.0},
        {m: {"model_revision": "r1", "official": True} for m in models},
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = {
                "task_name": "NFCorpus",
                "source": "mteb",
                "config": {},
                "diagnostics": {},
                "ratings": [
                    {"model": m, "rating": r} for m, r in (("model_a", 1100), ("model_b", 1000), ("model_c", 900))
                ],
            }
            res = Result(rec, Path(tmp) / "NFCorpus" / "NFCorpus__judge__original-queries.json")
            res.to_disk()
            agr = res.agreement(bootstrap=100, seed=0)
            assert agr["spearman_rho"] == 1.0 and agr["kendall_tau"] == 1.0
            assert Result.from_disk(res.path).record["agreement"]["truth_source"] == {
                m: {"model_revision": "r1", "official": True} for m in ("model_a", "model_b", "model_c")
            }
            assert load_results(tmp).agreement(bootstrap=10)[str(res.path)]["spearman_rho"] == 1.0
            assert (
                Result({"source": "local", "task_name": "x", "ratings": []})
                .agreement()["error"]
                .startswith("local corpus")
            )
    finally:
        agreement.fetch_truth = original


def test_official_scores_across_revisions():
    """The results repository files many models' scores under "external", not under the revision
    mteb pins; the lookup must find those, and prefer the pinned one when both exist."""
    import mteb
    from mteb.results import TaskResult

    meta = mteb.get_model_meta("BAAI/bge-base-en-v1.5")

    def cache_with(revisions):
        tmp = tempfile.mkdtemp()
        for revision, score in revisions:
            folder = Path(tmp) / "results" / meta.model_name_as_path() / revision
            folder.mkdir(parents=True)
            (folder / "model_meta.json").write_text(json.dumps({"name": meta.name, "revision": revision}))
            TaskResult(
                task_name="NFCorpus",
                dataset_revision="x",
                mteb_version="2.20.10",
                evaluation_time=1.0,
                scores={"test": [{"main_score": score, "hf_subset": "default", "languages": ["eng-Latn"]}]},
            ).to_disk(folder / "NFCorpus.json")
        return mteb.ResultCache(cache_path=tmp)

    external_only = cache_with([("external", 0.40)])
    assert external_only.load_task_result("NFCorpus", meta) is None  # mteb's per-revision lookup: nothing
    assert agreement.official_scores(external_only, "NFCorpus", [meta.name]) == {meta.name: (0.40, "external")}

    both = cache_with([("external", 0.40), (meta.revision, 0.55)])
    assert agreement.official_scores(both, "NFCorpus", [meta.name]) == {meta.name: (0.55, meta.revision)}

    assert agreement.official_scores(external_only, "SciFact", [meta.name]) == {}
