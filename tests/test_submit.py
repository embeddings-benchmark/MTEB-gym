import json
import tempfile
from pathlib import Path

import pytest


def test_submit_prepares_a_commit(monkeypatch):
    """Without create_pr, new records are committed to a clone of the results repository, and the
    cached files they were computed from are listed; nothing leaves the machine. A record whose
    files are not all in the cache is refused."""
    import subprocess

    from mteb_gym import submit
    from mteb_gym.cache import cache_files

    for k, v in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(k, v)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        remote, seed = tmp / "remote.git", tmp / "seed"
        subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(remote)], check=True)
        subprocess.run(["git", "clone", "--quiet", str(remote), str(seed)], check=True)
        (seed / "README.md").write_text("results\n")
        subprocess.run(["git", "add", "."], cwd=seed, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=seed, check=True)
        subprocess.run(["git", "push", "--quiet", "origin", "main"], cwd=seed, check=True)

        record = {
            "task_name": "demo",
            "source": "mteb",
            "config": {
                "judging": "pairwise",
                "pairs_per_query": None,
                "query_source": "original",
                "query_set": "qs",
                "models": ["m/a", "m/b"],
                "model_revisions": {"m/a": "1", "m/b": "1"},
                "judge_model": "mock",
                "judge_system": "s",
                "top_k": 10,
                "doc_chars": 2000,
            },
            "ratings": [],
        }
        results = tmp / "results" / "demo"
        results.mkdir(parents=True)
        (results / "demo__mock__original-queries__q1-s0-abc.json").write_text(json.dumps(record))
        cache = tmp / "cache"
        files = [p for group in cache_files(record, cache).values() for p in group]
        with pytest.raises(FileNotFoundError):  # a record goes up only with everything it was computed from
            submit(tmp / "results", cache_folder=cache, repository=str(remote))
        for p in files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}\n")

        out = submit(tmp / "results", cache_folder=cache, repository=str(remote))
        assert [str(r) for r in out["records"]] == ["results/demo/demo__mock__original-queries__q1-s0-abc.json"]
        assert out["files"] == sorted(files) and "pr_url" not in out
        assert (
            subprocess.run(
                ["git", "log", "-1", "--format=%s"], cwd=out["clone"], capture_output=True, text=True
            ).stdout.strip()
            == "Add 1 records"
        )
        # the results repository still lacks it, so a second call prepares the same commit again
        again = submit(tmp / "results", cache_folder=cache, repository=str(remote))
        assert again["records"] == out["records"] and again["branch"] == out["branch"]

        # a run on a local corpus stays local
        local = {**record, "source": "local"}
        (results / "demo__mock__original-queries__q1-s0-abc.json").write_text(json.dumps(local))
        assert submit(tmp / "results", cache_folder=cache, repository=str(remote))["records"] == []

        # a random subset of pairs leaves the other pairs without a file, which must not block submitting
        subset = {**record, "config": {**record["config"], "pairs_per_query": 1}}
        judgments = cache_files(subset, cache)["judgments"]
        assert len(judgments) == 1
        judgments[0].unlink()
        assert cache_files(subset, cache)["judgments"] == []


def test_a_record_already_in_the_repository_is_not_submitted_again(tmp_path, monkeypatch):
    import subprocess

    from mteb_gym import submit

    for k, v in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(k, v)
    name = "demo__mock__original-queries__q1-s0-abc.json"
    record = json.dumps({"task_name": "demo", "source": "mteb", "ratings": []})
    remote, seed = tmp_path / "remote.git", tmp_path / "seed"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(remote), str(seed)], check=True)
    (seed / "results" / "demo").mkdir(parents=True)
    (seed / "results" / "demo" / name).write_text(record)
    subprocess.run(["git", "add", "."], cwd=seed, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=seed, check=True)
    subprocess.run(["git", "push", "--quiet", "origin", "main"], cwd=seed, check=True)
    (tmp_path / "results" / "demo").mkdir(parents=True)
    (tmp_path / "results" / "demo" / name).write_text(record)
    out = submit(tmp_path / "results", cache_folder=tmp_path / "cache", repository=str(remote))
    assert out == {"records": [], "files": []}
