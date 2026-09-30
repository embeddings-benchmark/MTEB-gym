"""Submit records to the results repository on GitHub, and the files behind them to the dataset on Hugging Face.

As mteb's ResultCache.submit_results: the results repository is cloned into the cache, new records are
committed there, and with create_pr=True the branch is pushed to the submitter's fork and a pull
request opened. The query set, predictions and verdicts a record was computed from are too large
for git; they go to the dataset as a pull request.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

from .results import load_results
from .run import cache_files, default_cache_folder

logger = logging.getLogger(__name__)

RESULTS_REPOSITORY = "https://github.com/embeddings-benchmark/gym-results"
DATASET = "mteb/gym-runs"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def submit(
    results_folder: str | Path = "results",
    *,
    cache_folder: str | Path | None = None,
    create_pr: bool = False,
    repository: str = RESULTS_REPOSITORY,
    dataset: str = DATASET,
) -> dict:
    """Commit the records in `results_folder` that the results repository lacks, and with `create_pr`
    open a pull request there and one on the dataset with their queries and verdicts. Without it,
    the commit is prepared locally and nothing leaves this machine.

    Returns:
        The records committed, the dataset files they were computed from, the local clone and branch,
        and the pull request URLs when opened.
    """
    cache = Path(cache_folder) if cache_folder is not None else default_cache_folder()
    clone = cache / "remote" / "gym-results"
    if clone.exists():
        _git("fetch", "--quiet", "origin", cwd=clone)
        _git("checkout", "--quiet", "--force", "-B", "main", "origin/main", cwd=clone)
        _git("clean", "--quiet", "-fd", cwd=clone)  # drop records a failed call copied in
    else:
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", repository, str(clone)], check=True)

    new = []
    for result in load_results(results_folder).results:
        target = clone / "results" / result.record["task_name"] / result.path.name
        if target.exists() and filecmp.cmp(result.path, target, shallow=False):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result.path, target)
        new.append(target.relative_to(clone))
    if not new:
        logger.info("nothing to submit: every record is already in the results repository")
        return {"records": [], "files": []}

    files = [
        p
        for record in new
        for group in cache_files(json.loads((clone / record).read_text()), cache).values()
        for p in group
    ]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files the records were computed from are not in {cache}: {missing[:3]}"
        )

    branch = "submit-" + hashlib.sha256("".join(sorted(map(str, new))).encode()).hexdigest()[:10]
    _git("checkout", "--quiet", "-B", branch, cwd=clone)
    _git("add", "results", cwd=clone)
    _git("commit", "--quiet", "-m", f"Add {len(new)} records", cwd=clone)
    out = {"records": new, "files": files, "clone": clone, "branch": branch}
    if not create_pr:
        logger.info(
            "prepared, nothing uploaded: %d records committed on branch %s of %s (for %s), and %d files for %s; "
            "create_pr=True opens both pull requests",
            len(new),
            branch,
            clone,
            repository,
            len(files),
            dataset,
        )
        return out

    from huggingface_hub import CommitOperationAdd, HfApi

    # Submitters need not have write access to the results repository, so the branch goes to their fork.
    if "fork" not in _git("remote", cwd=clone).split():
        subprocess.run(
            ["gh", "repo", "fork", "--remote", "--remote-name", "fork"], cwd=clone, check=True, capture_output=True
        )
    _git("push", "--quiet", "--force", "fork", f"HEAD:refs/heads/{branch}", cwd=clone)
    user = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"], check=True, capture_output=True, text=True
    ).stdout.strip()
    title = f"Add {len(new)} records"
    body = "\n".join(f"- `{r}`" for r in new)
    upstream = repository.removeprefix("https://github.com/").removesuffix(".git")
    out["pr_url"] = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            upstream,
            "--base",
            "main",
            "--head",
            f"{user}:{branch}",
            "--title",
            title,
            "--body",
            body,
        ],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = HfApi().create_commit(
        repo_id=dataset,
        repo_type="dataset",
        operations=[CommitOperationAdd(str(p.relative_to(cache)), p) for p in files],
        commit_message=f"{title}: queries, predictions and verdicts",
        commit_description=out["pr_url"],
        create_pr=True,
    )
    out["dataset_pr_url"] = commit.pr_url
    return out
