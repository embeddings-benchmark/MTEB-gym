"""
Judge reliability against a corpus's own labels.

A record from the original-query arm (queries="original") was judged on the dataset's human
queries, and the corpus carries their qrels. Scoring each pairwise verdict against the official
nDCG@k winner of that pair on that query asks a narrower question than rank agreement: does the
judge agree with the human labels on the comparisons it makes, inside this corpus's own query
space? A corpus where it barely beats a coin cannot support a ranking, whatever the ranking says.

    committed_agreement     P(judge picks the qrels winner | judge commits, qrels decisive)
    s_committed             2p - 1: chance-corrected under a uniform-marginal null (Bennett, Alpert and
                            Goldstein's S; Byrt's PABAK for two categories). This is the paper's kappa.
                            It is NOT Cohen's kappa; cohen_kappa_committed is the marginal-corrected companion.
    clear_winner_agreement  the same ratio on pairs the official scores separate by >= CLEAR_MARGIN nDCG

Intervals are a query-clustered bootstrap (queries resampled with replacement, seed pinned). The tier
is read from the interval's lower bound, not the point estimate: A >= 0.40, B >= 0.20, C below. Project
thresholds, not Landis-Koch categories; the leaderboard warns below B.

Everything comes from the run's own artifacts. The record names its verdict and prediction files in
the cache (run.cache_files: the folder passed in, else $MTEB_GYM_CACHE, else ~/.cache/mteb_gym), and
the qrels come from the corpus or are passed in. A comparison where either presentation order failed
to parse is left out and counted (n_unparsed), as rank.rate leaves it out of the ranking: keeping the
order that parsed would bring back the position bias that judging both orders cancels.

The layout that predates the results/cache split (verdicts/ and predictions/ next to records/, .json
lists, a verdict key without doc_chars) is not read. Those runs cannot be reproduced under the current
package, and the files that remain have a different identity from the one their records describe.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .judge import Verdict
from .results import Result, load_results
from .retrieval import hits
from .run import cache_files

logger = logging.getLogger(__name__)

OBJ_EPS = 1e-9  # nDCG margin at or below which the official winner is a tie
CLEAR_MARGIN = 0.10  # "clear winner" band
TIER_A_MIN, TIER_B_MIN = 0.40, 0.20  # on the S scale, applied to the CI lower bound

ESTIMATOR_NOTE = (
    "s_committed = 2 * committed_agreement - 1: chance-corrected agreement under a uniform-marginal null "
    "(Bennett/Alpert/Goldstein S; Byrt PABAK for two categories). Not Cohen's kappa; see cohen_kappa_committed."
)


# ----------------------------------------------------------------------------- per-comparison winners
def ndcg_at_k(ranked_ids: Sequence[str], rels: Mapping[str, float], k: int) -> float | None:
    """Graded nDCG@k, linear gains, log2(rank + 1) discount; None when the query has no positive label."""
    ideal = sorted((float(g) for g in rels.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return None
    dcg = sum(max(0.0, float(rels.get(d, 0))) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    return dcg / sum(g / math.log2(i + 2) for i, g in enumerate(ideal))


def obj_winner(ndcg_a: float, ndcg_b: float, eps: float = OBJ_EPS) -> str:
    return "A" if ndcg_a > ndcg_b + eps else "B" if ndcg_b > ndcg_a + eps else "tie"


def judge_winner(score_a: float) -> str:
    """From the position-averaged score: above 0.5 is a commitment to A, below to B, 0.5 a tie or a split."""
    return "A" if score_a > 0.5 else "B" if score_a < 0.5 else "tie"


# ----------------------------------------------------------------------------- statistics on 2x2 cells
def _ratio(correct: int, wrong: int) -> float | None:
    return correct / (correct + wrong) if correct + wrong else None


def s_of(cells: Sequence[int]) -> float | None:
    """S = 2p - 1 from [AA, AB, BA, BB] (rows: official winner, columns: judge winner)."""
    aa, ab, ba, bb = cells
    p = _ratio(aa + bb, ab + ba)
    return None if p is None else 2.0 * p - 1.0


def cohen_kappa_of(cells: Sequence[int]) -> float | None:
    """Cohen's kappa from [AA, AB, BA, BB]; None when a marginal is degenerate."""
    aa, ab, ba, bb = cells
    n = aa + ab + ba + bb
    if n == 0:
        return None
    p_o = (aa + bb) / n
    p_e = ((aa + ab) / n) * ((aa + ba) / n) + ((ba + bb) / n) * ((ab + bb) / n)
    return None if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)


def tier_of(value: float | None) -> str | None:
    if value is None:
        return None
    return "A" if value >= TIER_A_MIN else "B" if value >= TIER_B_MIN else "C"


def assign_tier(ci_low: float | None, ci_high: float | None) -> dict[str, Any]:
    """Tier from the interval's lower bound, with a label that says when the interval spans a boundary."""
    if ci_low is None or ci_high is None:
        return {"tier": None, "tier_label": None, "tier_basis": "ci_lower_bound"}
    lo, hi = tier_of(ci_low), tier_of(ci_high)
    return {"tier": lo, "tier_label": f"{lo} (CI spans {hi})" if lo != hi else lo, "tier_basis": "ci_lower_bound"}


# ----------------------------------------------------------------------------- the run's artifacts
def load_verdicts(path: Path) -> dict[str, Verdict]:
    """{qid: verdict} from a pair's JSONL, the last row per query. A line that does not read back as
    a Verdict (blank, cut short by a crash, or from another schema) is skipped, as run.judge_pair_cached
    skips it when it resumes the pair."""
    if not path.exists():
        raise FileNotFoundError(f"no verdicts at {path}")
    rows: dict[str, Verdict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            v = Verdict(**json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
        rows[v.qid] = v
    return rows


def per_query_ndcg(
    path: Path, qrels: Mapping[str, Mapping[str, float]], k: int, ignore_identical_ids: bool = False
) -> dict[str, float]:
    """{qid: nDCG@k} from mteb's prediction file, over the list the judge was shown (a query that is its
    own document is dropped where mteb's flag says so); queries without a positive label are left out."""
    if not path.exists():
        raise FileNotFoundError(f"no predictions at {path}")
    out = {}
    for qid, scores in hits(path, ignore_identical_ids).items():
        if qid not in qrels:
            continue
        n = ndcg_at_k(sorted(scores, key=scores.get, reverse=True), qrels[qid], k)
        if n is not None:
            out[qid] = n
    return out


# ----------------------------------------------------------------------------- tally and bootstrap
_CELLS = 6  # AA, AB, BA, BB, clear-correct, clear-wrong
_CELL_INDEX = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}
_COUNTS = ("comparisons", "unparsed", "decisive", "committed", "abstain", "missing_ndcg", "clear", "clear_committed")


def tally(
    pairs: Mapping[tuple[str, str], Mapping[str, Verdict]], ndcg: Mapping[str, Mapping[str, float]]
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Per-query 2x2 cells over decisive-and-committed comparisons, plus the counts that explain the rest.
    A comparison with an unparsed order is left out before anything else is read from it."""
    per_query: dict[str, list[int]] = {}
    n = dict.fromkeys(_COUNTS, 0)
    for (a, b), rows in pairs.items():
        for qid, v in rows.items():
            n["comparisons"] += 1
            if not all(v.parsed_ok):  # the rule rank.rate applies; an identical-retrieval row ([]) is a tie
                n["unparsed"] += 1
                continue
            nd_a, nd_b = ndcg.get(a, {}).get(qid), ndcg.get(b, {}).get(qid)
            if nd_a is None or nd_b is None:
                n["missing_ndcg"] += 1
                continue
            obj, jw = obj_winner(nd_a, nd_b), judge_winner(float(v.score_a))
            if obj == "tie":
                continue
            n["decisive"] += 1
            clear = abs(nd_a - nd_b) >= CLEAR_MARGIN
            n["clear"] += clear
            if jw == "tie":
                n["abstain"] += 1
                continue
            n["committed"] += 1
            n["clear_committed"] += clear
            cells = per_query.setdefault(qid, [0] * _CELLS)
            cells[_CELL_INDEX[obj + jw]] += 1
            if clear:
                cells[4 if obj == jw else 5] += 1
    return per_query, n


def _stats(cells: Sequence[int]) -> tuple[float | None, float | None, float | None]:
    return s_of(cells[:4]), cohen_kappa_of(cells[:4]), _ratio(cells[4], cells[5])


def _percentile(sorted_vals: Sequence[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_vals[lo] * (hi - idx) + sorted_vals[hi] * (idx - lo) if lo != hi else sorted_vals[int(idx)]


def bootstrap_ci(per_query: Mapping[str, Sequence[int]], n: int, seed: int) -> dict[str, list[float | None]]:
    """Query-clustered percentile intervals for S, Cohen's kappa and clear-winner agreement."""
    cells = [per_query[q] for q in sorted(per_query)]
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"s": [], "cohen": [], "clear": []}
    for _ in range(n if cells else 0):
        acc = [0] * _CELLS
        for _ in cells:
            for j, v in enumerate(cells[rng.randrange(len(cells))]):
                acc[j] += v
        for name, value in zip(draws, _stats(acc)):
            if value is not None:
                draws[name].append(value)
    return {k: [_percentile(sorted(v), 2.5), _percentile(sorted(v), 97.5)] for k, v in draws.items()}


def readout(per_query: Mapping[str, Sequence[int]], n: Mapping[str, int], *, bootstrap: int, seed: int) -> dict:
    total = [sum(c[j] for c in per_query.values()) for j in range(_CELLS)]
    s, cohen, clear = _stats(total)
    ci = bootstrap_ci(per_query, bootstrap, seed)
    return {
        "n_queries_scored": len(per_query),
        "n_comparisons": n["comparisons"],
        "n_unparsed": n["unparsed"],  # left out: the judge's answer did not parse in one of the orders
        "n_decisive": n["decisive"],
        "n_committed": n["committed"],
        "n_abstain": n["abstain"],
        "n_missing_ndcg": n["missing_ndcg"],
        "committed_agreement": _ratio(total[0] + total[3], total[1] + total[2]),
        "s_committed": s,
        "s_committed_ci95": ci["s"],
        "kappa_committed": s,  # the paper's kappa = 2p - 1; an alias of s_committed
        "cohen_kappa_committed": cohen,
        "cohen_kappa_ci95": ci["cohen"],
        "clear_winner_margin": CLEAR_MARGIN,
        "n_clear_winner": n["clear"],
        "n_clear_winner_committed": n["clear_committed"],
        "clear_winner_agreement": clear,
        "clear_winner_ci95": ci["clear"],
        "cells": dict(zip(_CELL_INDEX, total)),
        **assign_tier(*ci["s"]),
        "tier_of_point_estimate": tier_of(s),
        "bootstrap": bootstrap,
        "seed": seed,
        "estimator_note": ESTIMATOR_NOTE,
    }


# ----------------------------------------------------------------------------- entry points
def _result(result: Result | dict | str | Path) -> Result:
    if isinstance(result, Result):
        return result
    return Result(result) if isinstance(result, dict) else Result.from_disk(result)


def judge_reliability(
    result: Result | dict | str | Path,
    cache_folder: str | Path | None = None,
    *,
    qrels: Mapping[str, Mapping[str, float]] | None = None,
    ignore_identical_ids: bool | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    write: bool = True,
) -> dict:
    """Score one record's verdicts against qrels; the readout is stored under record["reliability"].

    `result` is a Result, a record, or a record's path. Its verdict and prediction files are the ones
    run.cache_files names under `cache_folder` ($MTEB_GYM_CACHE, else ~/.cache/mteb_gym, when None).
    Without explicit `qrels` the record must come from the original-query arm, and the labels and
    mteb's ignore_identical_ids flag are loaded from the corpus; with explicit qrels, pass the flag for
    a task where a query is its own document (ArguAna), since the judge never saw that document.
    A record that cannot be scored gets {"error": ...} and is left untouched, so "not measured" never
    looks like "agreed 0% of the time".
    """
    res = _result(result)
    rec = res.record
    c = rec["config"]
    if qrels is None:
        if c.get("arm") != "original":
            return {"error": f"arm is {c.get('arm')!r}: reliability needs the original-query arm or explicit qrels"}
        from .corpus import load

        corp = load(rec["task_name"])
        qrels = corp.qrels
        if not qrels:
            return {"error": f"{rec['task_name']} carries no qrels"}
        if ignore_identical_ids is None:
            ignore_identical_ids = corp.ignore_identical_ids
    try:
        files = cache_files(rec, cache_folder)
    except KeyError as e:
        return {"error": f"config has no {e.args[0]}: the record predates the cache layout this reads"}
    models, k = list(c["models"]), int(c["top_k"])
    try:
        ndcg = {
            m: per_query_ndcg(p, qrels, k, bool(ignore_identical_ids)) for m, p in zip(models, files["predictions"])
        }
        pairs = {ab: load_verdicts(p) for ab, p in zip(itertools.combinations(models, 2), files["verdicts"])}
    except FileNotFoundError as e:
        return {"error": str(e)}
    per_query, n = tally(pairs, ndcg)
    if n["unparsed"]:
        logger.warning(
            "%s: %d of %d comparisons had an unparsed order and are left out",
            rec["task_name"],
            n["unparsed"],
            n["comparisons"],
        )
    if not per_query:
        return {"error": "no decisive, committed comparison to score (empty verdicts or no labelled query)"}
    r = readout(per_query, n, bootstrap=bootstrap, seed=seed)
    r["n_models"] = len(models)
    r["ndcg_k"] = k
    rec["reliability"] = r
    if write and res.path:
        res.to_disk()
    return r


def reliability_all(root: str | Path, cache_folder: str | Path | None = None, **kwargs) -> dict[str, dict]:
    """judge_reliability over every original-arm record under `root` (a results folder's or results
    repository's <task>/ folders, as load_results reads them): {record path: readout}. Every record's
    artifacts are read from the one cache."""
    out = {}
    for res in load_results(root).results:
        if res.record["config"].get("arm") == "original":
            out[str(res.path)] = judge_reliability(res, cache_folder, **kwargs)
    return out
