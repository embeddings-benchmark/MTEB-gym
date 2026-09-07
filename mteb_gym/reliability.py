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

Everything comes from the run's own artifacts: the record's config identifies the verdict files and
mteb prediction files under the output folder, and the qrels come from the corpus (or are passed in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .results import Result, load_results
from .retrieval import slug

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
    """From the position-averaged score: 1.0 / 0.0 are commitments, 0.5 is a tie or a split."""
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
def _sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def verdict_file(out: Path, record: dict, a: str, b: str) -> Path:
    """The pair's verdict file, from the same identity run() used to write it (see run.verdict_key)."""
    c = record["config"]
    rev = c.get("model_revisions") or {}
    key = _sha(
        c["judge_model"], c["judge_system"], c["top_k"], c["query_set"], f"{a}@{rev.get(a)}", f"{b}@{rev.get(b)}"
    )
    return out / "verdicts" / f"{slug(a)}__{slug(b)}-{key}.json"


def prediction_file(out: Path, record: dict, model: str) -> Path:
    c = record["config"]
    rev = (c.get("model_revisions") or {}).get(model)
    return out / "predictions" / f"{slug(model)}@{rev}" / c["query_set"] / f"{record['task_name']}_predictions.json"


def load_verdicts(path: Path) -> dict[str, float]:
    """{qid: score_a} from a pair's finished .json list, or its .jsonl stream if the run stopped early."""
    if path.exists():
        rows = json.loads(path.read_text())
    elif path.with_suffix(".jsonl").exists():
        rows = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines() if line.strip()]
    else:
        raise FileNotFoundError(f"no verdicts at {path} (or .jsonl)")
    return {v["qid"]: float(v["score_a"]) for v in rows}


def per_query_ndcg(path: Path, qrels: Mapping[str, Mapping[str, float]], k: int) -> dict[str, float]:
    """{qid: nDCG@k} from mteb's prediction file; queries without a positive label are left out."""
    hits = json.loads(path.read_text())["default"]["test"]
    out = {}
    for qid, scores in hits.items():
        if qid not in qrels:
            continue
        n = ndcg_at_k(sorted(scores, key=scores.get, reverse=True), qrels[qid], k)
        if n is not None:
            out[qid] = n
    return out


# ----------------------------------------------------------------------------- tally and bootstrap
_CELLS = 6  # AA, AB, BA, BB, clear-correct, clear-wrong
_CELL_INDEX = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}
_COUNTS = ("comparisons", "decisive", "committed", "abstain", "missing_ndcg", "clear", "clear_committed")


def tally(
    pairs: Mapping[tuple[str, str], Mapping[str, float]], ndcg: Mapping[str, Mapping[str, float]]
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Per-query 2x2 cells over decisive-and-committed comparisons, plus the counts that explain the rest."""
    per_query: dict[str, list[int]] = {}
    n = dict.fromkeys(_COUNTS, 0)
    for (a, b), scores in pairs.items():
        for qid, score_a in scores.items():
            n["comparisons"] += 1
            nd_a, nd_b = ndcg.get(a, {}).get(qid), ndcg.get(b, {}).get(qid)
            if nd_a is None or nd_b is None:
                n["missing_ndcg"] += 1
                continue
            obj, jw = obj_winner(nd_a, nd_b), judge_winner(score_a)
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
def judge_reliability(
    result: Result | str | Path,
    output_folder: str | Path,
    *,
    qrels: Mapping[str, Mapping[str, float]] | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    write: bool = True,
) -> dict:
    """Score one record's verdicts against qrels; the readout is stored under record["reliability"].

    Without explicit `qrels` the record must come from the original-query arm, and the labels are
    loaded from the corpus. A record that cannot be scored gets {"error": ...} and is left untouched,
    so "not measured" never looks like "agreed 0% of the time".
    """
    res = result if isinstance(result, Result) else Result.from_disk(result)
    rec, out = res.record, Path(output_folder)
    c = rec["config"]
    if qrels is None:
        if c.get("arm") != "original":
            return {"error": f"arm is {c.get('arm')!r}: reliability needs the original-query arm or explicit qrels"}
        from .corpus import load

        qrels = load(rec["task_name"]).qrels
        if not qrels:
            return {"error": f"{rec['task_name']} carries no qrels"}
    models = [r["model"] for r in rec["ratings"]]
    try:
        ndcg = {m: per_query_ndcg(prediction_file(out, rec, m), qrels, int(c["top_k"])) for m in models}
        pairs = {}
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                path = verdict_file(out, rec, a, b)
                if path.exists() or path.with_suffix(".jsonl").exists():
                    pairs[(a, b)] = load_verdicts(path)
                else:  # the run ordered the pair the other way round
                    pairs[(b, a)] = load_verdicts(verdict_file(out, rec, b, a))
    except FileNotFoundError as e:
        return {"error": str(e)}
    per_query, n = tally(pairs, ndcg)
    if not per_query:
        return {"error": "no decisive, committed comparison to score (empty verdicts or no labelled query)"}
    r = readout(per_query, n, bootstrap=bootstrap, seed=seed)
    r["n_models"] = len(models)
    r["ndcg_k"] = int(c["top_k"])
    rec["reliability"] = r
    if write and res.path:
        res.to_disk()
    return r


def reliability_all(root: str | Path, **kwargs) -> dict[str, dict]:
    """judge_reliability over every original-arm record under `root`: {record path: readout}. Each record's
    artifacts are read from its own output folder, the parent of the records/ directory it sits in."""
    out = {}
    for res in load_results(root).results:
        if res.record["config"].get("arm") == "original":
            out[str(res.path)] = judge_reliability(res, res.path.parent.parent, **kwargs)
    return out
