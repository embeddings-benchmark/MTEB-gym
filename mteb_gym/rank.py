"""
Ratings. Pointwise (rate_pointwise): nDCG@k over graded pools, x100, with the ideal ranking taken from the pool's best
grades. Pairwise (rate): fractional scores fold into a win matrix W (W[i][j] = fractional wins of i over j), fit with
Bradley-Terry via the MM algorithm (Hunter 2004) and reported on an Elo-style scale. Confidence intervals bootstrap
over queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .judge import Verdict
    from .retrieval import Ranked


logger = logging.getLogger(__name__)

BASE, SCALE = 1000.0, 400.0
BOOTSTRAP = 1000
TIE = 1e-9  # nDCG differences below this are rounding


@dataclass
class ModelRating:
    name: str
    rating: float
    ci_low: float
    ci_high: float
    wins: float
    losses: float
    ties: float
    n: int


def _win_matrix(verdicts: list[Verdict], names: list[str]) -> npt.NDArray[np.float64]:
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        W[a, b] += v.score_a
        W[b, a] += 1.0 - v.score_a
    return W


def _bradley_terry(
    W: npt.NDArray[np.float64], iters: int = 20000, tol: float = 1e-9, smoothing: float = 0.1, warn: bool = True
) -> npt.NDArray[np.float64]:
    """MM fixed point; strengths normalised to geometric mean 1. `smoothing` adds a symmetric pseudo-win per pair, so a
    winless model converges to the bottom instead of staying at its initial strength."""
    n = W.shape[0]
    W = W + smoothing * (1.0 - np.eye(n))
    wins = W.sum(axis=1)
    games = W + W.T
    p: npt.NDArray[np.float64] = np.ones(n)
    update: npt.NDArray[np.float64]
    for _ in range(iters):
        update = wins / (games / (p[:, None] + p[None, :])).sum(axis=1)
        update = update / np.exp(np.mean(np.log(update)))
        converged = np.max(np.abs(np.log(update) - np.log(p))) < tol
        p = update
        if converged:
            return p
    if warn:
        logger.warning("Bradley-Terry did not converge in %d iterations", iters)
    return p


def _ratings(verdicts: list[Verdict], names: list[str], warn: bool = True) -> npt.NDArray[np.float64]:
    r = SCALE * np.log10(np.clip(_bradley_terry(_win_matrix(verdicts, names), warn=warn), 1e-12, None))
    ratings: npt.NDArray[np.float64] = r - r.mean() + BASE
    return ratings


def rate(verdicts: list[Verdict], bootstrap: int = BOOTSTRAP, seed: int = 0) -> list[ModelRating]:
    # A comparison with a failed order is left out whole: keeping one order would bring back position bias.
    failed = sum(not all(v.parsed_ok) for v in verdicts)
    if failed:
        logger.warning("%d of %d comparisons had an unparseable judge answer and are left out", failed, len(verdicts))
    verdicts = [v for v in verdicts if all(v.parsed_ok)]
    if not verdicts:
        return []
    names = sorted({m for v in verdicts for m in (v.model_a, v.model_b)})
    point = _ratings(verdicts, names)

    # Resample queries, not verdicts: every query yields one verdict per pair, so
    # verdicts sharing a qid are correlated and per-verdict resampling understates the CI.
    lo = hi = point
    if bootstrap and len(verdicts) > 1:
        by_qid: dict[str, list[Verdict]] = {}
        for v in verdicts:
            by_qid.setdefault(v.qid, []).append(v)
        qids = np.array(list(by_qid), dtype=object)
        rng = np.random.default_rng(seed)
        boot = np.vstack(
            [
                _ratings(
                    [v for q in rng.choice(qids, size=len(qids), replace=True) for v in by_qid[q]], names, warn=False
                )
                for _ in range(bootstrap)
            ]
        )
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    idx = {n: i for i, n in enumerate(names)}
    wins, losses, ties, games = (np.zeros(len(names)) for _ in range(4))
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        games[a] += 1
        games[b] += 1
        if v.score_a > 0.5:
            wins[a] += 1
            losses[b] += 1
        elif v.score_a < 0.5:
            losses[a] += 1
            wins[b] += 1
        else:
            ties[a] += 1
            ties[b] += 1
    out = [
        ModelRating(
            names[i],
            float(point[i]),
            float(lo[i]),
            float(hi[i]),
            float(wins[i]),
            float(losses[i]),
            float(ties[i]),
            int(games[i]),
        )
        for i in range(len(names))
    ]
    return sorted(out, key=lambda m: m.rating, reverse=True)


def ndcg_per_query(doc_ids: list[str], grades: dict[str, float], pool_grades: list[float], k: int) -> float:
    """nDCG@k with linear gain (as trec_eval) and the ideal ranking from the k best grades in the query's pool."""
    ideal = sorted(pool_grades, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    dcg = sum(grades.get(d, 0.0) / np.log2(i + 2) for i, d in enumerate(doc_ids[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def rate_pointwise(
    grades: dict[tuple[str, str], float],
    ranked: dict[str, list[Ranked]],
    k: int = 10,
    bootstrap: int = BOOTSTRAP,
    seed: int = 0,
) -> list[ModelRating]:
    """`grades` maps (qid, doc_id) to a 0-5 grade, `ranked` each model to its Ranked lists. A model's rating is its
    mean nDCG@k x 100; the CI bootstraps over queries. wins/losses/ties count queries on which the model's nDCG is
    above, below or equal to the mean of the other models."""
    names = sorted(ranked)
    qids = sorted({r.qid for lists in ranked.values() for r in lists})
    by_q: dict[str, dict[str, float]] = {q: {} for q in qids}
    for (q, d), g in grades.items():
        if q in by_q:
            by_q[q][d] = g
    table = np.zeros((len(qids), len(names)))  # nDCG per query and model
    for j, m in enumerate(names):
        lists = {r.qid: r for r in ranked[m]}
        for i, q in enumerate(qids):
            r = lists.get(q)
            table[i, j] = ndcg_per_query(r.doc_ids, by_q[q], list(by_q[q].values()), k) if r else 0.0
    point = 100 * table.mean(axis=0)
    lo = hi = point
    if bootstrap and len(qids) > 1:
        rng = np.random.default_rng(seed)
        boot = np.vstack([100 * table[rng.integers(0, len(qids), len(qids))].mean(axis=0) for _ in range(bootstrap)])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    others = (table.sum(axis=1, keepdims=True) - table) / max(1, len(names) - 1)
    out = [
        ModelRating(
            names[j],
            float(point[j]),
            float(lo[j]),
            float(hi[j]),
            float((table[:, j] > others[:, j] + TIE).sum()),
            float((table[:, j] < others[:, j] - TIE).sum()),
            float((np.abs(table[:, j] - others[:, j]) <= TIE).sum()),
            len(qids),
        )
        for j in range(len(names))
    ]
    return sorted(out, key=lambda m: m.rating, reverse=True)


def format_leaderboard(ratings: list[dict[str, Any]]) -> str:
    """From the record's rating rows."""
    lines = ["=" * 78, f"{'Rank':<5}{'Model':<38}{'Rating':>8}{'CI±':>7}{'W':>5}{'L':>5}{'T':>5}", "-" * 78]
    for i, r in enumerate(ratings, 1):
        ci = (r["ci_high"] - r["ci_low"]) / 2
        lines.append(
            f"{i:<5}{r['model'].split('/')[-1][:36]:<38}{r['rating']:>8.0f}{ci:>7.0f}"
            f"{r['wins']:>5.0f}{r['losses']:>5.0f}{r['ties']:>5.0f}"
        )
    return "\n".join(lines + ["=" * 78])
