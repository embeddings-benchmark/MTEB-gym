"""
Rating from pairwise verdicts. Fractional scores (judge.py) fold into a win
matrix W (W[i][j] = fractional wins of i over j), fit with Bradley-Terry via the
MM algorithm (Hunter 2004) and reported on an Elo-style scale, as Chatbot Arena
and the MTEB Arena do. Confidence intervals bootstrap over queries.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)

BASE, SCALE = 1000.0, 400.0
BOOTSTRAP = 1000


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

    @property
    def ci(self) -> float:
        return (self.ci_high - self.ci_low) / 2


def _win_matrix(verdicts, names) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        W[a, b] += v.score_a
        W[b, a] += 1.0 - v.score_a
    return W


def _bradley_terry(W: np.ndarray, iters: int = 200, tol: float = 1e-9, smoothing: float = 0.1) -> np.ndarray:
    """MM fixed point; strengths normalised to geometric mean 1. `smoothing` adds a
    symmetric pseudo-win per pair so a winless model converges to the bottom
    instead of staying at its initial strength."""
    n = W.shape[0]
    W = W + smoothing * (1.0 - np.eye(n))
    p = np.ones(n)
    wins = W.sum(axis=1)
    games = W + W.T
    for _ in range(iters):
        p_old = p.copy()
        for i in range(n):
            denom = sum(games[i, j] / (p[i] + p[j]) for j in range(n) if j != i and games[i, j] > 0)
            if denom > 0:
                p[i] = wins[i] / denom
        p = p / np.exp(np.mean(np.log(np.clip(p, 1e-12, None))))
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p


def _ratings(verdicts, names) -> np.ndarray:
    r = SCALE * np.log10(np.clip(_bradley_terry(_win_matrix(verdicts, names)), 1e-12, None))
    return r - r.mean() + BASE


def rate(verdicts, bootstrap: int = BOOTSTRAP, seed: int = 0) -> list[ModelRating]:
    # A comparison where either order failed to parse is missing data, not a tie: it is left out,
    # as MT-Bench and Arena-Hard do. Keeping the half that parsed would bring back position bias.
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
        by_qid: dict[str, list] = {}
        for v in verdicts:
            by_qid.setdefault(v.qid, []).append(v)
        qids = np.array(list(by_qid), dtype=object)
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(bootstrap):
            sample = [v for q in rng.choice(qids, size=len(qids), replace=True) for v in by_qid[q]]
            try:
                rows.append(_ratings(sample, names))
            except Exception:  # noqa: BLE001 - degenerate resample, excluded from the CI
                continue
        if len(rows) < bootstrap:
            logger.warning("bootstrap: %d/%d resamples were degenerate and excluded", bootstrap - len(rows), bootstrap)
        if rows:
            boot = np.vstack(rows)
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


def format_leaderboard(ratings: list) -> str:
    """From ModelRating objects or the record's rating rows."""
    rows = [asdict(m) | {"model": m.name} if isinstance(m, ModelRating) else m for m in ratings]
    lines = ["=" * 78, f"{'Rank':<5}{'Model':<38}{'Rating':>8}{'CI±':>7}{'W':>5}{'L':>5}{'T':>5}", "-" * 78]
    for i, r in enumerate(rows, 1):
        ci = (r["ci_high"] - r["ci_low"]) / 2
        lines.append(
            f"{i:<5}{r['model'].split('/')[-1][:36]:<38}{r['rating']:>8.0f}{ci:>7.0f}"
            f"{r['wins']:>5.0f}{r['losses']:>5.0f}{r['ties']:>5.0f}"
        )
    return "\n".join(lines + ["=" * 78])
