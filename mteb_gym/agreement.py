"""
Agreement between a gym ranking and the official MTEB ranking on the same task
(used by Result.agreement). Official scores come from the MTEB results repository
through mteb's result cache; models without one are skipped or, with
evaluate_missing=True, evaluated by mteb on the real task and stored in that cache.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Two models with equal total wins fit to strengths that differ only by the MM stopping
# tolerance in rank.py, a rating gap of about 1e-8 once the fit has converged (a fit that
# stops at the iteration cap can leave a larger gap). Gaps below this count as ties.
RATING_TIE_TOL = 1e-6


def official_scores(cache, task: str, models: list[str]) -> dict[str, tuple[float, str]]:
    """Each model's official score on `task` and the revision it came from. A model can have
    results under several revisions; `join_revisions` picks one by mteb's own rule."""
    frame = (
        cache.load_results(models=models, tasks=[task])
        .join_revisions()
        .to_dataframe(format="long", include_model_revision=True)
    )
    return {row.model_name: (float(row.score), row.model_revision) for row in frame.itertuples()}


def fetch_truth(
    models: list[str], task: str, *, evaluate_missing: bool = False
) -> tuple[dict[str, float], dict[str, dict]]:
    """Official main score (nDCG@10, x100) per model on `task`, and per model where it came from:
    {"model_revision": the revision the score is filed under, "official": whether it is MTEB's
    published score or one we ran}. Field names follow mteb's ModelResult."""
    import mteb

    cache = mteb.ResultCache()
    try:
        cache.download_from_remote()
    except Exception:  # noqa: BLE001 - offline: use the local copy
        logger.warning("could not refresh the MTEB results cache; using the local copy")
    official = official_scores(cache, task, models)
    scores: dict[str, float] = {}
    source: dict[str, dict] = {}
    for name in models:
        meta = mteb.get_model_meta(name)
        if name in official:
            score, revision = official[name]
            source[name] = {"model_revision": revision, "official": True}
            if revision != meta.revision:
                logger.info(
                    "%s on %s: score comes from revision %s, not the pinned %s", name, task, revision, meta.revision
                )
        elif evaluate_missing:
            logger.info("no official result for %s on %s; evaluating with mteb", name, task)
            res = mteb.evaluate(
                meta, mteb.get_task(task), cache=cache, overwrite_strategy="only-missing", show_progress_bar=False
            )
            if not res.task_results:
                continue
            score = float(res.task_results[0].get_score())
            source[name] = {"model_revision": meta.revision, "official": False}
        else:
            logger.warning("no official result for %s on %s; skipped (evaluate_missing=True to run it)", name, task)
            continue
        scores[name] = round(score * 100, 2)
    return scores, source


def _snap_ties(ratings: np.ndarray, tol: float = RATING_TIE_TOL) -> np.ndarray:
    """Ratings closer than `tol` to the next one in sorted order are set equal, so tied
    strengths get tied ranks instead of an order fixed by the fit's rounding noise."""
    order = np.argsort(ratings)
    snapped = ratings[order].copy()
    for i in range(1, len(snapped)):
        if snapped[i] - snapped[i - 1] < tol:
            snapped[i] = snapped[i - 1]
    out = np.empty_like(ratings)
    out[order] = snapped
    return out


def _tau_ap(g: np.ndarray, t: np.ndarray) -> float:
    """AP rank correlation (Yilmaz et al. 2008): Kendall's tau weighted toward the
    top of the reference ranking t. 1 = identical order, -1 = reversed. A pair tied in g
    counts half, as in tau-b."""
    g_ord = g[np.argsort(-t)]
    n = len(g_ord)
    total = sum(float(np.sum(g_ord[:i] > g_ord[i]) + 0.5 * np.sum(g_ord[:i] == g_ord[i])) / i for i in range(1, n))
    return float(2.0 * total / (n - 1) - 1.0)


def correlate(
    gym_ratings: dict[str, float], ground_truth: dict[str, float], bootstrap: int = 1000, seed: int = 0
) -> dict:
    """Rank agreement over the models present in both dicts."""
    from scipy.stats import kendalltau, spearmanr

    shared = [m for m in gym_ratings if m in ground_truth]
    if len(shared) < 3:
        return {"error": f"need >=3 shared models, have {len(shared)}", "shared": shared}
    g = _snap_ties(np.array([gym_ratings[m] for m in shared]))
    snapped = dict(zip(shared, g.tolist()))
    t = np.array([ground_truth[m] for m in shared])
    rho, p_rho = spearmanr(g, t)
    tau, p_tau = kendalltau(g, t)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(shared), len(shared))
        if len(set(idx)) < 3:
            continue
        r, _ = spearmanr(g[idx], t[idx])
        if not np.isnan(r):
            boots.append(r)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None]

    top10 = None
    if len(shared) >= 12:  # among the officially top-10 models only: where a selection decision is made
        top = np.argsort(-t)[:10]
        r10, _ = spearmanr(g[top], t[top])
        top10 = None if np.isnan(r10) else float(r10)

    return {
        "n_models": len(shared),
        "models": shared,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        "spearman_top10": top10,
        "kendall_ap": _tau_ap(g, t),
        "spearman_ci95": ci,
        "gym_ranking": sorted(shared, key=lambda m: -snapped[m]),
        "truth_ranking": sorted(shared, key=lambda m: -ground_truth[m]),
    }
