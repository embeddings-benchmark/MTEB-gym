"""
Ranking agreement as the run scales (issue #52).

A gym ranking is fit from n_queries synthetic queries judged on every pair of n_models models. This
script asks how much of that is needed for the label-free ranking to agree with the official one: it
draws random subsamples of the record's own verdicts, refits Bradley-Terry on each, and correlates the
refit ranking with the truth. The three axes are varied one at a time with the other two held at full,
so the output is three one-dimensional curves plus the full-data reference:

    queries   5, 10, 20, 40, 80 queries (capped at the record's count), every pair, every model
    pairs     25%, 50%, 100% of the judged pairs, every query, every model
    models    4 .. n models, every query, every pair among them

Per grid point, R seeded draws give Spearman's rho between the refit rating and the truth over the
sampled models: the mean, the 2.5 and 97.5 percentiles, and the fraction of valid draws within 0.05 of
the full-data rho. A draw that leaves a sampled model with no judged pair cannot rate it, and a draw
whose refit ratings all tie has no rank order; both are dropped, so n_valid < draws says the budget was
too small to separate the models, not that the ranking held. Two models with equal win totals get
equal Bradley-Terry strengths and are ordered by the fit's iteration residue, as on the leaderboard,
so rho on a near-tied record is only defined up to that order (the mock fixture is such a case).

Truth is ordinal: the record's agreement.truth_ranking (the official order, no scores, so mteb is never
needed), or --truth, a JSON {model: score} that is used instead when given. Verdict files are located
through mteb_gym.reliability.verdict_file, the same identity run() used to write them.

    python -m analysis.scaling --output-folder results --record results/records/<record>.json --out scaling.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mteb_gym.judge import Verdict
from mteb_gym.rank import rate
from mteb_gym.reliability import load_verdicts, verdict_file

logger = logging.getLogger(__name__)

QUERY_GRID = (5, 10, 20, 40, 80)
PAIR_FRACTIONS = (0.25, 0.5, 1.0)
MIN_MODELS = 4
DRAWS = 200
TOLERANCE = 0.05  # a draw "agrees" when its rho is this close to the full-data rho

Pairs = dict[tuple[str, str], list[Verdict]]
AXES = (("queries", "n_queries"), ("pairs", "pair_fraction"), ("models", "n_models"))


# ----------------------------------------------------------------------------- inputs
def load_truth(record: dict, truth: Mapping[str, float] | None = None) -> dict[str, float]:
    """{model: score} over the record's models. Without `truth`, the official order in
    agreement.truth_ranking is turned into ordinal scores (first = highest)."""
    models = list(record["config"]["models"])
    if truth is None:
        ranking = (record.get("agreement") or {}).get("truth_ranking")
        if not ranking:
            raise ValueError("record has no agreement.truth_ranking; pass --truth with {model: score}")
        truth = {m: float(len(ranking) - i) for i, m in enumerate(ranking)}
    missing = [m for m in models if m not in truth]
    if missing:
        raise ValueError(f"no truth score for {missing}")
    return {m: float(truth[m]) for m in models}


def load_pairs(out: Path, record: dict) -> Pairs:
    """Every judged pair's verdicts, located through reliability.verdict_file in either model order."""
    pairs: Pairs = {}
    for a, b in itertools.combinations(record["config"]["models"], 2):
        path = verdict_file(out, record, a, b)
        if not (path.exists() or path.with_suffix(".jsonl").exists()):  # the run ordered the pair the other way
            a, b, path = b, a, verdict_file(out, record, b, a)
        pairs[(a, b)] = [Verdict(qid, qid, a, b, s) for qid, s in load_verdicts(path).items()]
    return pairs


def qids_of(pairs: Pairs) -> list[str]:
    return sorted({v.qid for vs in pairs.values() for v in vs})


def axis_values(
    n_queries: int, n_models: int, query_grid: Sequence[int], pair_fractions: Sequence[float], min_models: int
) -> dict[str, list]:
    """Each axis's grid, capped at what the record has and always ending at the full value."""
    return {
        "queries": sorted({q for q in query_grid if q < n_queries} | {n_queries}),
        "pairs": sorted({f for f in pair_fractions if f < 1.0} | {1.0}),
        "models": sorted(set(range(min_models, n_models)) | {n_models}),
    }


# ----------------------------------------------------------------------------- one draw
def subsample(
    pairs: Pairs,
    models: Sequence[str],
    qids: Sequence[str],
    n_queries: int,
    pair_fraction: float,
    n_models: int,
    rng: np.random.Generator,
) -> tuple[list[Verdict], list[str]]:
    """Verdicts for n_models random models, ceil(pair_fraction * their pairs) random pairs, n_queries random
    queries (the same queries for every pair, as in a run). Returns the verdicts and the sampled models."""
    chosen = [str(m) for m in rng.choice(list(models), size=n_models, replace=False)]
    keep = {str(q) for q in rng.choice(list(qids), size=n_queries, replace=False)}
    candidates = [p for p in pairs if p[0] in chosen and p[1] in chosen]
    n_pairs = max(1, math.ceil(pair_fraction * len(candidates)))
    picked = rng.choice(len(candidates), size=n_pairs, replace=False)
    return [v for i in picked for v in pairs[candidates[i]] if v.qid in keep], chosen


def refit_rho(verdicts: Sequence[Verdict], models: Sequence[str], truth: Mapping[str, float]) -> float | None:
    """Spearman's rho between the refit rating and the truth over `models`; None when a model went
    unrated (no sampled pair mentions it) or the refit ratings are all tied."""
    rating = {m.name: m.rating for m in rate(list(verdicts), bootstrap=0)}
    if len(models) < 3 or any(m not in rating for m in models):
        return None
    xs, ys = [rating[m] for m in models], [truth[m] for m in models]
    if len(set(xs)) == 1 or len(set(ys)) == 1:  # no rank order: spearmanr would warn and return nan
        return None
    rho, _ = spearmanr(xs, ys)
    return None if np.isnan(rho) else float(rho)


# ----------------------------------------------------------------------------- the grid
def summarize(rhos: Sequence[float | None], full_rho: float) -> dict[str, Any]:
    valid = np.array([r for r in rhos if r is not None])
    if not len(valid):
        return {"mean": None, "p2.5": None, "p97.5": None, "within": None, "n_valid": 0, "n_draws": len(rhos)}
    return {
        "mean": float(valid.mean()),
        "p2.5": float(np.percentile(valid, 2.5)),
        "p97.5": float(np.percentile(valid, 97.5)),
        "within": float(np.mean(np.abs(valid - full_rho) <= TOLERANCE)),
        "n_valid": int(len(valid)),
        "n_draws": len(rhos),
    }


def grid_point(
    pairs: Pairs,
    models: Sequence[str],
    qids: Sequence[str],
    truth: Mapping[str, float],
    full_rho: float,
    *,
    n_queries: int,
    pair_fraction: float,
    n_models: int,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rhos = []
    for _ in range(draws):
        sub, chosen = subsample(pairs, models, qids, n_queries, pair_fraction, n_models, rng)
        rhos.append(refit_rho(sub, chosen, truth))
    n_pairs = max(1, math.ceil(pair_fraction * (n_models * (n_models - 1) // 2)))
    point = {"n_queries": n_queries, "pair_fraction": pair_fraction, "n_pairs": n_pairs, "n_models": n_models}
    return point | summarize(rhos, full_rho)


def scaling(
    output_folder: str | Path,
    record: dict | str | Path,
    *,
    truth: Mapping[str, float] | None = None,
    draws: int = DRAWS,
    seed: int = 0,
    query_grid: Sequence[int] = QUERY_GRID,
    pair_fractions: Sequence[float] = PAIR_FRACTIONS,
    min_models: int = MIN_MODELS,
) -> dict[str, Any]:
    """The three curves and the full-data reference for one record; see the module docstring."""
    out = Path(output_folder)
    rec = record if isinstance(record, dict) else json.loads(Path(record).read_text())
    models = list(rec["config"]["models"])
    if len(models) < 3:
        raise ValueError(f"need at least 3 models for a rank correlation, record has {len(models)}")
    if min_models < 3:
        raise ValueError(f"min_models must be at least 3 for a rank correlation, got {min_models}")
    if not query_grid or any(q < 1 for q in query_grid):
        raise ValueError(f"query counts must be positive, got {list(query_grid)}")
    if not pair_fractions or any(not 0.0 < f <= 1.0 for f in pair_fractions):
        raise ValueError(f"pair fractions must lie in (0, 1], got {list(pair_fractions)}")
    scores = load_truth(rec, truth)
    pairs = load_pairs(out, rec)
    qids = qids_of(pairs)
    every = [v for vs in pairs.values() for v in vs]
    full_rho = refit_rho(every, models, scores)
    if full_rho is None:
        raise ValueError("the full data leaves a model unrated or every rating tied: nothing to compare with")
    full = {"n_queries": len(qids), "pair_fraction": 1.0, "n_models": len(models)}
    rng = np.random.default_rng(seed)
    axes = axis_values(len(qids), len(models), query_grid, pair_fractions, min_models)
    grid = {
        axis: [
            grid_point(pairs, models, qids, scores, full_rho, draws=draws, rng=rng, **{**full, key: value})
            for value in axes[axis]
        ]
        for axis, key in AXES
    }
    refit = {m.name: m.rating for m in rate(every, bootstrap=0)}
    result = {
        "task_name": rec.get("task_name"),
        "config_hash": rec["config"].get("config_hash"),
        "truth_source": "truth file" if truth is not None else "agreement.truth_ranking (ordinal)",
        "n_models": len(models),
        "n_pairs": len(pairs),
        "n_queries": len(qids),
        "draws": draws,
        "seed": seed,
        "tolerance": TOLERANCE,
        "full": {
            "rho": full_rho,
            "gym_ranking": sorted(models, key=lambda m: -refit[m]),
            "truth_ranking": sorted(models, key=lambda m: -scores[m]),
        },
        "grid": grid,
    }
    result["table"] = format_table(result)
    return result


# ----------------------------------------------------------------------------- report
def _cell(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def format_table(result: Mapping[str, Any]) -> str:
    """Markdown: one row per grid point, the full-data reference in the heading."""
    head = (
        f"full data: {result['n_models']} models, {result['n_pairs']} pairs, {result['n_queries']} queries, "
        f"rho = {result['full']['rho']:.3f} vs {result['truth_source']}; {result['draws']} draws per point"
    )
    lines = [
        head,
        "",
        "| axis | queries | pairs | models | mean rho | 2.5% | 97.5% | within 0.05 | valid draws |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for axis, points in result["grid"].items():
        for p in points:
            lines.append(
                f"| {axis} | {p['n_queries']} | {p['n_pairs']} | {p['n_models']} | {_cell(p['mean'])} "
                f"| {_cell(p['p2.5'])} | {_cell(p['p97.5'])} | {_cell(p['within'], 2)} "
                f"| {p['n_valid']}/{p['n_draws']} |"
            )
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--output-folder", required=True, help="the gym's output folder (verdicts/ inside)")
    ap.add_argument("--record", required=True, help="one synthetic-arm record .json from that folder")
    ap.add_argument("--out", required=True, help="where to write the JSON (grid, table, full-data reference)")
    ap.add_argument("--truth", default=None, help="JSON {model: score} to use instead of agreement.truth_ranking")
    ap.add_argument("--draws", type=int, default=DRAWS, help="random subsamples per grid point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queries", default=",".join(map(str, QUERY_GRID)), help="query counts, comma separated")
    ap.add_argument("--pair-fractions", default=",".join(map(str, PAIR_FRACTIONS)), help="comma separated")
    ap.add_argument("--min-models", type=int, default=MIN_MODELS, help="smallest model count on the models axis")
    args = ap.parse_args(argv)
    truth = json.loads(Path(args.truth).read_text()) if args.truth else None
    result = scaling(
        args.output_folder,
        args.record,
        truth=truth,
        draws=args.draws,
        seed=args.seed,
        query_grid=[int(q) for q in args.queries.split(",")],
        pair_fractions=[float(f) for f in args.pair_fractions.split(",")],
        min_models=args.min_models,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(result["table"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
