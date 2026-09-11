"""
Baseline ranking without a judge, from the seed documents behind each synthetic query.

Every synthetic query was written from a few seed documents (queries/<query_set>.json, seed_doc_ids).
Treating those seed documents as the only relevant documents gives a synthetic qrels for free, and
nDCG@k against them ranks the models with no judge at all. This is the AIR-Bench recipe the paper
discusses as a comparator: if this judge-free ranking agrees with the official ranking as well as
the gym's Bradley-Terry ranking does, the judge adds nothing on that corpus.

    seed_ndcg      per model, mean nDCG@k over the queries that carry at least one seed document
    vs_gym         Spearman and Kendall between the seed-label ranking and the gym's rating ranking
    vs_truth       the same against the official ranking: record["agreement"]["truth_ranking"] read
                   ordinally (first = best), or a --truth JSON of {model: score}, higher better
    gym_vs_truth   the gym's own agreement with that ranking, so the two comparators sit side by side
    coverage       queries where at least one model put a seed document in its top k; a query nobody
                   covers scores 0 for every model and says nothing about their order

Seed labels are weaker than human qrels: the seed documents are relevant by construction, but every
other relevant document in the corpus is unlabelled, so the absolute nDCG values are not comparable
with official scores. Only the order is used.

    python -m analysis.seed_baseline --output-folder results --record results/records/<synthetic>.json \\
        --out analysis/out/seed_baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mteb_gym.reliability import per_query_ndcg, prediction_file


class SeedBaselineError(Exception):
    pass


# ----------------------------------------------------------------------------- the run's artifacts
def _check_template(path: Path) -> str:
    """File contents are data; a file that looks like an unfilled template is refused, not interpreted."""
    text = path.read_text()
    if "${" in text:
        raise SeedBaselineError(f"{path} contains template-looking text ('${{...}}'); refusing to read it")
    return text


def _read_json(path: Path) -> Any:
    return json.loads(_check_template(path))


def queries_file(out: Path, record: dict) -> Path:
    return out / "queries" / f"{record['config']['query_set']}.json"


def seed_qrels(path: Path) -> dict[str, dict[str, float]]:
    """{qid: {seed_doc_id: 1.0}} from the query set; queries without a seed document are left out."""
    data = _read_json(path)
    return {str(q["qid"]): {str(d): 1.0 for d in q["seed_doc_ids"]} for q in data["queries"] if q.get("seed_doc_ids")}


def seed_ndcg(out: Path, record: dict, qrels: Mapping[str, Mapping[str, float]], k: int) -> dict[str, dict[str, float]]:
    """{model: {qid: nDCG@k}} against the seed qrels, from each model's prediction file. The gym runs every
    query in the set through mteb, so a prediction file missing a seeded query is refused rather than
    averaged over fewer queries than the other models."""
    scores = {}
    for rating in record["ratings"]:
        model = rating["model"]
        path = prediction_file(out, record, model)
        if not path.exists():
            raise SeedBaselineError(f"no predictions for {model} at {path}")
        _check_template(path)
        scores[model] = per_query_ndcg(path, qrels, k)
        missing = sorted(set(qrels) - set(scores[model]))
        if missing:
            shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
            raise SeedBaselineError(f"{path} has no rows for {len(missing)} seeded queries ({shown})")
    return scores


# ----------------------------------------------------------------------------- scores and rankings
def coverage(ndcg: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, Any]:
    """A query is covered when some model has nDCG > 0 on it, i.e. put a seed document in its top k."""
    covered = [q for q in qids if any(per_model.get(q, 0.0) > 0 for per_model in ndcg.values())]
    return {
        "n_queries": len(qids),
        "n_covered": len(covered),
        "fraction": len(covered) / len(qids) if qids else None,
        "uncovered": [q for q in qids if q not in covered],
    }


def mean_ndcg(ndcg: Mapping[str, Mapping[str, float]]) -> dict[str, float | None]:
    return {m: (sum(v.values()) / len(v) if v else None) for m, v in ndcg.items()}


def ranking(scores: Mapping[str, float | None]) -> list[str]:
    """Models with a score, best first; ties keep the input order."""
    return sorted((m for m in scores if scores[m] is not None), key=lambda m: -scores[m])


def gym_ratings(record: dict) -> dict[str, float]:
    return {r["model"]: float(r["rating"]) for r in record["ratings"]}


def truth_scores(record: dict, truth: Path | None) -> tuple[dict[str, float] | None, str | None]:
    """{model: score}, higher better, and where it came from. Without --truth the record's official
    ranking is read ordinally (first = best); a record whose agreement failed has none."""
    if truth is not None:
        data = _read_json(truth)
        if not isinstance(data, dict):
            raise SeedBaselineError(f"{truth}: expected a JSON object of {{model: score}}")
        return {str(m): float(s) for m, s in data.items()}, f"--truth {truth}"
    order = (record.get("agreement") or {}).get("truth_ranking")
    if not order:
        return None, None
    return {m: float(len(order) - i) for i, m in enumerate(order)}, "record.agreement.truth_ranking (ordinal)"


def _finite(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def correlate(x: Mapping[str, float | None], y: Mapping[str, float | None] | None) -> dict[str, Any]:
    """Spearman and Kendall over the models present in both; fewer than three is an error, not a number."""
    if y is None:
        return {"error": "no official ranking: agreement failed on the record and no --truth given", "n_models": 0}
    from scipy.stats import kendalltau, spearmanr

    shared = [m for m in x if m in y and x[m] is not None and y[m] is not None]
    if len(shared) < 3:
        return {"error": f"need >=3 shared models, have {len(shared)}", "n_models": len(shared)}
    a, b = [x[m] for m in shared], [y[m] for m in shared]
    rho, p_rho = spearmanr(a, b)
    tau, p_tau = kendalltau(a, b)
    return {
        "n_models": len(shared),
        "spearman_rho": _finite(rho),
        "spearman_p": _finite(p_rho),
        "kendall_tau": _finite(tau),
        "kendall_p": _finite(p_tau),
    }


def _rank_of(order: Sequence[str]) -> dict[str, int]:
    return {m: i + 1 for i, m in enumerate(order)}


# ----------------------------------------------------------------------------- entry points
def seed_baseline(
    output_folder: str | Path, record_path: str | Path, *, k: int = 10, truth: Path | None = None
) -> dict:
    """The seed-label baseline for one synthetic-arm record, as a JSON-ready dict."""
    out, record = Path(output_folder), _read_json(Path(record_path))
    c = record["config"]
    if c.get("arm") != "synthetic":
        raise SeedBaselineError(f"arm is {c.get('arm')!r}: seed labels exist only for the synthetic arm")
    qrels = seed_qrels(queries_file(out, record))
    if not qrels:
        raise SeedBaselineError(f"no query in {queries_file(out, record)} carries seed_doc_ids")
    ndcg = seed_ndcg(out, record, qrels, k)
    seed, gym = mean_ndcg(ndcg), gym_ratings(record)
    truth_s, truth_source = truth_scores(record, truth)
    orders = {"seed": ranking(seed), "gym": ranking(gym), "truth": ranking(truth_s) if truth_s else None}
    ranks = {name: _rank_of(order or []) for name, order in orders.items()}
    rows = [
        {
            "model": m,
            "seed_ndcg": seed[m],
            "n_queries_scored": len(ndcg[m]),
            "gym_rating": gym[m],
            "truth_score": (truth_s or {}).get(m),
            "seed_rank": ranks["seed"].get(m),
            "gym_rank": ranks["gym"].get(m),
            "truth_rank": ranks["truth"].get(m),
        }
        for m in orders["seed"]
    ]
    return {
        "task_name": record["task_name"],
        "record": str(record_path),
        "config_hash": c.get("config_hash"),
        "query_set": c["query_set"],
        "k": k,
        "n_models": len(rows),
        "coverage": coverage(ndcg, sorted(qrels)),
        "models": rows,
        "seed_ranking": orders["seed"],
        "gym_ranking": orders["gym"],
        "truth_ranking": orders["truth"],
        "truth_source": truth_source,
        "vs_gym": correlate(seed, gym),
        "vs_truth": correlate(seed, truth_s),
        "gym_vs_truth": correlate(gym, truth_s),
    }


def _corr_line(label: str, r: Mapping[str, Any]) -> str:
    if "error" in r:
        return f"- {label}: not computed ({r['error']})"
    if r["spearman_rho"] is None or r["kendall_tau"] is None:
        return f"- {label}: undefined, a constant input has no rank correlation (n={r['n_models']})"
    return (
        f"- {label}: Spearman {r['spearman_rho']:.3f} (p={r['spearman_p']:.2f}), "
        f"Kendall {r['kendall_tau']:.3f} (p={r['kendall_p']:.2f}), n={r['n_models']}"
    )


def format_markdown(report: Mapping[str, Any]) -> str:
    """The report as a markdown table plus the coverage and correlation lines."""
    k, cov = report["k"], report["coverage"]
    lines = [
        f"# Seed-label baseline: {report['task_name']} ({report['query_set']})",
        "",
        f"| Rank | Model | Seed nDCG@{k} | Gym rating (rank) | Official (rank) |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in report["models"]:
        official = "-" if r["truth_score"] is None else f"{r['truth_score']:.2f} ({r['truth_rank']})"
        lines.append(
            f"| {r['seed_rank']} | {r['model']} | {r['seed_ndcg']:.4f} | "
            f"{r['gym_rating']:.1f} ({r['gym_rank']}) | {official} |"
        )
    uncovered = f" (uncovered: {', '.join(cov['uncovered'])})" if cov["uncovered"] else ""
    lines += [
        "",
        f"- coverage: {cov['n_covered']}/{cov['n_queries']} queries have a seed document in some model's top {k}"
        + uncovered,
        _corr_line("seed vs gym", report["vs_gym"]),
        _corr_line("seed vs official", report["vs_truth"]),
        _corr_line("gym vs official", report["gym_vs_truth"]),
        f"- official ranking source: {report['truth_source'] or 'none'}",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--output-folder", default="results", help="the gym's output folder (records/ inside)")
    ap.add_argument("--record", required=True, help="a synthetic-arm record under that folder")
    ap.add_argument("--out", default="analysis/out/seed_baseline.json", help="JSON report; .md is written next to it")
    ap.add_argument("--k", type=int, default=10, help="nDCG cutoff")
    ap.add_argument("--truth", type=Path, default=None, help="JSON {model: score} to use as the official ranking")
    args = ap.parse_args(argv)
    report = seed_baseline(args.output_folder, args.record, k=args.k, truth=args.truth)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    md = out.with_suffix(".md")
    md.write_text(format_markdown(report) + "\n")
    print(format_markdown(report))
    print(f"\n-> {out}, {md}")


if __name__ == "__main__":
    main()
