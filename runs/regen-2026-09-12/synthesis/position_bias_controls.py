"""Controls that separate de-biasing from de-noising in the two-order judge design.

position_bias.py reports, per record, rho_both (Bradley-Terry refit from the two-order averaged
score_a, Spearman vs official nDCG@10), rho_order1 (refit from SCORE_OF[w1] only) and rho_order2 (refit
from 1 - SCORE_OF[w2] only). The single-order refits are not comparable with each other because both use
the same pair enumeration, so order 1 always hands the first-position bonus to each pair file's
model_a and order 2 to its model_b, and the roster (config.models) correlates with quality. Two
effects are confounded in "rho_both beats a single-order refit": de-biasing (the two-order average
cancels the slot bonus) and de-noising (a single-order refit rests on one verdict per pair instead
of two). Three controls computed from the raw verdict rows separate them.

  C1 random-order refit. For every verdict row (one model pair on one query) draw independently
     which order's verdict to use, SCORE_OF[w1] or 1 - SCORE_OF[w2], refit, Spearman vs truth. One verdict per pair
     like a single-order refit, but the slot bonus lands on a random model per pair. The draw is per
     row (pair x query), not once per model pair: a per-pair coin would leave every query of that pair
     with the same slot bonus, and C1 would no longer be unbiased in expectation.
  C2 roster-permuted fixed-slot refit. Per draw, permute the roster; for every row use SCORE_OF[w1] if
     model_a precedes model_b in the permuted roster, else 1 - SCORE_OF[w2]. The fixed-slot design with a
     random roster. The identity permutation reproduces rho_order1 and the reversed roster
     rho_order2 when every pair file is oriented along the roster (checked and reported). rho_order1
     and rho_order2 are then compared with the C2 2.5 to 97.5 band per record.
  C3 half-queries both-order refit. A random half of the queries (by qid), both orders averaged,
     refit. The same number of judge verdicts as a single-order refit over all queries, but unbiased.
  C1 per pair (c1_pair). As C1 but one random order per model pair per draw, applied to all of that
     pair's queries, so a pair's slot bonus repeats across its queries. This is the deployment where the
     presentation order is fixed per pair; C1 is the deployment where the order is drawn per query. Same
     number of draws and the same fields as C1, from its own random stream seeded from (seed, 2).
  C2 at 1000 draws (c2_1000). C2 redrawn --c2-draws times (default 1000) from its own random stream
     seeded from (seed, 1), so a 2.5 percent band edge rests on about 25 draws instead of 5. rho_order1
     and rho_order2 are compared with both bands and the cases that change status are listed. The C1, C2
     and C3 streams are untouched by the two additions, so their draws are identical to the 200-draw run.

Decomposition per record
  bias share  = mean(C1) - mean(rho_order1, rho_order2)
  noise share = rho_both - mean(C1)
  both minus C3 = rho_both - mean(C3)   (the pure noise cost of halving the verdict count)
  C1 minus C1 per pair = mean(C1) - mean(C1 per pair)   (the cost of fixing the order per pair)

Arm-level tests are two-sided Wilcoxon signed-rank over the records (scipy default, exact at n = 14)
and the mean over its standard error across records.

spearman_ci95. The interval in a record's agreement block comes from mteb_gym.agreement.correlate: it
resamples the shared models with replacement (bootstrap=1000 resamples, seed 0, as passed by
regen_nano.py; a resample with fewer than 3 distinct models or an undefined rho is dropped), recomputes
Spearman between the fixed point-estimate ratings and truth on each resample, and takes the 2.5 and
97.5 percentiles. Queries and verdicts are not resampled, so it is a model-roster interval, not a
query-level or judge-noise uncertainty. The script recomputes it with that function from the record
ratings and truth.json for every record that carries one and fails loudly on a mismatch beyond 1e-9.

Scoring conventions follow position_bias.py. SCORE_OF = {A: 1, tie: 0.5, B: 0}; raw == ["identical"] and any
raw token outside {A, B, tie} score 0.5 in every refit and are excluded from decisive counts. The
refit is the package's Bradley-Terry (mteb_gym.rank._bradley_terry on the same win matrix that
mteb_gym.rank.rate builds), which reproduces the stored record ratings. The script fails loudly on
any record whose recomputed a_first_rate differs from the record, whose recomputed rho_both differs
from the record where the record carries one (synthetic arm; original records carry no
agreement.spearman_rho), or whose recomputed rho_both, rho_order1, rho_order2 or a_first_rate
differs from position_bias.json, beyond 1e-9.

Tie-break. The design is balanced in every record (each model plays the same number of games), so
the Bradley-Terry order is the order of total fractional wins. Where two models have exactly equal
total wins the fitted strengths differ only by the MM stopping tolerance (a rating gap of about 1e-8)
and spearmanr scores that as a strict rank difference. rho_both_tie_aware gives tied models the mean
of their ratings before correlating; it is reported alongside rho_both and does not enter the
decomposition.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from mteb_gym import agreement as agree
from mteb_gym.judge import Verdict
from mteb_gym.rank import BASE, SCALE, _bradley_terry, rate
from mteb_gym.reliability import verdict_file

SCORE_OF = {"A": 1.0, "tie": 0.5, "B": 0.0}
TEMPLATE_MARK = "$" + "{"
TOL = 1e-9
CI95_BOOTSTRAP, CI95_SEED = 1000, 0  # regen_nano.py: res.agreement(bootstrap=BOOTSTRAP, seed=0) with BOOTSTRAP = 1000
SIGNED_KEYS = ("bias_share", "noise_share", "both_minus_c3", "c1_minus_c2", "c1_minus_c1_pair")
BAND_FLAGS = (
    ("rho_order1", "order1_in_c2_band", "order1_in_c2_1000_band"),
    ("rho_order2", "order2_in_c2_band", "order2_in_c2_1000_band"),
)


def load_rows(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    p2 = path.with_suffix(".jsonl")
    if p2.exists():
        return [json.loads(line) for line in p2.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(str(path))


def pair_rows(out: Path, rec: dict, roster: list[str]) -> tuple[list[dict], int]:
    """Verdict rows for every roster pair, enumerated in roster order (a before b), with the
    reversed file as fallback. Returns the rows and the number of files found reversed."""
    rows: list[dict] = []
    n_reversed = 0
    for i, a in enumerate(roster):
        for b in roster[i + 1 :]:
            p = verdict_file(out, rec, a, b)
            if not (p.exists() or p.with_suffix(".jsonl").exists()):
                p = verdict_file(out, rec, b, a)
                n_reversed += 1
            rows.extend(load_rows(p))
    return rows, n_reversed


def spearman(ratings: np.ndarray, names: list[str], truth: dict) -> float:
    shared = [i for i, m in enumerate(names) if m in truth]
    if len(shared) < 3:
        raise RuntimeError("fewer than 3 models shared with truth")
    return float(spearmanr(ratings[shared], [truth[names[i]] for i in shared])[0])


class Refitter:
    """Vectorised win-matrix builder over the record's verdict rows plus the package's BT fit."""

    def __init__(self, names: list[str], ia: np.ndarray, ib: np.ndarray, truth: dict):
        self.names, self.ia, self.ib, self.truth = names, ia, ib, truth
        self.n = len(names)

    def win_matrix(self, score_a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        ia, ib, s = self.ia, self.ib, score_a
        if mask is not None:
            ia, ib, s = ia[mask], ib[mask], s[mask]
        W = np.zeros((self.n, self.n))
        np.add.at(W, (ia, ib), s)
        np.add.at(W, (ib, ia), 1.0 - s)
        return W

    def ratings(self, score_a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        r = SCALE * np.log10(np.clip(_bradley_terry(self.win_matrix(score_a, mask)), 1e-12, None))
        return r - r.mean() + BASE

    def rho(self, score_a: np.ndarray, mask: np.ndarray | None = None) -> float:
        return spearman(self.ratings(score_a, mask), self.names, self.truth)

    def tie_aware(self, score_a: np.ndarray) -> dict:
        """Exact ties in total fractional wins on a balanced design, and rho with those ties snapped
        to the mean rating. Wins are multiples of 0.25 so their float sums are exact."""
        W = self.win_matrix(score_a)
        wins, games = W.sum(axis=1), (W + W.T).sum(axis=1)
        balanced = bool(np.all(games == games[0]))
        r = self.ratings(score_a)
        snapped, ties = r.copy(), []
        if balanced:
            for w in np.unique(wins):
                idx = np.flatnonzero(wins == w)
                if len(idx) > 1:
                    snapped[idx] = r[idx].mean()
                    ties.append(
                        {
                            "models": [self.names[i] for i in idx],
                            "wins": float(w),
                            "rating_gap": float(r[idx].max() - r[idx].min()),
                        }
                    )
        return {
            "balanced_design": balanced,
            "games_per_model": int(games[0]) if balanced else None,
            "exact_win_ties": ties,
            "rho_both_tie_aware": spearman(snapped, self.names, self.truth),
        }


def parse_rows(rows: list[dict], task: str) -> dict:
    """Per-row arrays. s_both = stored score_a, s1 = SCORE_OF[w1], s2 = 1 - SCORE_OF[w2]."""
    n = len(rows)
    s_both, s1, s2 = np.empty(n), np.empty(n), np.empty(n)
    n_ident = n_invalid = first = decisive = 0
    for k, v in enumerate(rows):
        raw = v["raw"]
        s_both[k] = float(v["score_a"])
        if raw == ["identical"] or len(raw) != 2:
            n_ident += raw == ["identical"]
            n_invalid += raw != ["identical"]
            s1[k] = s2[k] = 0.5
            continue
        w1, w2 = raw
        for w in raw:
            if w in ("A", "B"):
                decisive += 1
                first += w == "A"
            elif w != "tie":
                n_invalid += 1
        s1[k], s2[k] = SCORE_OF.get(w1, 0.5), 1.0 - SCORE_OF.get(w2, 0.5)
        if w1 in SCORE_OF and w2 in SCORE_OF and abs((s1[k] + s2[k]) / 2 - s_both[k]) > TOL:
            raise RuntimeError(f"score_a mismatch in {task} qid={v['qid']} raw={raw} score_a={v['score_a']}")
    return dict(
        s_both=s_both,
        s1=s1,
        s2=s2,
        n_identical=n_ident,
        n_invalid_raw=n_invalid,
        n_decisive_order_verdicts=decisive,
        n_first_wins=first,
        a_first_rate=(first / decisive) if decisive else None,
    )


def check_against_package(rows: list[dict], rec: dict, fit: Refitter, s_both: np.ndarray) -> float:
    """rate() on the stored score_a must reproduce the record ratings and the vectorised refit."""
    verdicts = [
        Verdict(
            qid=v["qid"],
            query=v.get("query", ""),
            model_a=v["model_a"],
            model_b=v["model_b"],
            score_a=float(v["score_a"]),
        )
        for v in rows
    ]
    pkg = {m.name: m.rating for m in rate(verdicts, bootstrap=0, seed=0)}
    rec_r = {r["model"]: r["rating"] for r in rec["ratings"]}
    vec = dict(zip(fit.names, fit.ratings(s_both)))
    d_rec = max(abs(pkg[m] - rec_r[m]) for m in pkg)
    d_vec = max(abs(pkg[m] - vec[m]) for m in pkg)
    if d_rec > 1e-6 or d_vec > 1e-6:
        raise RuntimeError(f"rating mismatch: package vs record {d_rec:.2e}, package vs vectorised {d_vec:.2e}")
    return max(d_rec, d_vec)


def pct(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "p2_5": float(np.percentile(x, 2.5)),
        "p97_5": float(np.percentile(x, 97.5)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def in_band(x: float, d: dict) -> bool:
    return bool(d["p2_5"] <= x <= d["p97_5"])


def controls(
    fit: Refitter, arr: dict, rows: list[dict], roster: list[str], draws: int, seed: int, c2_draws: int
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)
    s1, s2, s_both = arr["s1"], arr["s2"], arr["s_both"]
    # C1 random order per verdict row
    c1 = np.array([fit.rho(np.where(rng.random(n) < 0.5, s1, s2)) for _ in range(draws)])
    # C2 fixed slot under a permuted roster
    pos = {m: i for i, m in enumerate(roster)}
    ra = np.array([pos[v["model_a"]] for v in rows])
    rb = np.array([pos[v["model_b"]] for v in rows])

    def c2_rho(perm: np.ndarray) -> float:  # perm[i] = new position of roster index i
        return fit.rho(np.where(perm[ra] < perm[rb], s1, s2))

    c2 = np.array([c2_rho(rng.permutation(len(roster))) for _ in range(draws)])
    c2_identity = c2_rho(np.arange(len(roster)))
    c2_reversed = c2_rho(np.arange(len(roster))[::-1])
    # C3 both orders on a random half of the queries
    qids = np.array(sorted({v["qid"] for v in rows}), dtype=object)
    q_of_row = np.array([v["qid"] for v in rows], dtype=object)
    half = len(qids) // 2
    c3 = []
    for _ in range(draws):
        keep = set(rng.permutation(qids)[:half].tolist())
        c3.append(fit.rho(s_both, np.array([q in keep for q in q_of_row])))
    c3 = np.array(c3)
    # C2 again at c2_draws, from its own stream so the C1, C2 and C3 draws above are unchanged
    rng_c2 = np.random.default_rng([seed, 1])
    c2k = np.array([c2_rho(rng_c2.permutation(len(roster))) for _ in range(c2_draws)])
    # C1 with one random order per model pair per draw (the order fixed across the pair's queries)
    rng_pair = np.random.default_rng([seed, 2])
    pair_keys = sorted({tuple(sorted((v["model_a"], v["model_b"]))) for v in rows})
    pair_idx = {p: i for i, p in enumerate(pair_keys)}
    pair_of_row = np.array([pair_idx[tuple(sorted((v["model_a"], v["model_b"])))] for v in rows])
    c1p = np.array(
        [fit.rho(np.where((rng_pair.random(len(pair_keys)) < 0.5)[pair_of_row], s1, s2)) for _ in range(draws)]
    )
    return dict(
        c1=pct(c1),
        c2=pct(c2),
        c3=pct(c3),
        c2_identity=c2_identity,
        c2_reversed=c2_reversed,
        c2_1000={**pct(c2k), "se": float(np.std(c2k, ddof=1) / np.sqrt(c2_draws)), "draws": c2_draws},
        c1_pair=pct(c1p),
        n_model_pairs=len(pair_keys),
        n_unique_qids=int(len(qids)),
        n_half_queries=half,
        c1_draws=c1.tolist(),
        c2_draws=c2.tolist(),
        c3_draws=c3.tolist(),
        c2_1000_draws=c2k.tolist(),
        c1_pair_draws=c1p.tolist(),
    )


def analyse(args: tuple) -> dict:
    rec_path, root, draws, seed, prior, c2_draws = args
    t0 = time.time()
    out, an = root / "results", root / "analysis_out"
    if TEMPLATE_MARK in rec_path.name:
        raise RuntimeError(f"template-looking record name: {rec_path.name}")
    rec = json.loads(rec_path.read_text())
    task, arm = rec["task_name"], rec["config"]["arm"]
    truth_text = (an / task / "truth.json").read_text()
    if TEMPLATE_MARK in truth_text:
        raise RuntimeError(f"template-looking text in truth for {task}")
    truth = json.loads(truth_text)
    roster = list(rec["config"]["models"])
    if set(roster) != {r["model"] for r in rec["ratings"]}:
        raise RuntimeError(f"{rec_path.name}: config.models and ratings name different model sets")
    rows, n_reversed = pair_rows(out, rec, roster)
    names = sorted({m for v in rows for m in (v["model_a"], v["model_b"])})
    idx = {m: i for i, m in enumerate(names)}
    fit = Refitter(
        names, np.array([idx[v["model_a"]] for v in rows]), np.array([idx[v["model_b"]] for v in rows]), truth
    )
    arr = parse_rows(rows, task)
    max_rating_diff = check_against_package(rows, rec, fit, arr["s_both"])

    rho_both, rho_o1, rho_o2 = fit.rho(arr["s_both"]), fit.rho(arr["s1"]), fit.rho(arr["s2"])
    checks = verify(rec, rec_path.name, prior, rho_both, rho_o1, rho_o2, arr["a_first_rate"], truth)
    ties = fit.tie_aware(arr["s_both"])
    ctl = controls(fit, arr, rows, roster, draws, seed, c2_draws)
    single_mean = (rho_o1 + rho_o2) / 2
    ci95 = (rec.get("agreement") or {}).get("spearman_ci95")
    ci95 = list(ci95) if ci95 and ci95[0] is not None else None
    roster_in_truth = [m for m in roster if m in truth]
    roster_truth_rho = float(spearmanr(range(len(roster_in_truth)), [truth[m] for m in roster_in_truth])[0])
    return {
        "task": task,
        "arm": arm,
        "record": rec_path.name,
        "n_models": len(names),
        "n_models_in_truth": sum(m in truth for m in names),
        "n_queries": rec["config"]["n_queries"],
        "n_unique_qids": ctl.pop("n_unique_qids"),
        "n_half_queries": ctl.pop("n_half_queries"),
        "n_pairs_judged": len(rows),
        "n_pair_files_reversed": n_reversed,
        "n_model_pairs": ctl["n_model_pairs"],
        "n_identical": arr["n_identical"],
        "n_invalid_raw": arr["n_invalid_raw"],
        "rho_both": rho_both,
        "rho_order1": rho_o1,
        "rho_order2": rho_o2,
        "single_order_mean": single_mean,
        "a_first_rate": arr["a_first_rate"],
        "roster_truth_rho": roster_truth_rho,
        "spearman_ci95": ci95,
        "spearman_ci95_width": (ci95[1] - ci95[0]) if ci95 else None,
        "max_abs_rating_diff": max_rating_diff,
        "checks": checks,
        **ties,
        "c1": ctl["c1"],
        "c2": ctl["c2"],
        "c3": ctl["c3"],
        "c1_pair": ctl["c1_pair"],
        "c2_1000": ctl["c2_1000"],
        "c2_identity": ctl["c2_identity"],
        "c2_reversed": ctl["c2_reversed"],
        "c2_identity_equals_order1": abs(ctl["c2_identity"] - rho_o1) < TOL,
        "c2_reversed_equals_order2": abs(ctl["c2_reversed"] - rho_o2) < TOL,
        "order1_in_c2_band": in_band(rho_o1, ctl["c2"]),
        "order2_in_c2_band": in_band(rho_o2, ctl["c2"]),
        "order1_in_c2_1000_band": in_band(rho_o1, ctl["c2_1000"]),
        "order2_in_c2_1000_band": in_band(rho_o2, ctl["c2_1000"]),
        "bias_share": ctl["c1"]["mean"] - single_mean,
        "noise_share": rho_both - ctl["c1"]["mean"],
        "both_minus_c3": rho_both - ctl["c3"]["mean"],
        "c1_minus_c1_pair": ctl["c1"]["mean"] - ctl["c1_pair"]["mean"],
        "draws": {
            "c1": ctl["c1_draws"],
            "c2": ctl["c2_draws"],
            "c3": ctl["c3_draws"],
            "c1_pair": ctl["c1_pair_draws"],
            "c2_1000": ctl["c2_1000_draws"],
        },
        "seconds": time.time() - t0,
    }


def check_ci95(rec: dict, name: str, truth: dict) -> tuple[str, float | None]:
    """Recompute the record's spearman_ci95 with the package's model bootstrap (models resampled with
    replacement, CI95_BOOTSTRAP resamples, seed CI95_SEED) from the record ratings and truth.json."""
    ci = (rec.get("agreement") or {}).get("spearman_ci95")
    if not ci or ci[0] is None:
        return "absent (record carries no agreement block)", None
    ratings = {r["model"]: r["rating"] for r in rec["ratings"]}  # record order, as Result.agreement builds it
    got = agree.correlate(ratings, truth, bootstrap=CI95_BOOTSTRAP, seed=CI95_SEED)["spearman_ci95"]
    diff = max(abs(got[0] - ci[0]), abs(got[1] - ci[1]))
    if diff > TOL:
        raise RuntimeError(
            f"{name}: spearman_ci95 {got!r} (model bootstrap, {CI95_BOOTSTRAP} resamples) != record {ci!r}"
        )
    return f"match (model bootstrap, {CI95_BOOTSTRAP} resamples)", diff


def verify(
    rec: dict, name: str, prior: dict | None, rho_both: float, o1: float, o2: float, af: float | None, truth: dict
) -> dict:
    """Loud failure on any mismatch with the record or with position_bias.json."""
    rec_rho = (rec.get("agreement") or {}).get("spearman_rho")
    rec_af = rec["diagnostics"].get("a_first_rate")
    checks = {"rho_both_record": rec_rho, "a_first_rate_record": rec_af}
    if rec_rho is not None and abs(rec_rho - rho_both) > TOL:
        raise RuntimeError(f"{name}: rho_both {rho_both!r} != record {rec_rho!r}")
    if rec_af is None or af is None or abs(rec_af - af) > TOL:
        raise RuntimeError(f"{name}: a_first_rate {af!r} != record {rec_af!r}")
    checks["spearman_ci95"], checks["spearman_ci95_max_abs_edge_diff"] = check_ci95(rec, name, truth)
    if prior is not None:
        p = next((r for r in prior[rec["config"]["arm"]] if r["record"] == name), None)
        if p is None:
            raise RuntimeError(f"{name}: not present in position_bias.json")
        for key, val in (("rho_both", rho_both), ("rho_order1", o1), ("rho_order2", o2), ("a_first_rate", af)):
            if abs(p[key] - val) > TOL:
                raise RuntimeError(f"{name}: {key} {val!r} != position_bias.json {p[key]!r}")
        checks["position_bias_json"] = "match"
    checks["record"] = "match" if rec_rho is not None else "a_first_rate match, no record rho (original arm)"
    return checks


def signed_test(v: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank over records plus mean over its standard error."""
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return {
        "n_positive": int((v > 0).sum()),
        "wilcoxon_p": float(wilcoxon(v).pvalue),
        "mean_over_se": float(v.mean() / se),
    }


TEST_LABELS = {
    "bias_share": "bias share",
    "noise_share": "noise share",
    "both_minus_c3": "both minus C3",
    "c1_minus_c2": "mean(C1) minus mean(C2)",
    "c1_minus_c1_pair": "C1 minus C1 per pair",
}


def multiple_testing(S: dict, alpha: float = 0.05) -> dict:
    """Bonferroni, Holm and Benjamini-Hochberg over every signed-rank test on every arm, at level alpha."""
    tests = sorted(
        (S[arm][k]["wilcoxon_p"], arm, k) for arm in ("synthetic", "original") if arm in S for k in SIGNED_KEYS
    )
    m = len(tests)
    bonferroni = [t for t in tests if t[0] <= alpha / m]
    holm: list[tuple] = []
    for i, t in enumerate(tests):
        if t[0] > alpha / (m - i):
            break
        holm.append(t)
    k_max = max((i + 1 for i, t in enumerate(tests) if t[0] <= (i + 1) * alpha / m), default=0)

    def names(ts: list[tuple]) -> list[str]:
        return [f"{arm} {TEST_LABELS[k]}" for _, arm, k in ts]

    return {
        "alpha": alpha,
        "n_tests": m,
        "sorted": [{"arm": a, "key": k, "p": p} for p, a, k in tests],
        "bonferroni": names(bonferroni),
        "holm": names(holm),
        "benjamini_hochberg": names(tests[:k_max]),
    }


def summarise(rows: list[dict], draws: int, c2_draws: int) -> dict:
    keys = {
        "rho_both": lambda r: r["rho_both"],
        "rho_both_tie_aware": lambda r: r["rho_both_tie_aware"],
        "rho_order1": lambda r: r["rho_order1"],
        "rho_order2": lambda r: r["rho_order2"],
        "single_order_mean": lambda r: r["single_order_mean"],
        "c1_mean": lambda r: r["c1"]["mean"],
        "c2_mean": lambda r: r["c2"]["mean"],
        "c3_mean": lambda r: r["c3"]["mean"],
        "c1_sd": lambda r: r["c1"]["sd"],
        "c2_sd": lambda r: r["c2"]["sd"],
        "c3_sd": lambda r: r["c3"]["sd"],
        "bias_share": lambda r: r["bias_share"],
        "noise_share": lambda r: r["noise_share"],
        "both_minus_c3": lambda r: r["both_minus_c3"],
        "a_first_rate": lambda r: r["a_first_rate"],
        "c1_minus_c2": lambda r: r["c1"]["mean"] - r["c2"]["mean"],
        "c1_pair_mean": lambda r: r["c1_pair"]["mean"],
        "c1_pair_sd": lambda r: r["c1_pair"]["sd"],
        "c2_1000_mean": lambda r: r["c2_1000"]["mean"],
        "c2_1000_sd": lambda r: r["c2_1000"]["sd"],
        "c1_minus_c1_pair": lambda r: r["c1_minus_c1_pair"],
    }
    s = {}
    for k, f in keys.items():
        v = np.array([f(r) for r in rows])
        s[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(len(v))}
    for k in SIGNED_KEYS:
        v = np.array([keys[k](r) for r in rows])
        s[k].update(signed_test(v))
        s[k].update({"min": float(v.min()), "max": float(v.max())})
    s["n_records_c1_above_single_mean"] = int(sum(r["c1"]["mean"] > r["single_order_mean"] for r in rows))
    s["n_records_both_above_c1"] = int(sum(r["rho_both"] > r["c1"]["mean"] for r in rows))
    s["n_records_both_above_c3"] = int(sum(r["rho_both"] > r["c3"]["mean"] for r in rows))
    s["n_records_c1_above_c1_pair"] = int(sum(r["c1"]["mean"] > r["c1_pair"]["mean"] for r in rows))
    s["n_records_c1_within_c2_band"] = int(sum(in_band(r["c1"]["mean"], r["c2"]) for r in rows))
    s["n_records_c1_within_c2_1000_band"] = int(sum(in_band(r["c1"]["mean"], r["c2_1000"]) for r in rows))
    s["n_records_both_within_c3_band"] = int(sum(in_band(r["rho_both"], r["c3"]) for r in rows))
    s["n_records_order1_within_c2_band"] = int(sum(r["order1_in_c2_band"] for r in rows))
    s["n_records_order2_within_c2_band"] = int(sum(r["order2_in_c2_band"] for r in rows))
    s["n_records_order1_within_c2_1000_band"] = int(sum(r["order1_in_c2_1000_band"] for r in rows))
    s["n_records_order2_within_c2_1000_band"] = int(sum(r["order2_in_c2_1000_band"] for r in rows))
    s["max_abs_noise_share"] = float(max(abs(r["noise_share"]) for r in rows))
    s["max_abs_noise_share_over_c1_sd"] = float(max(abs(r["noise_share"]) / r["c1"]["sd"] for r in rows))
    s["max_mc_se"] = {c: float(max(r[c]["sd"] for r in rows) / np.sqrt(draws)) for c in ("c1", "c2", "c3", "c1_pair")}
    s["max_mc_se"]["c2_1000"] = float(max(r["c2_1000"]["sd"] for r in rows) / np.sqrt(c2_draws))
    s["n_records_with_win_ties"] = int(sum(bool(r["exact_win_ties"]) for r in rows))
    s["n_records_balanced"] = int(sum(r["balanced_design"] for r in rows))
    s["max_abs_tie_shift"] = float(max(abs(r["rho_both"] - r["rho_both_tie_aware"]) for r in rows))
    widths = [r["spearman_ci95_width"] for r in rows if r["spearman_ci95_width"] is not None]
    s["spearman_ci95_width"] = (
        {"n": len(widths), "mean": float(np.mean(widths)), "min": float(min(widths)), "max": float(max(widths))}
        if widths
        else {"n": 0}
    )
    diffs = [
        r["checks"]["spearman_ci95_max_abs_edge_diff"]
        for r in rows
        if r["checks"]["spearman_ci95_max_abs_edge_diff"] is not None
    ]
    s["spearman_ci95_max_abs_edge_diff"] = float(max(diffs)) if diffs else None
    return s


def f3(x: float) -> str:
    return f"{x:.3f}"


def band(d: dict) -> str:
    return f"{d['mean']:.3f} (sd {d['sd']:.3f}) [{d['p2_5']:.3f}, {d['p97_5']:.3f}]"


def short(model: str) -> str:
    return model.split("/")[-1]


def arm_paragraph(arm: str, s: dict, n: int, c2_draws: int) -> str:
    m = {k: s[k]["mean"] for k in s if isinstance(s[k], dict) and "mean" in s[k]}
    md = {k: s[k]["median"] for k in s if isinstance(s[k], dict) and "median" in s[k]}
    return (
        f"Over the {n} {arm} records the mean (median) rho_both is {f3(m['rho_both'])} ({f3(md['rho_both'])}), "
        f"rho_order1 {f3(m['rho_order1'])} ({f3(md['rho_order1'])}), rho_order2 {f3(m['rho_order2'])} ({f3(md['rho_order2'])}), "
        f"so the mean of the two single-order refits is {f3(m['single_order_mean'])} ({f3(md['single_order_mean'])}). "
        f"C1, one random-order verdict per pair, averages {f3(m['c1_mean'])} ({f3(md['c1_mean'])}) with a within-record sd of "
        f"{f3(m['c1_sd'])}; C2, the fixed-slot design under a random roster, averages {f3(m['c2_mean'])} ({f3(md['c2_mean'])}) "
        f"with a within-record sd of {f3(m['c2_sd'])}; C3, both orders on half the queries, averages {f3(m['c3_mean'])} "
        f"({f3(md['c3_mean'])}) with a within-record sd of {f3(m['c3_sd'])}. "
        f"The bias share, mean(C1) minus the single-order mean, is {f3(m['bias_share'])} on average (median {f3(md['bias_share'])}), "
        f"positive in {s['n_records_c1_above_single_mean']} of {n} records. The noise share, rho_both minus mean(C1), is "
        f"{f3(m['noise_share'])} on average (median {f3(md['noise_share'])}), positive in {s['n_records_both_above_c1']} of {n}. "
        f"rho_both minus mean(C3), the cost of halving the verdict count with both orders kept, is {f3(m['both_minus_c3'])} "
        f"on average (median {f3(md['both_minus_c3'])}), positive in {s['n_records_both_above_c3']} of {n}; rho_both lies inside "
        f"the C3 2.5 to 97.5 band in {s['n_records_both_within_c3_band']} of {n} records, and mean(C1) lies inside the C2 band in "
        f"{s['n_records_c1_within_c2_band']} of {n} ({s['n_records_c1_within_c2_1000_band']} of {n} against the "
        f"{c2_draws}-draw band). C1 per pair, one random order per model pair held across its queries, averages "
        f"{f3(m['c1_pair_mean'])} ({f3(md['c1_pair_mean'])}) with a within-record sd of {f3(m['c1_pair_sd'])}, so C1 minus "
        f"C1 per pair is {f3(m['c1_minus_c1_pair'])} on average (median {f3(md['c1_minus_c1_pair'])}, range "
        f"{f3(s['c1_minus_c1_pair']['min'])} to {f3(s['c1_minus_c1_pair']['max'])}), positive in "
        f"{s['n_records_c1_above_c1_pair']} of {n}. C2 at {c2_draws} draws averages {f3(m['c2_1000_mean'])} "
        f"({f3(md['c2_1000_mean'])}) with a within-record sd of {f3(m['c2_1000_sd'])}. The mean a_first_rate is "
        f"{f3(m['a_first_rate'])}."
    )


def neg_bias(rows: list[dict]) -> str:
    return (
        "; ".join(
            f"{r['task']} (rho_both {f3(r['rho_both'])}, a_first_rate {f3(r['a_first_rate'])})"
            for r in rows
            if r["bias_share"] < 0
        )
        or "none"
    )


def out_of_band(rows: list[dict], which: str = "c2") -> str:
    """Single-order refits outside the C2 band ('c2', the --draws band) or the c2_1000 band ('c2_1000')."""
    items = []
    for r in rows:
        for key, flag200, flag1000 in BAND_FLAGS:
            if not r[flag200 if which == "c2" else flag1000]:
                below = r[key] < r[which]["p2_5"]
                edge = r[which]["p2_5"] if below else r[which]["p97_5"]
                items.append(f"{r['task']} {key} {f3(r[key])} {'below' if below else 'above'} {f3(edge)}")
    return "; ".join(items) or "none"


def tail_share(value: float, draws: list[float]) -> float:
    """Share of C2 draws strictly beyond a single-order value, on the side of the band it sits nearer to."""
    d = np.asarray(draws)
    return float((d > value).mean() if value >= np.median(d) else (d < value).mean())


def band_moves(rows: list[dict], draws: int, c2_draws: int) -> str:
    """Cases whose in-band status differs between the --draws band and the --c2-draws band, with the share of
    draws beyond the value (the band edge is a step function of few distinct C2 values, so a status can flip on
    an exact equality)."""
    items = []
    for r in rows:
        for key, flag200, flag1000 in BAND_FLAGS:
            if r[flag200] != r[flag1000]:
                items.append(
                    f"{r['task']} {key} {f3(r[key])} {'inside' if r[flag200] else 'outside'} at {draws} draws "
                    f"[{f3(r['c2']['p2_5'])}, {f3(r['c2']['p97_5'])}], {'inside' if r[flag1000] else 'outside'} at "
                    f"{c2_draws} draws [{f3(r['c2_1000']['p2_5'])}, {f3(r['c2_1000']['p97_5'])}]; share of draws "
                    f"beyond it {tail_share(r[key], r['draws']['c2']):.3f} at {draws} and "
                    f"{tail_share(r[key], r['draws']['c2_1000']):.3f} at {c2_draws}"
                )
    return "; ".join(items) or "none"


def most_aligned(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["roster_truth_rho"])


def tie_list(rows: list[dict]) -> str:
    items = []
    for r in rows:
        for t in r["exact_win_ties"]:
            items.append(
                f"{r['task']} {r['arm']}: {' = '.join(short(m) for m in t['models'])} at {t['wins']:g} wins, "
                f"rating gap {t['rating_gap']:.1e}, rho_both {f3(r['rho_both'])} reported, "
                f"{f3(r['rho_both_tie_aware'])} with tied ranks"
            )
    return "; ".join(items) or "none"


def ci95_sentences(S: dict, syn: list[dict], org: list[dict]) -> str:
    """What spearman_ci95 is, from agreement.py, with the reproduction check and the widths."""
    n_shared = sorted({r["n_models_in_truth"] for r in syn + org})
    shared = (
        f"all {n_shared[0]} roster models on every record here"
        if len(n_shared) == 1
        else f"{n_shared[0]} to {n_shared[-1]} models here"
    )
    parts = [
        f"The spearman_ci95 interval in a record's agreement block is a model bootstrap, not a query-level "
        f"uncertainty. mteb_gym.agreement.correlate resamples the shared models ({shared}) with replacement "
        f"{CI95_BOOTSTRAP} times (seed {CI95_SEED}; a resample with fewer than 3 distinct models or an undefined rho is "
        f"dropped), recomputes Spearman between the fixed point-estimate ratings and truth on each resample, and takes "
        f"the 2.5 and 97.5 percentiles, so it measures how much rho depends on which models are in the roster; the "
        f"queries and the verdicts are never resampled, and the judge noise that the controls above measure does not "
        f"enter it."
    ]
    for arm, rows in (("synthetic", syn), ("original", org)):
        if not rows:
            continue
        S_arm = S[arm]
        w = S_arm["spearman_ci95_width"]
        if w["n"]:
            parts.append(
                f"Recomputed with that function from the record ratings and truth.json, it is reproduced on "
                f"{w['n']} of {len(rows)} {arm} records (largest edge difference "
                f"{S_arm['spearman_ci95_max_abs_edge_diff']:.1e}); its width averages {f3(w['mean'])} (smallest "
                f"{f3(w['min'])}), against a largest absolute noise share of {f3(S_arm['max_abs_noise_share'])} "
                f"on that arm."
            )
        else:
            parts.append(f"The {arm} records carry no agreement block, so no interval is reported there.")
    return " ".join(parts)


def gap_se(se: dict) -> float:
    """Upper bound on the Monte Carlo standard error of a per-record C1 minus C1 per pair gap."""
    return float(np.sqrt(se["c1"] ** 2 + se["c1_pair"] ** 2))


def correction_sentence(mt: dict) -> str:
    """Which signed-rank results survive Bonferroni, Holm and Benjamini-Hochberg at the counted number of tests."""
    p_of = {f"{t['arm']} {TEST_LABELS[t['key']]}": t["p"] for t in mt["sorted"]}
    bonf, holm, bh = mt["bonferroni"], mt["holm"], mt["benjamini_hochberg"]
    holm_extra = [n for n in holm if n not in bonf]
    bh_only = [f"{n} (p = {p_of[n]:.3f})" for n in bh if n not in holm]
    none = [f"{n} (p = {p_of[n]:.3f})" for n in p_of if n not in bh]

    def listing(names: list[str]) -> str:
        return ", ".join(names) if names else "nothing"

    return (
        f"At {mt['n_tests']} tests and alpha {mt['alpha']:g}, Bonferroni keeps {listing(bonf)}; Holm adds "
        f"{listing(holm_extra)}; only a Benjamini-Hochberg false-discovery-rate step keeps {listing(bh_only)}; none "
        f"of the three keeps {listing(none)}."
    )


def results_section(out: dict, draws: int, c2_draws: int) -> list[str]:
    """What the controls show, generated from the summaries and the per-record rows."""
    S, syn, org = out["summary"], out["synthetic"], out["original"]
    if not syn or not org:
        return single_arm_section(out, draws, c2_draws)
    mt = S["multiple_testing"]
    bs, bo = S["synthetic"]["bias_share"], S["original"]["bias_share"]
    zs, zo = S["synthetic"]["noise_share"], S["original"]["noise_share"]
    gs, go = S["synthetic"]["c1_minus_c1_pair"], S["original"]["c1_minus_c1_pair"]
    n_s, n_o = len(syn), len(org)
    se_s, se_o = S["synthetic"]["max_mc_se"], S["original"]["max_mc_se"]
    top_s, top_o = most_aligned(syn), most_aligned(org)
    n_ties = S["synthetic"]["n_records_with_win_ties"] + S["original"]["n_records_with_win_ties"]
    n_bal = S["synthetic"]["n_records_balanced"] + S["original"]["n_records_balanced"]
    Ss, So = S["synthetic"], S["original"]
    return [
        "## What the controls show",
        "",
        f"De-biasing accounts for almost all of the rho_both advantage over a fixed-slot single order. The bias share, "
        f"mean(C1) minus the single-order mean, averages {f3(bs['mean'])} on the synthetic arm and {f3(bo['mean'])} on the "
        f"original arm, positive in {bs['n_positive']} of {n_s} and {bo['n_positive']} of {n_o} records (Wilcoxon "
        f"p = {bs['wilcoxon_p']:.3f} and {bo['wilcoxon_p']:.3f}; mean over standard error {bs['mean_over_se']:.1f} and "
        f"{bo['mean_over_se']:.1f}). Anchoring the fixed-slot baseline on the C2 mean instead of on the actual roster and its "
        f"reverse gives mean(C1) minus mean(C2) of {f3(S['synthetic']['c1_minus_c2']['mean'])} and "
        f"{f3(S['original']['c1_minus_c2']['mean'])}, so the conclusion does not rest on the roster. The negative bias shares "
        f"fall on records where rho_both is at or below zero or where the judge shows no first-slot preference (a_first_rate "
        f"below 0.5): synthetic {neg_bias(syn)}; original {neg_bias(org)}. Where the judge does not track truth a higher rho "
        f"is not a less biased rho, so the sign of the bias share carries no bias reading on those records. That grouping "
        f"describes where the negative values fall; it was not fixed in advance.",
        "",
        f"Once the slot is random the second verdict per pair adds little. The noise share, rho_both "
        f"minus mean(C1), averages {f3(zs['mean'])} on the synthetic arm (positive in {zs['n_positive']} of {n_s}, Wilcoxon "
        f"p = {zs['wilcoxon_p']:.2f}, mean over standard error {zs['mean_over_se']:.1f}) and {f3(zo['mean'])} on the original "
        f"arm ({zo['n_positive']} of {n_o}, p = {zo['wilcoxon_p']:.2f}, mean over standard error {zo['mean_over_se']:.1f}). "
        f"No record's absolute noise share exceeds its own C1 draw sd (largest ratio "
        f"{S['synthetic']['max_abs_noise_share_over_c1_sd']:.2f} synthetic, {S['original']['max_abs_noise_share_over_c1_sd']:.2f} "
        f"original), so the per-record signs in that column are not individually meaningful; the arm means are. On the "
        f"synthetic arm the cost of halving the verdicts is small, a tenth of the bias share, and only nominally "
        f"significant; on the original arm nothing is detectable. {ci95_sentences(S, syn, org)}",
        "",
        f"The p values are nominal. {mt['n_tests']} signed-rank tests were run ({len(SIGNED_KEYS)} quantities on two "
        "arms) with no correction, the 14 records in an arm share the roster, the judge and the corpus set, and the two arms "
        f"share the 14 corpora, so the effective sample is smaller than the counts suggest. {correction_sentence(mt)}",
        "",
        f"The actual roster is not special except on {top_s['task']}. Against the {draws}-draw C2 band, rho_order1 lies "
        f"inside the 2.5 to 97.5 band on {Ss['n_records_order1_within_c2_band']} of {n_s} synthetic and "
        f"{So['n_records_order1_within_c2_band']} of {n_o} original records, rho_order2 on "
        f"{Ss['n_records_order2_within_c2_band']} of {n_s} and {So['n_records_order2_within_c2_band']} of {n_o}. Out of band "
        f"at {draws} draws, synthetic: {out_of_band(syn)}; original: {out_of_band(org)}. Against the {c2_draws}-draw band "
        f"(c2_1000, its own random stream), rho_order1 lies inside on {Ss['n_records_order1_within_c2_1000_band']} of {n_s} "
        f"synthetic and {So['n_records_order1_within_c2_1000_band']} of {n_o} original records, rho_order2 on "
        f"{Ss['n_records_order2_within_c2_1000_band']} of {n_s} and {So['n_records_order2_within_c2_1000_band']} of {n_o}. "
        f"Out of band at {c2_draws} draws, synthetic: {out_of_band(syn, 'c2_1000')}; original: "
        f"{out_of_band(org, 'c2_1000')}. Cases whose status differs between the two bands, synthetic: "
        f"{band_moves(syn, draws, c2_draws)}; original: {band_moves(org, draws, c2_draws)}. A 2.5 percent band edge rests "
        f"on about {max(1, draws // 40)} draws at {draws} draws and about {max(1, c2_draws // 40)} at {c2_draws}, so the "
        f"{c2_draws}-draw lists are the ones to read; a status move whose share of draws beyond the value sits near "
        f"0.025 on both bands is an edge equality, not a change. The record whose roster is most aligned with truth (Spearman of "
        f"roster position against official nDCG@10) is {top_s['task']} ({f3(top_s['roster_truth_rho'])}) on the synthetic "
        f"arm and {top_o['task']} ({f3(top_o['roster_truth_rho'])}) on the original arm; a fixed roster-aligned slot "
        f"injects truth into the slot bonus there, which is why its two single-order refits sit far apart "
        f"({f3(top_s['rho_order1'])} and {f3(top_s['rho_order2'])} synthetic, {f3(top_o['rho_order1'])} and "
        f"{f3(top_o['rho_order2'])} original).",
        "",
        f"Monte Carlo error. With {draws} draws the standard error of a control mean is at most {se_s['c1']:.3f} (C1), "
        f"{se_s['c2']:.3f} (C2), {se_s['c3']:.3f} (C3) and {se_s['c1_pair']:.3f} (C1 per pair) per record on the synthetic "
        f"arm and {se_o['c1']:.3f}, {se_o['c2']:.3f}, {se_o['c3']:.3f} and {se_o['c1_pair']:.3f} on the original arm; with "
        f"{c2_draws} draws the C2 standard error is at most {se_s['c2_1000']:.3f} and {se_o['c2_1000']:.3f}. The "
        f"original-arm noise share and both minus C3 are of the same order as these, so their per-record signs are not "
        f"individually meaningful; the bias share is far above them. C1 draws the order per verdict row (one pair on one "
        f"query). C1 per pair draws one order per model pair per draw, so a pair's slot bonus repeats across its queries; "
        f"it averages {f3(Ss['c1_pair_mean']['mean'])} on the synthetic arm and {f3(So['c1_pair_mean']['mean'])} on the "
        f"original arm, and C1 minus C1 per pair averages {f3(gs['mean'])} (range {f3(gs['min'])} to {f3(gs['max'])}, "
        f"positive in {gs['n_positive']} of {n_s}, Wilcoxon p = {gs['wilcoxon_p']:.3f}, mean over standard error "
        f"{gs['mean_over_se']:.1f}) and {f3(go['mean'])} (range {f3(go['min'])} to {f3(go['max'])}, positive in "
        f"{go['n_positive']} of {n_o}, p = {go['wilcoxon_p']:.3f}, mean over standard error {go['mean_over_se']:.1f}). Each "
        f"per-record gap carries a Monte Carlo standard error of at most {gap_se(se_s):.3f} (synthetic) and "
        f"{gap_se(se_o):.3f} (original), so a single record's gap of that size is not individually meaningful. The "
        f"headline bias share depends on that convention; with the per-pair draw it would be "
        f"{f3(Ss['c1_pair_mean']['mean'] - Ss['single_order_mean']['mean'])} on the synthetic arm and "
        f"{f3(So['c1_pair_mean']['mean'] - So['single_order_mean']['mean'])} on the original arm.",
        "",
        f"Practical reading. Judging each pair once in a random per-query order gives rho within {f3(zs['mean'])} "
        f"(synthetic) and {f3(zo['mean'])} (original) of the two-order design at half the judge calls, while fixing the "
        f"order per pair costs a further {f3(gs['mean'])} (synthetic) and, on average and only nominally, {f3(go['mean'])} "
        f"(original, p = {go['wilcoxon_p']:.3f}).",
        "",
        f"Tie-break. The design is balanced in {n_bal} of {n_s + n_o} records, so the Bradley-Terry order is the order of "
        f"total fractional wins. In {n_ties} records two models have exactly equal total wins and the stored rating gap is the "
        f"MM stopping tolerance, which spearmanr scores as a strict rank difference: {tie_list(syn + org)}. Giving tied models "
        f"the mean rating moves the arm mean of rho_both from {f3(S['synthetic']['rho_both']['mean'])} to "
        f"{f3(S['synthetic']['rho_both_tie_aware']['mean'])} (synthetic) and from {f3(S['original']['rho_both']['mean'])} to "
        f"{f3(S['original']['rho_both_tie_aware']['mean'])} (original); the largest per-record shift is "
        f"{f3(S['synthetic']['max_abs_tie_shift'])} synthetic and {f3(S['original']['max_abs_tie_shift'])} original. The "
        f"decomposition above uses the reported rho_both. No other record has a tie.",
        "",
    ]


def single_arm_section(out: dict, draws: int, c2_draws: int) -> list[str]:
    """The results section when only one arm is present (for example a --only run), from that arm's
    summary alone; the two-arm comparison is not written."""
    arm = "synthetic" if out["synthetic"] else "original"
    rows, s, mt = out[arm], out["summary"][arm], out["summary"]["multiple_testing"]
    n, se = len(rows), s["max_mc_se"]
    bs, zs, cs, gs = s["bias_share"], s["noise_share"], s["both_minus_c3"], s["c1_minus_c1_pair"]
    top = most_aligned(rows)
    tests = (
        f" Wilcoxon p = {bs['wilcoxon_p']:.3f} (bias share), {zs['wilcoxon_p']:.3f} (noise share), "
        f"{cs['wilcoxon_p']:.3f} (both minus C3) and {gs['wilcoxon_p']:.3f} (C1 minus C1 per pair); "
        f"{correction_sentence(mt)}"
        if n > 1
        else " Signed-rank tests need more than one record and are not read here."
    )
    return [
        "## What the controls show",
        "",
        f"This run covers the {arm} arm only ({n} record{'s' if n != 1 else ''}), so the two-arm comparison is not "
        f"written and the numbers below are this arm's own.",
        "",
        f"The bias share, mean(C1) minus the single-order mean, averages {f3(bs['mean'])} (range {f3(bs['min'])} to "
        f"{f3(bs['max'])}), positive in {bs['n_positive']} of {n}; mean(C1) minus mean(C2) averages "
        f"{f3(s['c1_minus_c2']['mean'])}. The noise share, rho_both minus mean(C1), averages {f3(zs['mean'])} (range "
        f"{f3(zs['min'])} to {f3(zs['max'])}), positive in {zs['n_positive']} of {n}; rho_both minus mean(C3) averages "
        f"{f3(cs['mean'])} (range {f3(cs['min'])} to {f3(cs['max'])}), positive in {cs['n_positive']} of {n}. The largest "
        f"absolute noise share over its own C1 draw sd is {s['max_abs_noise_share_over_c1_sd']:.2f}. C1 per pair averages "
        f"{f3(s['c1_pair_mean']['mean'])}, so C1 minus C1 per pair averages {f3(gs['mean'])} (range {f3(gs['min'])} to "
        f"{f3(gs['max'])}), positive in {gs['n_positive']} of {n}.{tests} Negative bias shares: {neg_bias(rows)}. "
        f"{ci95_sentences(out['summary'], out['synthetic'], out['original'])}",
        "",
        f"Against the {draws}-draw C2 band, rho_order1 lies inside on {s['n_records_order1_within_c2_band']} of {n} records "
        f"and rho_order2 on {s['n_records_order2_within_c2_band']} of {n}; out of band: {out_of_band(rows)}. Against the "
        f"{c2_draws}-draw band, rho_order1 lies inside on {s['n_records_order1_within_c2_1000_band']} of {n} and rho_order2 "
        f"on {s['n_records_order2_within_c2_1000_band']} of {n}; out of band: {out_of_band(rows, 'c2_1000')}. Cases whose "
        f"status differs between the two bands: {band_moves(rows, draws, c2_draws)}. The record whose roster is most "
        f"aligned with truth is {top['task']} ({f3(top['roster_truth_rho'])}); its single-order refits are "
        f"{f3(top['rho_order1'])} and {f3(top['rho_order2'])}.",
        "",
        f"Monte Carlo error. With {draws} draws the standard error of a control mean is at most {se['c1']:.3f} (C1), "
        f"{se['c2']:.3f} (C2), {se['c3']:.3f} (C3) and {se['c1_pair']:.3f} (C1 per pair) per record; with {c2_draws} draws "
        f"the C2 standard error is at most {se['c2_1000']:.3f}. Each per-record C1 minus C1 per pair gap carries a Monte "
        f"Carlo standard error of at most {gap_se(se):.3f}.",
        "",
        f"Tie-break. The design is balanced in {s['n_records_balanced']} of {n} records. Exact ties in total wins: "
        f"{tie_list(rows)}. Giving tied models the mean rating moves the arm mean of rho_both from "
        f"{f3(s['rho_both']['mean'])} to {f3(s['rho_both_tie_aware']['mean'])}; the largest per-record shift is "
        f"{f3(s['max_abs_tie_shift'])}.",
        "",
    ]


def write_md(out: dict, path: Path, draws: int, seed: int, c2_draws: int) -> None:
    lines = [
        "# Position bias controls (regeneration sweep)",
        "",
        __doc__.strip(),
        "",
        f"Draws per control {draws}, seed {seed}; C2 is drawn again {c2_draws} times (c2_1000) from a separate stream. "
        "Each control cell is mean (sd over draws) [2.5, 97.5 percentiles over draws]. n_q is the number of queries in "
        "the record; C3 uses n_q // 2 of them per draw. The checks column reports the position_bias.json check, whether "
        "the record itself carries a rho to check against, the spearman_ci95 reproduction (model bootstrap), the C2 "
        f"identity and reversal checks, any single-order refit outside the C2 band at {draws} or {c2_draws} draws, and "
        "any exact tie in total wins with the tie-aware rho_both.",
        "",
    ]
    cols = (
        "| task | n_q | rho_both | rho_order1 | rho_order2 | C1 mean (sd) [2.5, 97.5] | "
        "C1 per pair mean (sd) [2.5, 97.5] | C2 mean (sd) [2.5, 97.5] | "
        f"C2 at {c2_draws} mean (sd) [2.5, 97.5] | C3 mean (sd) [2.5, 97.5] | bias share | noise share | "
        "both minus C3 | C1 minus C1 per pair | checks |"
    )
    for arm in ("synthetic", "original"):
        rows = out[arm]
        if not rows:
            continue
        lines += [
            f"## {arm} arm ({len(rows)} records)",
            "",
            cols,
            "|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            chk = [
                "prior " + r["checks"].get("position_bias_json", "absent"),
                "record match" if r["checks"]["rho_both_record"] is not None else "no record rho",
                "ci95 " + r["checks"]["spearman_ci95"],
                "c2 id=o1" if r["c2_identity_equals_order1"] else "c2 id!=o1",
                "c2 rev=o2" if r["c2_reversed_equals_order2"] else "c2 rev!=o2",
            ]
            for key, flag200, flag1000 in BAND_FLAGS:
                if not r[flag200] or not r[flag1000]:
                    where = [str(d) for d, f in ((draws, flag200), (c2_draws, flag1000)) if not r[f]]
                    chk.append(f"{key} outside C2 band at {' and '.join(where)} draws")
            if r["n_pair_files_reversed"]:
                chk.append(f"{r['n_pair_files_reversed']} files reversed")
            for t in r["exact_win_ties"]:
                chk.append(
                    f"win tie {' = '.join(short(m) for m in t['models'])}, tie-aware rho_both {f3(r['rho_both_tie_aware'])}"
                )
            lines.append(
                f"| {r['task']} | {r['n_queries']} | {f3(r['rho_both'])} | {f3(r['rho_order1'])} | {f3(r['rho_order2'])} | "
                f"{band(r['c1'])} | {band(r['c1_pair'])} | {band(r['c2'])} | {band(r['c2_1000'])} | {band(r['c3'])} | "
                f"{f3(r['bias_share'])} | {f3(r['noise_share'])} | {f3(r['both_minus_c3'])} | "
                f"{f3(r['c1_minus_c1_pair'])} | {'; '.join(chk)} |"
            )
        lines += ["", arm_paragraph(arm, out["summary"][arm], len(rows), c2_draws), ""]
    lines += results_section(out, draws, c2_draws)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="run root holding results/ and analysis_out/")
    ap.add_argument("--out-dir", default=None, help="default <root>/analysis_out/controls")
    ap.add_argument(
        "--prior",
        default=None,
        help="position_bias.json to check against; default <root>/analysis_out/position_bias.json",
    )
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--c2-draws", type=int, default=1000, help="draws for the c2_1000 block (own random stream)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="substring filter on record file names (debug)")
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "analysis_out" / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = Path(a.prior) if a.prior else root / "analysis_out" / "position_bias.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior is None:
        print(f"warning: no prior at {prior_path}; skipping the position_bias.json check", flush=True)
    recs = sorted((root / "results" / "records").glob("*.json"))
    if a.only:
        recs = [p for p in recs if a.only in p.name]
    if not recs:
        raise SystemExit("no records")
    t0 = time.time()
    jobs = [(p, root, a.draws, a.seed, prior, a.c2_draws) for p in recs]
    with Pool(a.workers) as pool:
        results = pool.map(analyse, jobs, chunksize=1)
    out = {"synthetic": [], "original": []}
    for r in results:
        out[r["arm"]].append(r)
        tie = f" ties={len(r['exact_win_ties'])} tie_aware={f3(r['rho_both_tie_aware'])}" if r["exact_win_ties"] else ""
        print(
            f"{r['arm']:9s} {r['task']:28s} both={f3(r['rho_both'])} o1={f3(r['rho_order1'])} o2={f3(r['rho_order2'])} "
            f"C1={band(r['c1'])} C1pair={band(r['c1_pair'])} C2={band(r['c2'])} C2k={band(r['c2_1000'])} "
            f"C3={band(r['c3'])} bias={f3(r['bias_share'])} noise={f3(r['noise_share'])} "
            f"both-C3={f3(r['both_minus_c3'])} C1-C1pair={f3(r['c1_minus_c1_pair'])} "
            f"o1_in_c2={int(r['order1_in_c2_band'])}/{int(r['order1_in_c2_1000_band'])} "
            f"o2_in_c2={int(r['order2_in_c2_band'])}/{int(r['order2_in_c2_1000_band'])} "
            f"ci95={r['checks']['spearman_ci95']}{tie} {r['seconds']:.0f}s",
            flush=True,
        )
    out["summary"] = {arm: summarise(rows, a.draws, a.c2_draws) for arm, rows in out.items() if rows}
    out["summary"]["multiple_testing"] = multiple_testing(out["summary"])
    out["settings"] = {
        "draws": a.draws,
        "c2_draws": a.c2_draws,
        "seed": a.seed,
        "n_records": len(results),
        "prior_checked": prior is not None,
        "ci95_bootstrap": CI95_BOOTSTRAP,
        "ci95_seed": CI95_SEED,
        "wall_seconds": time.time() - t0,
    }
    out["definitions"] = __doc__
    (out_dir / "position_bias_controls.json").write_text(json.dumps(out, indent=1))
    write_md(out, out_dir / "position_bias_controls.md", a.draws, a.seed, a.c2_draws)
    print(f"wrote {out_dir / 'position_bias_controls.json'} and .md in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
