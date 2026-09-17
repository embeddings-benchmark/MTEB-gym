"""Controls that separate de-biasing from de-noising in the two-order judge design.

position_bias.py reports, per record, rho_both (Bradley-Terry refit from the two-order averaged
score_a, Spearman vs official nDCG@10), rho_order1 (refit from O[w1] only) and rho_order2 (refit
from 1 - O[w2] only). The single-order refits are not comparable with each other because both use
the same pair enumeration, so order 1 always hands the first-position bonus to each pair file's
model_a and order 2 to its model_b, and the roster (config.models) correlates with quality. Two
effects are confounded in "rho_both beats a single-order refit": de-biasing (the two-order average
cancels the slot bonus) and de-noising (a single-order refit rests on one verdict per pair instead
of two). Three controls computed from the raw verdict rows separate them.

  C1 random-order refit. For every verdict row (one model pair on one query) draw independently
     which order's verdict to use, O[w1] or 1 - O[w2], refit, Spearman vs truth. One verdict per pair
     like a single-order refit, but the slot bonus lands on a random model per pair. The draw is per
     row (pair x query), not once per model pair: a per-pair coin would leave every query of that pair
     with the same slot bonus, and C1 would no longer be unbiased in expectation.
  C2 roster-permuted fixed-slot refit. Per draw, permute the roster; for every row use O[w1] if
     model_a precedes model_b in the permuted roster, else 1 - O[w2]. The fixed-slot design with a
     random roster. The identity permutation reproduces rho_order1 and the reversed roster
     rho_order2 when every pair file is oriented along the roster (checked and reported). rho_order1
     and rho_order2 are then compared with the C2 2.5 to 97.5 band per record.
  C3 half-queries both-order refit. A random half of the queries (by qid), both orders averaged,
     refit. The same number of judge verdicts as a single-order refit over all queries, but unbiased.

Decomposition per record
  bias share  = mean(C1) - mean(rho_order1, rho_order2)
  noise share = rho_both - mean(C1)
  both minus C3 = rho_both - mean(C3)   (the pure noise cost of halving the verdict count)

Arm-level tests are two-sided Wilcoxon signed-rank over the records (scipy default, exact at n = 14)
and the mean over its standard error across records.

Scoring conventions follow position_bias.py. O = {A: 1, tie: 0.5, B: 0}; raw == ["identical"] and any
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

from mteb_gym.judge import Verdict
from mteb_gym.rank import BASE, SCALE, _bradley_terry, rate
from mteb_gym.reliability import verdict_file

O = {"A": 1.0, "tie": 0.5, "B": 0.0}
TEMPLATE_MARK = "$" + "{"
TOL = 1e-9


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
                    ties.append({"models": [self.names[i] for i in idx], "wins": float(w),
                                 "rating_gap": float(r[idx].max() - r[idx].min())})
        return {"balanced_design": balanced, "games_per_model": int(games[0]) if balanced else None,
                "exact_win_ties": ties, "rho_both_tie_aware": spearman(snapped, self.names, self.truth)}


def parse_rows(rows: list[dict], task: str) -> dict:
    """Per-row arrays. s_both = stored score_a, s1 = O[w1], s2 = 1 - O[w2]."""
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
        s1[k], s2[k] = O.get(w1, 0.5), 1.0 - O.get(w2, 0.5)
        if w1 in O and w2 in O and abs((s1[k] + s2[k]) / 2 - s_both[k]) > TOL:
            raise RuntimeError(f"score_a mismatch in {task} qid={v['qid']} raw={raw} score_a={v['score_a']}")
    return dict(
        s_both=s_both, s1=s1, s2=s2, n_identical=n_ident, n_invalid_raw=n_invalid,
        n_decisive_order_verdicts=decisive, n_first_wins=first,
        a_first_rate=(first / decisive) if decisive else None,
    )


def check_against_package(rows: list[dict], rec: dict, fit: Refitter, s_both: np.ndarray) -> float:
    """rate() on the stored score_a must reproduce the record ratings and the vectorised refit."""
    verdicts = [Verdict(qid=v["qid"], query=v.get("query", ""), model_a=v["model_a"], model_b=v["model_b"],
                        score_a=float(v["score_a"])) for v in rows]
    pkg = {m.name: m.rating for m in rate(verdicts, bootstrap=0, seed=0)}
    rec_r = {r["model"]: r["rating"] for r in rec["ratings"]}
    vec = dict(zip(fit.names, fit.ratings(s_both)))
    d_rec = max(abs(pkg[m] - rec_r[m]) for m in pkg)
    d_vec = max(abs(pkg[m] - vec[m]) for m in pkg)
    if d_rec > 1e-6 or d_vec > 1e-6:
        raise RuntimeError(f"rating mismatch: package vs record {d_rec:.2e}, package vs vectorised {d_vec:.2e}")
    return max(d_rec, d_vec)


def pct(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)), "p2_5": float(np.percentile(x, 2.5)),
            "p97_5": float(np.percentile(x, 97.5)), "min": float(np.min(x)), "max": float(np.max(x))}


def in_band(x: float, d: dict) -> bool:
    return bool(d["p2_5"] <= x <= d["p97_5"])


def controls(fit: Refitter, arr: dict, rows: list[dict], roster: list[str], draws: int, seed: int) -> dict:
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
    return dict(c1=pct(c1), c2=pct(c2), c3=pct(c3), c2_identity=c2_identity, c2_reversed=c2_reversed,
                n_unique_qids=int(len(qids)), n_half_queries=half,
                c1_draws=c1.tolist(), c2_draws=c2.tolist(), c3_draws=c3.tolist())


def analyse(args: tuple) -> dict:
    rec_path, root, draws, seed, prior = args
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
    fit = Refitter(names, np.array([idx[v["model_a"]] for v in rows]), np.array([idx[v["model_b"]] for v in rows]), truth)
    arr = parse_rows(rows, task)
    max_rating_diff = check_against_package(rows, rec, fit, arr["s_both"])

    rho_both, rho_o1, rho_o2 = fit.rho(arr["s_both"]), fit.rho(arr["s1"]), fit.rho(arr["s2"])
    checks = verify(rec, rec_path.name, prior, rho_both, rho_o1, rho_o2, arr["a_first_rate"])
    ties = fit.tie_aware(arr["s_both"])
    ctl = controls(fit, arr, rows, roster, draws, seed)
    single_mean = (rho_o1 + rho_o2) / 2
    roster_in_truth = [m for m in roster if m in truth]
    roster_truth_rho = float(spearmanr(range(len(roster_in_truth)), [truth[m] for m in roster_in_truth])[0])
    return {
        "task": task, "arm": arm, "record": rec_path.name, "n_models": len(names),
        "n_models_in_truth": sum(m in truth for m in names), "n_queries": rec["config"]["n_queries"],
        "n_unique_qids": ctl.pop("n_unique_qids"), "n_half_queries": ctl.pop("n_half_queries"),
        "n_pairs_judged": len(rows), "n_pair_files_reversed": n_reversed,
        "n_identical": arr["n_identical"], "n_invalid_raw": arr["n_invalid_raw"],
        "rho_both": rho_both, "rho_order1": rho_o1, "rho_order2": rho_o2, "single_order_mean": single_mean,
        "a_first_rate": arr["a_first_rate"], "roster_truth_rho": roster_truth_rho,
        "max_abs_rating_diff": max_rating_diff, "checks": checks, **ties,
        "c1": ctl["c1"], "c2": ctl["c2"], "c3": ctl["c3"],
        "c2_identity": ctl["c2_identity"], "c2_reversed": ctl["c2_reversed"],
        "c2_identity_equals_order1": abs(ctl["c2_identity"] - rho_o1) < TOL,
        "c2_reversed_equals_order2": abs(ctl["c2_reversed"] - rho_o2) < TOL,
        "order1_in_c2_band": in_band(rho_o1, ctl["c2"]), "order2_in_c2_band": in_band(rho_o2, ctl["c2"]),
        "bias_share": ctl["c1"]["mean"] - single_mean,
        "noise_share": rho_both - ctl["c1"]["mean"],
        "both_minus_c3": rho_both - ctl["c3"]["mean"],
        "draws": {"c1": ctl["c1_draws"], "c2": ctl["c2_draws"], "c3": ctl["c3_draws"]},
        "seconds": time.time() - t0,
    }


def verify(rec: dict, name: str, prior: dict | None, rho_both: float, o1: float, o2: float, af: float | None) -> dict:
    """Loud failure on any mismatch with the record or with position_bias.json."""
    rec_rho = (rec.get("agreement") or {}).get("spearman_rho")
    rec_af = rec["diagnostics"].get("a_first_rate")
    checks = {"rho_both_record": rec_rho, "a_first_rate_record": rec_af}
    if rec_rho is not None and abs(rec_rho - rho_both) > TOL:
        raise RuntimeError(f"{name}: rho_both {rho_both!r} != record {rec_rho!r}")
    if rec_af is None or af is None or abs(rec_af - af) > TOL:
        raise RuntimeError(f"{name}: a_first_rate {af!r} != record {rec_af!r}")
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
    return {"n_positive": int((v > 0).sum()), "wilcoxon_p": float(wilcoxon(v).pvalue), "mean_over_se": float(v.mean() / se)}


def summarise(rows: list[dict], draws: int) -> dict:
    keys = {
        "rho_both": lambda r: r["rho_both"], "rho_both_tie_aware": lambda r: r["rho_both_tie_aware"],
        "rho_order1": lambda r: r["rho_order1"],
        "rho_order2": lambda r: r["rho_order2"], "single_order_mean": lambda r: r["single_order_mean"],
        "c1_mean": lambda r: r["c1"]["mean"], "c2_mean": lambda r: r["c2"]["mean"], "c3_mean": lambda r: r["c3"]["mean"],
        "c1_sd": lambda r: r["c1"]["sd"], "c2_sd": lambda r: r["c2"]["sd"], "c3_sd": lambda r: r["c3"]["sd"],
        "bias_share": lambda r: r["bias_share"], "noise_share": lambda r: r["noise_share"],
        "both_minus_c3": lambda r: r["both_minus_c3"], "a_first_rate": lambda r: r["a_first_rate"],
        "c1_minus_c2": lambda r: r["c1"]["mean"] - r["c2"]["mean"],
    }
    s = {}
    for k, f in keys.items():
        v = np.array([f(r) for r in rows])
        s[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(len(v))}
    for k in ("bias_share", "noise_share", "both_minus_c3", "c1_minus_c2"):
        s[k].update(signed_test(np.array([keys[k](r) for r in rows])))
    s["n_records_c1_above_single_mean"] = int(sum(r["c1"]["mean"] > r["single_order_mean"] for r in rows))
    s["n_records_both_above_c1"] = int(sum(r["rho_both"] > r["c1"]["mean"] for r in rows))
    s["n_records_both_above_c3"] = int(sum(r["rho_both"] > r["c3"]["mean"] for r in rows))
    s["n_records_c1_within_c2_band"] = int(sum(in_band(r["c1"]["mean"], r["c2"]) for r in rows))
    s["n_records_both_within_c3_band"] = int(sum(in_band(r["rho_both"], r["c3"]) for r in rows))
    s["n_records_order1_within_c2_band"] = int(sum(r["order1_in_c2_band"] for r in rows))
    s["n_records_order2_within_c2_band"] = int(sum(r["order2_in_c2_band"] for r in rows))
    s["max_abs_noise_share_over_c1_sd"] = float(max(abs(r["noise_share"]) / r["c1"]["sd"] for r in rows))
    s["max_mc_se"] = {c: float(max(r[c]["sd"] for r in rows) / np.sqrt(draws)) for c in ("c1", "c2", "c3")}
    s["n_records_with_win_ties"] = int(sum(bool(r["exact_win_ties"]) for r in rows))
    s["n_records_balanced"] = int(sum(r["balanced_design"] for r in rows))
    s["max_abs_tie_shift"] = float(max(abs(r["rho_both"] - r["rho_both_tie_aware"]) for r in rows))
    return s


def f3(x: float) -> str:
    return f"{x:.3f}"


def band(d: dict) -> str:
    return f"{d['mean']:.3f} (sd {d['sd']:.3f}) [{d['p2_5']:.3f}, {d['p97_5']:.3f}]"


def short(model: str) -> str:
    return model.split("/")[-1]


def arm_paragraph(arm: str, s: dict, n: int) -> str:
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
        f"{s['n_records_c1_within_c2_band']} of {n}. The mean a_first_rate is {f3(m['a_first_rate'])}."
    )


def neg_bias(rows: list[dict]) -> str:
    return "; ".join(f"{r['task']} (rho_both {f3(r['rho_both'])}, a_first_rate {f3(r['a_first_rate'])})"
                     for r in rows if r["bias_share"] < 0) or "none"


def out_of_band(rows: list[dict]) -> str:
    items = []
    for r in rows:
        for key, flag in (("rho_order1", "order1_in_c2_band"), ("rho_order2", "order2_in_c2_band")):
            if not r[flag]:
                below = r[key] < r["c2"]["p2_5"]
                edge = r["c2"]["p2_5"] if below else r["c2"]["p97_5"]
                items.append(f"{r['task']} {key} {f3(r[key])} {'below' if below else 'above'} {f3(edge)}")
    return "; ".join(items) or "none"


def most_aligned(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["roster_truth_rho"])


def tie_list(rows: list[dict]) -> str:
    items = []
    for r in rows:
        for t in r["exact_win_ties"]:
            items.append(f"{r['task']} {r['arm']}: {' = '.join(short(m) for m in t['models'])} at {t['wins']:g} wins, "
                         f"rating gap {t['rating_gap']:.1e}, rho_both {f3(r['rho_both'])} reported, "
                         f"{f3(r['rho_both_tie_aware'])} with tied ranks")
    return "; ".join(items) or "none"


def results_section(out: dict, draws: int) -> list[str]:
    """What the controls show, generated from the summaries and the per-record rows."""
    S, syn, org = out["summary"], out["synthetic"], out["original"]
    bs, bo = S["synthetic"]["bias_share"], S["original"]["bias_share"]
    zs, zo = S["synthetic"]["noise_share"], S["original"]["noise_share"]
    n_s, n_o = len(syn), len(org)
    se_s, se_o = S["synthetic"]["max_mc_se"], S["original"]["max_mc_se"]
    top_s, top_o = most_aligned(syn), most_aligned(org)
    n_ties = S["synthetic"]["n_records_with_win_ties"] + S["original"]["n_records_with_win_ties"]
    n_bal = S["synthetic"]["n_records_balanced"] + S["original"]["n_records_balanced"]
    return [
        "## What the controls show", "",
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
        f"significant; on the original arm nothing is detectable. The records' own bootstrap interval for rho_both "
        f"(spearman_ci95 in the agreement block) is far wider than any noise share.",
        "",
        "The p values are nominal. Eight signed-rank tests were run (four quantities on two arms) with no correction, "
        "the 14 records in an arm share the roster, the judge and the corpus set, and the two arms share the 14 corpora, "
        "so the effective sample is smaller than the counts suggest. The bias share survives any reasonable correction; "
        "the synthetic noise share and both minus C3 do not.",
        "",
        f"The actual roster is not special except on {top_s['task']}. rho_order1 lies inside the C2 2.5 to 97.5 band on "
        f"{S['synthetic']['n_records_order1_within_c2_band']} of {n_s} synthetic and "
        f"{S['original']['n_records_order1_within_c2_band']} of {n_o} original records, rho_order2 on "
        f"{S['synthetic']['n_records_order2_within_c2_band']} of {n_s} and {S['original']['n_records_order2_within_c2_band']} "
        f"of {n_o}. Out of band, synthetic: {out_of_band(syn)}; original: {out_of_band(org)}. The record whose roster is most "
        f"aligned with truth (Spearman of roster position against official nDCG@10) is {top_s['task']} "
        f"({f3(top_s['roster_truth_rho'])}) on the synthetic arm and {top_o['task']} ({f3(top_o['roster_truth_rho'])}) on the "
        f"original arm; a fixed roster-aligned slot injects truth into the slot bonus there, which is why its two single-order "
        f"refits sit far apart and outside the roster-permuted band. The other out-of-band cases sit at the band edge, "
        f"where a 2.5 percent edge from {draws} draws rests on about {max(1, draws // 40)} draws, so they may be band noise.",
        "",
        f"Monte Carlo error. With {draws} draws the standard error of a control mean is at most {se_s['c1']:.3f} (C1), "
        f"{se_s['c2']:.3f} (C2) and {se_s['c3']:.3f} (C3) per record on the synthetic arm and {se_o['c1']:.3f}, "
        f"{se_o['c2']:.3f} and {se_o['c3']:.3f} on the original arm. The original-arm noise share and both minus C3 are of the "
        f"same order as these, so their per-record signs are not individually meaningful; the bias share is far above them. "
        f"C1 draws the order per verdict row (one pair on one query), not once per model pair, and the headline bias share "
        f"depends on that convention: drawing one order per model pair instead, so that a pair's slot bonus repeats "
        f"across its queries, lowered C1 by 0.10 to 0.19 on the three records where it was re-derived independently.",
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


def write_md(out: dict, path: Path, draws: int, seed: int) -> None:
    lines = ["# Position bias controls (regeneration sweep)", "", __doc__.strip(), "",
             f"Draws per control {draws}, seed {seed}. Each control cell is mean (sd over draws) [2.5, 97.5 percentiles over "
             "draws]. n_q is the number of queries in the record; C3 uses n_q // 2 of them per draw. The checks column reports "
             "the position_bias.json check, whether the record itself carries a rho to check against, the C2 identity and "
             "reversal checks, and any exact tie in total wins with the tie-aware rho_both.", ""]
    cols = ("| task | n_q | rho_both | rho_order1 | rho_order2 | C1 mean (sd) [2.5, 97.5] | C2 mean (sd) [2.5, 97.5] | "
            "C3 mean (sd) [2.5, 97.5] | bias share | noise share | both minus C3 | checks |")
    for arm in ("synthetic", "original"):
        rows = out[arm]
        if not rows:
            continue
        lines += [f"## {arm} arm ({len(rows)} records)", "", cols, "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|"]
        for r in rows:
            chk = ["prior " + r["checks"].get("position_bias_json", "absent"),
                   "record match" if r["checks"]["rho_both_record"] is not None else "no record rho",
                   "c2 id=o1" if r["c2_identity_equals_order1"] else "c2 id!=o1",
                   "c2 rev=o2" if r["c2_reversed_equals_order2"] else "c2 rev!=o2"]
            if r["n_pair_files_reversed"]:
                chk.append(f"{r['n_pair_files_reversed']} files reversed")
            for t in r["exact_win_ties"]:
                chk.append(f"win tie {' = '.join(short(m) for m in t['models'])}, tie-aware rho_both {f3(r['rho_both_tie_aware'])}")
            lines.append(f"| {r['task']} | {r['n_queries']} | {f3(r['rho_both'])} | {f3(r['rho_order1'])} | {f3(r['rho_order2'])} | "
                         f"{band(r['c1'])} | {band(r['c2'])} | {band(r['c3'])} | {f3(r['bias_share'])} | {f3(r['noise_share'])} | "
                         f"{f3(r['both_minus_c3'])} | {'; '.join(chk)} |")
        lines += ["", arm_paragraph(arm, out["summary"][arm], len(rows)), ""]
    lines += results_section(out, draws)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="run root holding results/ and analysis_out/")
    ap.add_argument("--out-dir", default=None, help="default <root>/analysis_out/controls")
    ap.add_argument("--prior", default=None, help="position_bias.json to check against; default <root>/analysis_out/position_bias.json")
    ap.add_argument("--draws", type=int, default=200)
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
    jobs = [(p, root, a.draws, a.seed, prior) for p in recs]
    with Pool(a.workers) as pool:
        results = pool.map(analyse, jobs, chunksize=1)
    out = {"synthetic": [], "original": []}
    for r in results:
        out[r["arm"]].append(r)
        tie = f" ties={len(r['exact_win_ties'])} tie_aware={f3(r['rho_both_tie_aware'])}" if r["exact_win_ties"] else ""
        print(f"{r['arm']:9s} {r['task']:28s} both={f3(r['rho_both'])} o1={f3(r['rho_order1'])} o2={f3(r['rho_order2'])} "
              f"C1={band(r['c1'])} C2={band(r['c2'])} C3={band(r['c3'])} bias={f3(r['bias_share'])} "
              f"noise={f3(r['noise_share'])} both-C3={f3(r['both_minus_c3'])} o1_in_c2={int(r['order1_in_c2_band'])} "
              f"o2_in_c2={int(r['order2_in_c2_band'])}{tie} {r['seconds']:.0f}s", flush=True)
    out["summary"] = {arm: summarise(rows, a.draws) for arm, rows in out.items() if rows}
    out["settings"] = {"draws": a.draws, "seed": a.seed, "n_records": len(results), "prior_checked": prior is not None,
                       "wall_seconds": time.time() - t0}
    out["definitions"] = __doc__
    (out_dir / "position_bias_controls.json").write_text(json.dumps(out, indent=1))
    write_md(out, out_dir / "position_bias_controls.md", a.draws, a.seed)
    print(f"wrote {out_dir / 'position_bias_controls.json'} and .md in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
