"""Position-bias analysis of the judge over the regeneration sweep.

For every record under results/records/ (synthetic and original arms):
  rho_both   : Spearman(truth, ratings refit from the stored score_a)      [check vs record agreement.spearman_rho]
  rho_order1 : Spearman(truth, ratings refit from score_a = O[w1])          (model_a shown first)
  rho_order2 : Spearman(truth, ratings refit from score_a = 1 - O[w2])      (model_b shown first)
  split_rate : among pairs decisive in BOTH orders, fraction where the two orders favour different MODELS
               (w1 == w2 as letters: the judge picked the same presented slot both times)
  a_first_rate : over all decisive single-order verdicts, fraction won by the model shown first
               [check vs record diagnostics.a_first_rate]
Identical result sets (raw == ["identical"]) count as ties (0.5) for every rating refit and are excluded
from the order-specific counts. A raw token outside {A, B, tie, identical} is scored 0.5 in the
order-specific refits and excluded from decisive counts, and every verdict without exactly two tokens is
counted once in n_invalid_raw. The package coerces unparseable judge answers before it writes the
verdict file, so n_invalid_raw is 0 on this sweep even though the records report parse failures
(diagnostics.parse_failure_rate, 61 calls of 210,140); read that field for the parse-failure count.

How to read rho_order1 against rho_order2: both refits use the same pairs, so order 1 always hands the
first-position bonus to each pair file's model_a and order 2 always to its model_b. The roster is not
random with respect to quality (later entrants are better on average), so the two single-order refits
are biased in opposite directions along the roster and their gap measures the roster, not the judge.
Compare each of them with rho_both, or use their mean; do not compare them with each other. Part of
the rho_both advantage is de-noising (one verdict per pair against two) rather than de-biasing.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from mteb_gym.reliability import verdict_file
from mteb_gym.rank import rate
from mteb_gym.judge import Verdict

ROOT = Path("/data/home/niklas/tejas/tmp_regen")
OUT = ROOT / "results"
AN = ROOT / "analysis_out"
O = {"A": 1.0, "tie": 0.5, "B": 0.0}
TEMPLATE_MARK = "$" + "{"


def load_rows(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    p2 = path.with_suffix(".jsonl")
    if p2.exists():
        return [json.loads(l) for l in p2.read_text().splitlines() if l.strip()]
    raise FileNotFoundError(str(path))


def pair_rows(rec, models):
    rows, found = [], []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            p = verdict_file(OUT, rec, a, b)
            if p.exists() or p.with_suffix(".jsonl").exists():
                rows.extend(load_rows(p)); found.append(str(p.name))
            else:
                p = verdict_file(OUT, rec, b, a)
                rows.extend(load_rows(p)); found.append(str(p.name))
    return rows, found


def rho(ratings_list, truth):
    r = {m.name: m.rating for m in ratings_list}
    shared = [m for m in r if m in truth]
    if len(shared) < 3:
        return None, len(shared)
    g = np.array([r[m] for m in shared]); t = np.array([truth[m] for m in shared])
    return float(spearmanr(g, t)[0]), len(shared)


def analyse(rec_path: Path):
    rec = json.loads(rec_path.read_text())
    if TEMPLATE_MARK in rec_path.name:
        raise RuntimeError(f"template-looking record name: {rec_path.name}")
    task = rec["task_name"]
    truth_path = AN / task / "truth.json"
    truth_text = truth_path.read_text()
    if TEMPLATE_MARK in truth_text:
        raise RuntimeError(f"template-looking text in {truth_path}")
    truth = json.loads(truth_text)
    models = [r["model"] for r in rec["ratings"]]
    rows, files = pair_rows(rec, models)

    both, o1, o2 = [], [], []
    tok = Counter(); n_ident = 0; n_both_decisive = 0; n_split = 0
    first = decisive = 0; n_invalid = 0
    for v in rows:
        raw = v["raw"]
        base = dict(qid=v["qid"], query=v.get("query", ""), model_a=v["model_a"], model_b=v["model_b"])
        both.append(Verdict(score_a=float(v["score_a"]), **base))
        if raw == ["identical"]:
            n_ident += 1
            o1.append(Verdict(score_a=0.5, **base)); o2.append(Verdict(score_a=0.5, **base))
            continue
        if len(raw) != 2:
            n_invalid += 1; tok[f"len{len(raw)}"] += 1
            o1.append(Verdict(score_a=0.5, **base)); o2.append(Verdict(score_a=0.5, **base))
            continue
        w1, w2 = raw
        for w in raw:
            tok[w] += 1
            if w in ("A", "B"):
                decisive += 1; first += (w == "A")
            elif w != "tie":
                n_invalid += 1
        s1 = O.get(w1, 0.5); s2 = 1.0 - O.get(w2, 0.5)
        # sanity: stored score_a must equal the two-order average
        if w1 in O and w2 in O and abs((s1 + s2) / 2 - float(v["score_a"])) > 1e-9:
            raise RuntimeError(f"score_a mismatch in {task} qid={v['qid']} raw={raw} score_a={v['score_a']}")
        o1.append(Verdict(score_a=s1, **base)); o2.append(Verdict(score_a=s2, **base))
        if w1 in ("A", "B") and w2 in ("A", "B"):
            n_both_decisive += 1
            if w1 == w2:  # same presented slot both times -> different models favoured
                n_split += 1

    r_both = rate(both, bootstrap=0, seed=0)
    r_o1 = rate(o1, bootstrap=0, seed=0)
    r_o2 = rate(o2, bootstrap=0, seed=0)
    rec_r = {r["model"]: r["rating"] for r in rec["ratings"]}
    max_rating_diff = max(abs(m.rating - rec_r[m.name]) for m in r_both)
    rho_both, n_shared = rho(r_both, truth)
    rho_o1, _ = rho(r_o1, truth)
    rho_o2, _ = rho(r_o2, truth)
    a_first = first / decisive if decisive else None
    rec_rho = (rec.get("agreement") or {}).get("spearman_rho")
    rec_af = rec["diagnostics"].get("a_first_rate")
    return {
        "task": task,
        "arm": rec["config"]["arm"],
        "record": rec_path.name,
        "n_models": len(models),
        "n_models_in_truth": n_shared,
        "n_queries": rec["config"]["n_queries"],
        "n_pairs_judged": len(rows),
        "n_pair_files": len(files),
        "n_identical": n_ident,
        "n_invalid_raw": n_invalid,
        "raw_token_counts": dict(tok),
        "rho_both": rho_both,
        "rho_both_record": rec_rho,
        "rho_both_matches_record": (None if rec_rho is None or rho_both is None else abs(rho_both - rec_rho) < 1e-9),
        "max_abs_rating_diff_vs_record": max_rating_diff,
        "rho_order1": rho_o1,
        "rho_order2": rho_o2,
        "n_both_decisive": n_both_decisive,
        "n_split": n_split,
        "split_rate": (n_split / n_both_decisive) if n_both_decisive else None,
        "n_decisive_order_verdicts": decisive,
        "n_first_wins": first,
        "a_first_rate": a_first,
        "a_first_rate_record": rec_af,
        "a_first_rate_matches_record": (None if rec_af is None or a_first is None else abs(a_first - rec_af) < 1e-12),
        "ratings_both": {m.name: m.rating for m in r_both},
        "ratings_order1": {m.name: m.rating for m in r_o1},
        "ratings_order2": {m.name: m.rating for m in r_o2},
        "truth": truth,
    }


def fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def main():
    recs = sorted((OUT / "records").glob("*.json"))
    out = {"synthetic": [], "original": []}
    for p in recs:
        row = analyse(p)
        out[row["arm"]].append(row)
        print(f"{row['arm']:9s} {row['task']:28s} rho_both={fmt(row['rho_both'])} (rec {fmt(row['rho_both_record'])}) "
              f"o1={fmt(row['rho_order1'])} o2={fmt(row['rho_order2'])} split={fmt(row['split_rate'])} "
              f"a_first={fmt(row['a_first_rate'])} (rec {fmt(row['a_first_rate_record'])}) "
              f"maxdiff={row['max_abs_rating_diff_vs_record']:.2e} invalid={row['n_invalid_raw']}", flush=True)

    def summ(rows):
        keys = ["rho_both", "rho_order1", "rho_order2", "split_rate", "a_first_rate"]
        cols = {k: [r[k] for r in rows if r[k] is not None] for k in keys}
        return {k: {"mean": float(np.mean(v)) if v else None,
                    "median": float(np.median(v)) if v else None,
                    "n": len(v)} for k, v in cols.items()}
    out["summary"] = {arm: summ(rows) for arm, rows in out.items() if rows}
    out["definitions"] = __doc__
    (AN / "position_bias.json").write_text(json.dumps(out, indent=1))

    lines = ["# Position bias of the judge (regeneration sweep)", "", __doc__.strip(), ""]
    for arm in ("synthetic", "original"):
        rows = out[arm]
        lines += [f"## {arm} arm ({len(rows)} records)", "",
                  "| task | n_q | n_pairs | rho_both | rec rho | rho_order1 | rho_order2 | split_rate (n_split/n_both_decisive) | a_first_rate | rec a_first | checks |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
        for r in rows:
            chk = []
            chk.append("rho=rec" if r["rho_both_matches_record"] else ("rho no-record" if r["rho_both_matches_record"] is None else "RHO MISMATCH"))
            chk.append("af=rec" if r["a_first_rate_matches_record"] else "AF MISMATCH")
            chk.append(f"ratings maxdiff {r['max_abs_rating_diff_vs_record']:.1e}")
            if r["n_invalid_raw"]:
                chk.append(f"invalid raw {r['n_invalid_raw']}")
            lines.append(f"| {r['task']} | {r['n_queries']} | {r['n_pairs_judged']} | {fmt(r['rho_both'])} | {fmt(r['rho_both_record'])} | "
                         f"{fmt(r['rho_order1'])} | {fmt(r['rho_order2'])} | {fmt(r['split_rate'])} ({r['n_split']}/{r['n_both_decisive']}) | "
                         f"{fmt(r['a_first_rate'])} | {fmt(r['a_first_rate_record'])} | {'; '.join(chk)} |")
        s = out["summary"][arm]
        lines += ["", "mean / median over records: " + ", ".join(f"{k} {s[k]['mean']:.3f} / {s[k]['median']:.3f}" for k in s), ""]
    (AN / "position_bias.md").write_text("\n".join(lines) + "\n")
    print("wrote", AN / "position_bias.json", AN / "position_bias.md")


if __name__ == "__main__":
    main()
