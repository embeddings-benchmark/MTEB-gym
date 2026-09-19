"""Level check: does the rewritten package reproduce the paper's per-corpus level on the nano tier?

Reads the regen run's SUMMARY.jsonl (one row per corpus and arm) and the paper's tables in
results.tex, and prints, per corpus: paper rho (full-scale, tab:agreement) vs regen rho (nano,
synthetic arm, Result.agreement), and paper kappa (tab:humanbase) vs regen S (original arm,
judge_reliability). Different rosters, corpus sizes, judge prompt and generator, so this is a
level check, not a replication; the script says so in its header.

Usage: python level_check.py <SUMMARY.jsonl> <results.tex> [--md out.md]
"""

import json
import re
import sys
from pathlib import Path

NANO_TO_PAPER = {
    "NanoNFCorpusRetrieval": "NFCorpus",
    "NanoSciFactRetrieval": "SciFact",
    "NanoFiQA2018Retrieval": "FiQA-2018",
    "NanoArguAnaRetrieval": "ArguAna",
    "NanoSCIDOCSRetrieval": "SCIDOCS",
    "NanoQuoraRetrieval": "Quora",
    "NanoDBPediaRetrieval": "DBPedia-entity",
    "NanoHotpotQARetrieval": "HotpotQA",
    "NanoFEVERRetrieval": "FEVER",
    "NanoClimateFeverRetrieval": "Climate-FEVER",
    "NanoNQRetrieval": "NQ",
    "NanoTouche2020Retrieval": "Touch\\'e-2020",
    "NanoMSMARCORetrieval": None,  # not in the paper's tables
    "NFCorpus": "NFCorpus",  # the full corpus: same row, same scale as the paper
}


def paper_tables(tex: str) -> tuple[dict, dict]:
    """{corpus: (rho, n, p)} from tab:agreement rows and {corpus: (kappa, n)} from tab:humanbase rows."""
    rho, kappa = {}, {}
    row = re.compile(r"^([A-Za-z0-9\\' +\-]+?)\s*&\s*(\d+)\s*&\s*\$?(-?)\$?([0-9.]+)\s*&\s*\[.*?\]\s*&\s*\$?(-?)\$?[0-9.]+\s*&\s*\$?<?\$?([.0-9]+)")
    krow = re.compile(r"^([A-Za-z0-9\\' +\-]+?)(?:\s*\\emph\{[^}]*\})?\s*&\s*(BEIR|BRIGHT|RTEB)\s*&\s*(\d+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&")
    for line in tex.splitlines():
        m = row.match(line.strip())
        if m:
            name = m.group(1).strip()
            rho[name] = (float(("-" if m.group(3) else "") + m.group(4)), int(m.group(2)), m.group(6))
        k = krow.match(line.strip())
        if k:
            name = k.group(1).strip()
            if name == "ArguAna" and "own instruction" in line:
                name = "ArguAna (+instr)"
            kappa[name] = (float(k.group(5)), int(k.group(3)))
    return rho, kappa


def main(argv):
    summary = Path(argv[1])
    tex = Path(argv[2]).read_text(encoding="utf-8")
    md_out = Path(argv[argv.index("--md") + 1]) if "--md" in argv else None
    rho, kappa = paper_tables(tex)
    rows = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
    by = {}
    for r in rows:
        if r.get("status") == "ok":
            by.setdefault(r["task"], {})[r["arm"]] = r
    lines = [
        "# Level check of the paper (old package) against the regen (mteb_gym main; NFCorpus at full scale, the rest nano tier)",
        "",
        "Not a replication: nano corpora, a 12-model roster, the rewritten judge prompt with the mteb task",
        "description injected, and a gpt-oss-20b generator. The question is only whether the per-corpus",
        "level and ordering survive the rewrite.",
        "",
        "| corpus | paper rho (n) | regen rho (n) | paper kappa (n) | regen S (n models) | regen tier |",
        "|---|---|---|---|---|---|",
    ]
    for task, arms in by.items():
        paper = NANO_TO_PAPER.get(task, task)
        pr = rho.get(paper) if paper else None
        pk = kappa.get(paper) if paper else None
        show = (paper or task).replace("\\'", "")  # the paper label is a results.tex key; print it without the LaTeX accent escape
        syn = arms.get("synthetic", {}).get("agreement") or {}
        org = arms.get("original", {}).get("reliability") or {}
        r_txt = f"{syn['spearman_rho']:+.3f} ({syn['n_models']})" if syn.get("spearman_rho") is not None else ("err: " + str(syn.get("error", "no synthetic arm"))[:40])
        s_txt = f"{org['s_committed']:+.3f}" if org.get("s_committed") is not None else ("err: " + str(org.get("error", "no original arm"))[:40])
        lines.append(
            f"| {show} | {pr[0]:+.3f} ({pr[1]}) | {r_txt} | {pk[0]:.3f} ({pk[1]}) | {s_txt} | {org.get('tier', '')} |"
            if pr and pk
            else f"| {show} | {pr[0]:+.3f} ({pr[1]}) | {r_txt} | n/a | {s_txt} | {org.get('tier', '')} |"
            if pr
            else f"| {show} | n/a | {r_txt} | n/a | {s_txt} | {org.get('tier', '')} |"
        )
    # ordering check over corpora present in both
    both = [(rho[NANO_TO_PAPER[t]][0], a["synthetic"]["agreement"]["spearman_rho"]) for t, a in by.items()
            if NANO_TO_PAPER.get(t) in rho and (a.get("synthetic", {}).get("agreement") or {}).get("spearman_rho") is not None]
    if len(both) >= 4:
        from scipy.stats import spearmanr

        rr, _ = spearmanr([x for x, _ in both], [y for _, y in both])
        lines += ["", f"Spearman between paper rho and regen rho over {len(both)} shared corpora: {rr:+.3f}"]
    kb = [(kappa[NANO_TO_PAPER[t]][0], a["original"]["reliability"]["s_committed"]) for t, a in by.items()
          if NANO_TO_PAPER.get(t) in kappa and (a.get("original", {}).get("reliability") or {}).get("s_committed") is not None]
    if len(kb) >= 4:
        from scipy.stats import spearmanr

        rk, _ = spearmanr([x for x, _ in kb], [y for _, y in kb])
        lines += [f"Spearman between paper kappa and regen S over {len(kb)} shared corpora: {rk:+.3f}"]
    out = "\n".join(lines)
    print(out)
    if md_out:
        md_out.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
