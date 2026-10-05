"""
Judge agreement with the MTEB Arena's human votes.

The MTEB Arena (Hugging Face dataset mteb/arena-results, config retrieval_battle, split data) holds
4,087 human votes on pairwise retrieval battles: a query typed by a visitor, one document from each
of two anonymous models, and a vote (left, right, tie, both bad). Re-judging every battle with the
gym's own judge and scoring the judge against the vote asks the question of mteb_gym.reliability on
human preferences instead of qrels: does the judge agree with people on the comparisons it makes?
The paper reports this as kappa 0.571 on the 4,087 votes.

    committed_agreement   P(judge picks the voted side | vote is left or right, judge commits)
    s_committed           2p - 1, the paper's kappa (reliability.s_of); cohen_kappa_committed alongside
    human_ties_judge_tied  on tie and both-bad votes, how often the judge also scored 0.5
    n_parse_failed        verdicts with a judge answer that did not parse (the judge scores that order as a tie)

Ties and both-bad votes are not scored, the same rule as reliability's official ties. The interval is a
percentile bootstrap over the decisive battles, seed pinned; the tier is read from its lower bound.

The judging code is the gym's: mteb_gym.judge.Judge.judge_pair on mteb_gym.retrieval.Ranked lists,
both presentation orders, the same prompt, the same parsing. Only the lists differ: the arena showed one
document per side, so each Ranked list holds one text.

Fields of a retrieval_battle row (exact names; see analysis_notes/arena_schema.md):
    tstamp, task_type ("retrieval"), type (leftvote | rightvote | tievote | bothbadvote | share),
    models (["", ""]), ip (""), 0_conv_id, 0_model_name, 0_prompt, 0_output ([[query, document]]),
    0_corpus, 1_conv_id, 1_model_name, 1_prompt, 1_output, 1_corpus
A "share" row is a share click, not a vote, and is skipped (7 of the 4,094 rows).

    python -m analysis.arena --battles retrieval_battle.json --judge-model Qwen/Qwen3-8B \\
        --judge-url http://localhost:8000/v1 --out analysis/out/arena.json

--battles is a JSON list of rows, a page from the datasets-server rows endpoint ({"rows": [{"row": ...}]}),
or a .jsonl with one row per line (what datasets' Dataset.to_json writes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mteb_gym.judge import Judge
from mteb_gym.reliability import ESTIMATOR_NOTE, assign_tier, cohen_kappa_of, judge_winner, s_of, tier_of
from mteb_gym.retrieval import Ranked, slug

logger = logging.getLogger(__name__)

VOTES = {"leftvote": "A", "rightvote": "B", "tievote": "tie", "bothbadvote": "both_bad"}
NOT_VOTES = ("share",)  # the only row type that is skipped; anything else unknown is refused
DECISIVE = ("A", "B")
HUMAN_TIES = ("tie", "both_bad")


class ArenaError(Exception):
    pass


# ----------------------------------------------------------------------------- battles from arena rows
def _check_template(text: str, where: str) -> str:
    """Inputs are data; something that looks like an unfilled template is refused, not interpreted."""
    if "${" in text:
        raise ArenaError(f"{where} contains template-looking text ('${{...}}'); refusing to read it")
    return text


def _rows(path_or_rows: str | Path | Sequence[dict] | dict) -> list[dict]:
    """Arena rows from a list, a rows-endpoint page, or a .json / .jsonl file of either."""
    if isinstance(path_or_rows, (str, Path)):
        path = Path(path_or_rows)
        text = _check_template(path.read_text(encoding="utf-8"), str(path))
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return _rows(json.loads(text))
    if isinstance(path_or_rows, dict):  # datasets-server page: {"rows": [{"row_idx": ..., "row": {...}}]}
        return [r["row"] if "row" in r else r for r in path_or_rows["rows"]]
    return [r["row"] if "row" in r else r for r in path_or_rows]


def _docs(output: Any) -> list[str]:
    """The shown documents from `<side>_output`: every inner list is [query, doc, ...]; the query is dropped."""
    return [str(d) for inner in output or [] for d in inner[1:]]


def battle_of(row: dict, index: int) -> dict | None:
    """One battle from one row, or None for a share row. A row of any other unknown type, or a voted row
    with an empty side, is refused rather than skipped or judged as a tie."""
    kind = row.get("type")
    if kind in NOT_VOTES:
        return None
    vote = VOTES.get(kind)
    if vote is None:
        raise ArenaError(f"row {index}: type {kind!r} is neither a vote {sorted(VOTES)} nor in {NOT_VOTES}")
    ids = (row.get("0_conv_id"), row.get("1_conv_id"))
    battle = {
        "battle_id": f"{ids[0]}__{ids[1]}" if all(ids) else f"row{index}",
        "query": str(row["0_prompt"]),
        "model_a": str(row["0_model_name"]),
        "model_b": str(row["1_model_name"]),
        "docs_a": _docs(row.get("0_output")),
        "docs_b": _docs(row.get("1_output")),
        "vote": vote,
    }
    for side in ("a", "b"):
        if not battle[f"docs_{side}"]:
            raise ArenaError(f"battle {battle['battle_id']} (row {index}): side {side} shows no document")
    return battle


def load_battles(path_or_rows: str | Path | Sequence[dict] | dict) -> list[dict]:
    """{battle_id, query, model_a, model_b, docs_a, docs_b, vote} per voted row; share rows are skipped."""
    battles, skipped = [], 0
    for i, row in enumerate(_rows(path_or_rows)):
        b = battle_of(row, i)
        if b is None:
            skipped += 1
            continue
        _check_template(" ".join([b["query"], *b["docs_a"], *b["docs_b"]]), f"battle {b['battle_id']}")
        battles.append(b)
    if skipped:
        logger.info("%d rows are not votes (type in %s); skipped", skipped, NOT_VOTES)
    return battles


# ----------------------------------------------------------------------------- re-judging
def _doc_id(text: str) -> str:
    """Identical documents get identical ids, so Judge.judge_pair ties identical sides without a call."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def ranked(battle: dict, side: str, top_k: int) -> Ranked:
    docs = battle[f"docs_{side}"][:top_k]
    return Ranked(battle["battle_id"], battle["query"], [_doc_id(d) for d in docs], list(docs))


def judge_battles(client, battles: Sequence[dict], top_k: int = 10, workers: int = 8) -> list[dict]:
    """Every battle through Judge.judge_pair, both presentation orders: {battle_id, model_a, model_b, vote,
    score_a, raw, parsed_ok} in input order. The vote is carried along so kappa() needs nothing else."""
    if top_k < 1:
        raise ArenaError(f"top_k must be at least 1, got {top_k}; an empty side would tie every battle")
    judge = Judge(client, workers=workers)

    def one(b: dict) -> dict:
        v = judge.judge_pair(ranked(b, "a", top_k), ranked(b, "b", top_k), b["model_a"], b["model_b"])
        return {
            "battle_id": b["battle_id"],
            "model_a": b["model_a"],
            "model_b": b["model_b"],
            "vote": b["vote"],
            "score_a": float(v.score_a),
            "raw": list(v.raw),
            "parsed_ok": list(v.parsed_ok),
        }

    def collect(results) -> list[dict]:
        out = []
        for i, v in enumerate(results, 1):
            out.append(v)
            if i % 100 == 0:
                logger.info("%d/%d battles judged", i, len(battles))
        failed = sum(not all(v["parsed_ok"]) for v in out)
        if failed:
            logger.warning("%d/%d verdicts have a judge answer that did not parse (scored tie)", failed, len(out))
        return out

    if workers <= 1 or len(battles) <= 1:
        return collect(map(one, battles))
    with ThreadPoolExecutor(max_workers=workers) as pool:  # map keeps input order
        return collect(pool.map(one, battles))


# ----------------------------------------------------------------------------- agreement with the votes
_CELL_INDEX = {"AA": 0, "AB": 1, "BA": 2, "BB": 3}
_ABSTAIN = 4  # decisive vote, judge at 0.5


def tally(verdicts: Sequence[dict]) -> tuple[list[int], dict[str, int]]:
    """Per decisive battle, its cell (rows: vote, columns: judge) or _ABSTAIN, plus the counts."""
    cells: list[int] = []
    keys = ("battles", "decisive", "committed", "abstain", "tie_votes", "both_bad_votes", "human_ties_judge_tied")
    n = dict.fromkeys((*keys, "parse_failed"), 0)
    for v in verdicts:
        n["battles"] += 1
        vote, jw = v["vote"], judge_winner(float(v["score_a"]))
        n["parse_failed"] += not all(v.get("parsed_ok") or [])  # hand-built verdicts carry no parsed_ok
        if vote not in DECISIVE:
            if vote not in HUMAN_TIES:
                raise ArenaError(f"battle {v.get('battle_id')}: vote {vote!r} is not one of {sorted(VOTES.values())}")
            n["tie_votes" if vote == "tie" else "both_bad_votes"] += 1
            n["human_ties_judge_tied"] += jw == "tie"
            continue
        n["decisive"] += 1
        if jw == "tie":
            n["abstain"] += 1
            cells.append(_ABSTAIN)
            continue
        n["committed"] += 1
        cells.append(_CELL_INDEX[vote + jw])
    return cells, n


def _counts(cells: Sequence[int]) -> list[int]:
    return np.bincount(np.asarray(cells, dtype=int), minlength=5).tolist()[:4]


def _percentiles(draws: list[float]) -> list[float | None]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else [None, None]


def bootstrap_ci(cells: Sequence[int], n: int, seed: int) -> dict[str, list[float | None]]:
    """Percentile intervals for S and Cohen's kappa, resampling decisive battles with replacement."""
    arr = np.asarray(cells, dtype=int)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"s": [], "cohen": []}
    for _ in range(n if len(arr) else 0):
        counts = _counts(rng.choice(arr, size=len(arr), replace=True))
        for name, value in (("s", s_of(counts)), ("cohen", cohen_kappa_of(counts))):
            if value is not None:
                draws[name].append(value)
    return {k: _percentiles(v) for k, v in draws.items()}


def kappa(verdicts: Sequence[dict], *, bootstrap: int = 1000, seed: int = 0) -> dict:
    """The readout: agreement between the judge and the votes on decisive, committed battles."""
    cells, n = tally(verdicts)
    total = _counts(cells)
    s, cohen = s_of(total), cohen_kappa_of(total)
    ci = bootstrap_ci(cells, bootstrap, seed)
    human_ties = n["tie_votes"] + n["both_bad_votes"]
    return {
        "n_battles": n["battles"],
        "n_decisive": n["decisive"],
        "n_committed": n["committed"],
        "n_abstain": n["abstain"],
        "n_tie_votes": n["tie_votes"],
        "n_both_bad_votes": n["both_bad_votes"],
        "n_parse_failed": n["parse_failed"],
        "committed_agreement": (total[0] + total[3]) / n["committed"] if n["committed"] else None,
        "s_committed": s,
        "s_committed_ci95": ci["s"],
        "kappa_committed": s,  # the paper's kappa = 2p - 1; an alias of s_committed
        "cohen_kappa_committed": cohen,
        "cohen_kappa_ci95": ci["cohen"],
        "human_ties_judge_tied": n["human_ties_judge_tied"] / human_ties if human_ties else None,
        "cells": dict(zip(_CELL_INDEX, total)),
        **assign_tier(*ci["s"]),
        "tier_of_point_estimate": tier_of(s),
        "bootstrap": bootstrap,
        "seed": seed,
        "estimator_note": ESTIMATOR_NOTE,
    }


def format_readout(r: dict) -> str:
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    lo, hi = r["s_committed_ci95"]
    return "\n".join(
        [
            f"battles {r['n_battles']}: decisive {r['n_decisive']} (committed {r['n_committed']}, "
            f"abstain {r['n_abstain']}), tie votes {r['n_tie_votes']}, both bad {r['n_both_bad_votes']}",
            f"committed agreement {f(r['committed_agreement'])}",
            f"kappa (S = 2p - 1) {f(r['s_committed'])}  95% CI [{f(lo)}, {f(hi)}]  tier {r['tier_label']}",
            f"cohen kappa {f(r['cohen_kappa_committed'])}",
            f"judge also tied on {f(r['human_ties_judge_tied'])} of the human tie / both-bad votes",
            f"cells (vote x judge) {r['cells']}",
            f"judge answers that did not parse (scored tie) {r['n_parse_failed']}",
        ]
    )


# ----------------------------------------------------------------------------- entry point
def _client(model: str, base_url: str | None):
    """The judge: the gym's LLM on an OpenAI-compatible endpoint (api_key EMPTY), or MockLLM for a dry run."""
    from mteb_gym import LLM, MockLLM

    return MockLLM() if model == "mock" else LLM(model, base_url=base_url, api_key="EMPTY")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--battles", required=True, help="retrieval_battle rows as .json (list or rows page) or .jsonl")
    ap.add_argument("--judge-model", required=True, help="judge model id on the endpoint ('mock' for a dry run)")
    ap.add_argument("--judge-url", default=None, help="base url of the OpenAI-compatible judge endpoint")
    ap.add_argument("--top-k", type=int, default=10, help="documents per side shown to the judge (the arena has 1)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent judge calls")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N battles (smoke run)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default analysis/out/arena_<judge>.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    battles = load_battles(args.battles)[: args.limit]
    if not battles:
        raise ArenaError(f"no voted battle in {args.battles}")
    verdicts = judge_battles(_client(args.judge_model, args.judge_url), battles, top_k=args.top_k, workers=args.workers)
    readout = kappa(verdicts, bootstrap=args.bootstrap, seed=args.seed)
    out = Path(args.out or f"analysis/out/arena_{slug(args.judge_model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"judge_model": args.judge_model, "battles": args.battles, "readout": readout, "verdicts": verdicts},
            indent=1,
        )
    )
    print(format_readout(readout))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
