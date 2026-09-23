"""Nano-tier regeneration under the rewritten package (mteb_gym): both query arms per corpus,
judge reliability on the original arm, rank agreement on the synthetic arm, then the leaderboard
export. Launched by regen_nano.sbatch on a 2-GPU node; MOCK=1 exercises the whole path on the login
node with the mock LLM and three small models before any GPU slot is used.

Usage: python regen_nano.py <output_folder>
Env (real run): JUDGE_MODEL JUDGE_URL GEN_MODEL GENERATOR_URL; optional TASKS (comma list, limits the
pass), JUDGE_MAX_TOKENS (default 1024; raise it for a judge that reasons before answering), WORKERS (default 16;
concurrent judge calls, raise it for a server that batches well), ARMS (default synthetic,original),
JUDGE_TIMEOUT (seconds per judge call, default 120; a judge that thinks for thousands of tokens under
load needs more, a timeout ends the whole arm after the client's 4 retries).
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import mteb_gym as gym
from leaderboard.export import ExportError, build_export
from mteb_gym.reliability import judge_reliability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("regen")

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "NFCorpus",  # the full main-table corpus first: the end-to-end check Adnan asked for (Slack, Sep 9)
    "NanoNFCorpusRetrieval",
    "NanoSciFactRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoArguAnaRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoQuoraRetrieval",
    "NanoDBPediaRetrieval",
    "NanoHotpotQARetrieval",
    "NanoFEVERRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoNQRetrieval",
    "NanoTouche2020Retrieval",
    "NanoMSMARCORetrieval",
]
# twelve entrants, all in the cluster's HF cache: lexical, small, base, large, instruct; no Qwen-family
# embedder so the judge-family overlap the paper flags does not enter this level check
MODELS = [
    "mteb/baseline-bm25s",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-base-v2",
    "thenlper/gte-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "thenlper/gte-small",  # stella_en_400M_v5 needs xformers, absent from the venv (job 14210)
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "intfloat/multilingual-e5-small",  # multilingual-e5-large-instruct fails to load in the venv (Pooling config; job 14253)
]

if os.environ.get("TASKS"):  # e.g. TASKS=NFCorpus for a second judge on the level-check corpus only
    wanted = [t.strip() for t in os.environ["TASKS"].split(",") if t.strip()]
    unknown = sorted(set(wanted) - set(TASKS))
    if unknown:
        sys.exit(f"TASKS names tasks outside the sweep: {unknown}")
    TASKS = [t for t in TASKS if t in wanted]

if os.environ.get("MOCK"):
    TASKS = [os.environ.get("MOCK_TASK", "NanoNFCorpusRetrieval")]
    MODELS = ["mteb/baseline-bm25s", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
    judge, generator = gym.MockLLM(seed=1), gym.MockLLM(seed=2)
    N_QUERIES, BOOTSTRAP, WORKERS = 10, 50, 2
else:
    # judge: thinking off so the answer is the JSON, not a reasoning trace that overruns the cap
    judge = gym.LLM(
        os.environ["JUDGE_MODEL"],
        base_url=os.environ["JUDGE_URL"],
        api_key="EMPTY",
        max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "1024")),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=float(os.environ.get("JUDGE_TIMEOUT", "120")),
    )
    # generator: gpt-oss reasons before answering; low effort, no cap (a cap would count the reasoning),
    # the server's reasoning parser keeps the reasoning out of the content the gym parses
    generator = gym.LLM(
        os.environ["GEN_MODEL"],
        base_url=os.environ["GENERATOR_URL"],
        api_key="EMPTY",
        extra_body={"reasoning_effort": "low"},
    )
    N_QUERIES, BOOTSTRAP = 40, 1000
    WORKERS = int(
        os.environ.get("WORKERS", "16")
    )  # 16 drove the 27B judge at ~3 calls/s; a thinking MoE judge wants more in flight
N_QUERIES_BY_TASK = {"NFCorpus": 100}  # full-scale corpus: the package default; nano corpora keep 40

if os.environ.get("JUDGE_ONLY"):
    # a second judge over the SAME queries and predictions: both are cached by identity (generator id +
    # parameters, model@revision + query set), so only the verdicts are new. The generator endpoint is
    # never called; a cache miss would fail on the first call instead of silently regenerating queries.
    from mteb_gym.retrieval import slug

    gen_id = os.environ["GEN_MODEL"]
    generator = gym.LLM(gen_id, base_url="http://127.0.0.1:9/v1", api_key="EMPTY")
    cached = sorted((OUT / "queries").glob(f"*-{slug(gen_id)}-*.json"))  # one per task for this generator
    log.info("JUDGE_ONLY: %d cached query sets for generator %s", len(cached), gen_id)
    if len(cached) < len(TASKS):
        log.error(
            "JUDGE_ONLY: only %d query sets cached for %d tasks; run the generator job first", len(cached), len(TASKS)
        )
        sys.exit(2)

JUDGE_ID = os.environ.get("JUDGE_MODEL")  # None in MOCK mode
summary_path = OUT / (
    "SUMMARY.jsonl" if not os.environ.get("JUDGE_ONLY") else f"SUMMARY_{JUDGE_ID.replace('/', '_')}.jsonl"
)


def note(row: dict) -> None:
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with summary_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log.info("SUMMARY %s", json.dumps(row))


ARMS = tuple(a.strip() for a in os.environ.get("ARMS", "synthetic,original").split(",") if a.strip())
if set(ARMS) - {"synthetic", "original"}:
    sys.exit(f"ARMS must be synthetic and/or original, got {ARMS}")

for task in TASKS:
    for arm in ARMS:
        t0 = time.time()
        try:
            res = gym.run(
                task,
                MODELS,
                judge,
                generator,
                queries=arm,
                n_queries=N_QUERIES_BY_TASK.get(task, N_QUERIES),
                top_k=10,
                seed=0,
                output_folder=str(OUT),
                workers=WORKERS,
            )
        except Exception as e:  # keep going: one corpus must not sink the run, but say so loudly
            log.error("%s %s FAILED: %s\n%s", task, arm, e, traceback.format_exc())
            note({"task": task, "arm": arm, "status": "run_failed", "error": str(e)[:300]})
            continue
        row = {
            "task": task,
            "arm": arm,
            "status": "ok",
            "record": str(res.path),
            "n_queries": res.record["config"]["n_queries"],
            "diagnostics": res.record["diagnostics"],
            "seconds": round(time.time() - t0),
        }
        if arm == "original":
            r = judge_reliability(res, OUT, bootstrap=BOOTSTRAP, seed=0)
            row["reliability"] = (
                {"error": r["error"]}
                if "error" in r
                else {
                    k: r[k]
                    for k in (
                        "committed_agreement",
                        "s_committed",
                        "s_committed_ci95",
                        "clear_winner_agreement",
                        "tier",
                    )
                }
            )
        else:
            try:
                a = res.agreement(evaluate_missing=not os.environ.get("MOCK"), bootstrap=BOOTSTRAP, seed=0)
                row["agreement"] = (
                    {"error": a["error"]}
                    if "error" in a
                    else {
                        k: a.get(k) for k in ("n_models", "spearman_rho", "spearman_p", "spearman_top10", "kendall_tau")
                    }
                )
            except Exception as e:  # anchors need the MTEB results repo or a GPU pass; report, do not stop
                log.error("%s agreement FAILED: %s", task, e)
                row["agreement"] = {"error": str(e)[:300]}
        note(row)

try:  # judge filter: after a second-judge pass the folder holds two records per corpus and arm
    export = build_export(OUT, judge=JUDGE_ID)
    dropped = []
except ExportError as e:
    log.warning("export refused (%s); retrying with --allow-missing", e)
    export = build_export(OUT, judge=JUDGE_ID, allow_missing=True)
    dropped = export["meta"]["dropped"]
(
    OUT
    / (
        "leaderboard_export.json"
        if not os.environ.get("JUDGE_ONLY")
        else f"leaderboard_export_{JUDGE_ID.replace('/', '_')}.json"
    )
).write_text(json.dumps(export, indent=1))
note(
    {
        "task": "*",
        "arm": "*",
        "status": "export",
        "ranked": sorted(export["corpora"]),
        "reliability_rows": sorted(export["reliability"]),
        "dropped": dropped,
    }
)
log.info(
    "done: %d ranked corpora, %d reliability rows -> %s",
    len(export["corpora"]),
    len(export["reliability"]),
    OUT / "leaderboard_export.json",
)
