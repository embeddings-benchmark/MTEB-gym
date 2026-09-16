# Regeneration run, 2026-09-12: NFCorpus end to end plus the 13 NanoBEIR corpora

The end-to-end check asked for in the channel on Sep 9: one main-table corpus through the rewritten
package, both query arms, then the nano corpora as a smoke test. Everything here was produced by the
scripts in `scripts/` on the cluster at commit `71b2cf4` (branch `tejas/analysis-scripts`, which is
main at `eeb7e05` plus `mteb_gym/reliability.py` and `analysis/`), mteb 2.15.1.

## Setup

- Judge `Qwen/Qwen3.6-27B` (vllm 0.22.1, thinking off, `max_completion_tokens` 1024). Generator
  `openai/gpt-oss-20b` (reasoning effort low), a different family from the judge and from every entrant.
- 12 entrants, all in the cluster cache: `mteb/baseline-bm25s`, `all-MiniLM-L6-v2`, `all-mpnet-base-v2`,
  `bge-small/base/large-en-v1.5`, `e5-base-v2`, `gte-small`, `gte-large`, `mxbai-embed-large-v1`,
  `snowflake-arctic-embed-l-v2.0`, `multilingual-e5-small`. No Qwen-family embedder, so the judge-family
  overlap the paper flags stays out of this check.
- NFCorpus at full scale: 100 synthetic queries and the corpus's own 323 queries. The 13 NanoBEIR
  corpora: 40 synthetic queries and their own 50. `top_k` 10, seed 0, the package's default task
  description (the corpus's mteb prompt) injected into the judge.
- Two Slurm jobs on one node with 2 GPUs (14359, 14372; the second resumed the first's caches after a
  6 h time limit). NFCorpus synthetic arm 73 min, original arm 3.6 h; the nano sweep about 11 h in total.

## Results

`results/SUMMARY.jsonl` has one row per corpus and arm (diagnostics, rank agreement for the synthetic
arm, judge reliability for the original arm). `results/leaderboard_export.json` is the leaderboard
app's data file for the 14 ranked corpora. `results/level_check.md` puts the regen next to the paper's
tables. The per-run records (`records/*.json`, 28 files) are on the cluster under
`tmp_regen/results/records/` and will be added to this folder once cluster access is back (the login
node's SSH host key changed on Sep 16).

Paper value in brackets where the corpus is in the paper. rho is Spearman between the label-free
ranking and official nDCG@10 over the 12 entrants; S is the judge's chance-corrected agreement with
the corpus's own qrels on the original-query arm (the paper's kappa, 2p - 1 on committed verdicts).

| corpus | rho, synthetic arm | S, original arm | tier |
|---|---|---|---|
| NFCorpus (full) | 0.818 (0.670 on 25 models) | 0.486 (0.565) | A |
| NanoNFCorpus | 0.399 | 0.453 | B |
| NanoSciFact | 0.741 (0.743) | 0.436 (0.590) | B |
| NanoFiQA2018 | 0.580 (0.709) | 0.470 (0.552) | B |
| NanoArguAna | 0.420 (0.288) | 0.353 (0.151 generic, 0.349 with the task instruction) | C |
| NanoSCIDOCS | 0.573 (0.706) | 0.283 | C |
| NanoQuora | 0.238 (0.573) | 0.316 (0.334) | C |
| NanoDBPedia | 0.671 (0.423) | 0.546 (0.318) | A |
| NanoHotpotQA | 0.755 (0.386) | 0.904 | A |
| NanoFEVER | 0.490 (0.117) | 0.784 | A |
| NanoClimateFEVER | 0.168 (0.259) | 0.179 (0.058) | C |
| NanoNQ | 0.545 (-0.286 on 8 models) | 0.581 (0.405) | A |
| NanoTouche2020 | -0.280 (0.063) | 0.011 | C |
| NanoMSMARCO | 0.804 | 0.236 | C |

NFCorpus synthetic arm: 13,198 judge calls, commit rate 0.567, tie rate 0.433, first-position rate
0.693, parse failures 0.02%, top-10 rho 0.721, Kendall tau 0.636. Original arm: 42,636 calls, 262 of
323 queries carry a positive label, committed agreement 0.743, S 0.486 with a query-clustered 95% CI
of [0.440, 0.532], clear-winner agreement 0.830; ratings from the real queries correlate 0.902 with
official nDCG@10.

## What the run says

- The level holds under the rewrite where the corpus is at full scale: NFCorpus rho 0.82 against the
  paper's 0.67, S 0.49 against 0.57, on a smaller and weaker roster.
- ArguAna reproduces the paper's instruction repair: S 0.35, where the paper measured 0.15 with the
  generic prompt and 0.35 with the task instruction. The rewrite injects that instruction by default.
- The judge is position biased in this setup. The first-position rate runs 0.52 to 0.88 across the
  synthetic arms (mean 0.72; 0.64 on the original arms) and commit rates fall to 0.30 on the worst
  corpora. Among pairs the judge decides in both orders, the winner flips with the order on half of
  them (0.50 synthetic, 0.39 original). Both presentation orders are judged and averaged, which cancels
  the slot bonus by construction. Refitting from one order alone hands that bonus to whichever model
  the pair enumeration put in the slot, and the roster is not random with respect to quality, so the
  two single-order refits swing with the roster (NFCorpus 0.37 and 0.72 against 0.82 from both;
  NanoNQ -0.73 and 0.78 against 0.55). Read them as a bound on how far a fixed slot assignment can
  move a ranking, not as one order being more faithful than the other (`synthesis/position_bias.md`,
  all 28 records; the both-order refit reproduces each record's stored rho and first-position rate
  exactly).
- The nano tier is noisy at 40 queries. NFCorpus reads 0.40 nano against 0.82 full, and only 6 of 14
  synthetic-arm rhos reach p < 0.05. Treat the nano rows as a smoke test of the pipeline, not as
  estimates for the main table.
- Seed documents as labels, with no judge, give rho 0.67 on NFCorpus against the judge's 0.82
  (`analysis/NFCorpus/seed_baseline.md`; #60 now computes the same baseline inside the package).
- Query statistics on NFCorpus (`analysis/NFCorpus/query_stats.md`): the kept synthetic queries run
  13.9 words against 3.3 for the corpus's own, 100% are phrased as questions against 14%, 63% of the
  query words appear in the seed documents, and the quality gate scored every kept query 5 of 5, so
  it filtered nothing; 160 generated, 100 kept.
- Subsampling the NFCorpus verdicts (`analysis/NFCorpus/scaling.md`): 20 queries give rho 0.79, 40
  give 0.81, 80 give 0.82; a quarter of the pairs gives 0.81 and half gives 0.83. More queries help,
  more pairs barely do. The same curves for every nano corpus are under `analysis/<task>/`.

## Environment notes for the main table

The vllm env on the cluster needs, in the job script: the conda env's `bin` on `PATH` (ninja and nvcc
for the kernels vllm builds at first use), `VLLM_USE_FLASHINFER_SAMPLER=0` (the sampling kernel build
needs `curand.h`, absent from the env), and the env's `lib` on `LIBRARY_PATH` and `LD_LIBRARY_PATH`
(`-lcudart` at the GDN kernel link step). Two embedders do not load in that venv:
`NovaSearch/stella_en_400M_v5` (needs xformers) and `intfloat/multilingual-e5-large-instruct`
(sentence-transformers Pooling config). The Nano datasets are not in the offline HF cache by default;
`scripts/regen_nano.sbatch` has all of this. The judge answers with a thinking trace unless
`enable_thinking: false` is passed; `scripts/regen_nano.py` does that.

## Reproducing

```bash
sbatch scripts/regen_nano.sbatch            # both arms, NFCorpus then the 13 nano corpora
bash scripts/run_analysis.sh NFCorpus       # truth, original queries, then the three analyses
python scripts/level_check.py results/SUMMARY.jsonl <paper results.tex> --md level_check.md
sbatch scripts/regen_nano_397b.sbatch       # optional: re-judge the same queries with Qwen3.5-397B (8 GPUs, cached predictions)
```

`analysis/<task>/scaling.json`, `seed_baseline.json` and `query_stats.json` come from `analysis/` in
#57; `synthesis/position_bias.py` is the single-order refit. `synthesis/sweep_report.md` is a
cross-corpus summary of the three analyses, every number in it read from the JSON files under
`analysis/` and recomputed against them.
