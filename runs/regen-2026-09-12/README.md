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
tables. `results/records/` holds the 28 per-run records (one per corpus and arm; config, diagnostics,
ratings, and the agreement or reliability readout), the package's own output. The verdict, prediction
and query caches (179 MB) stay on the cluster under `tmp_regen/results/`.

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
- Three controls on the raw verdict rows separate the two effects behind that gap
  (`synthesis/position_bias_controls.md`, 28 records, 200 draws per control). Drawing one order at random
  per pair and query, so one verdict per pair but no fixed slot, gives rho 0.47 on the synthetic arm and
  0.57 on the original arm, against 0.49 and 0.57 from both orders and 0.22 and 0.44 from a single fixed
  order. The bias share, random order minus the fixed-order mean, is 0.25 synthetic and 0.13 original,
  positive in 13 of 14 and 11 of 14 records (nominal Wilcoxon p 0.002 and 0.017), and the same holds
  against a roster-permuted baseline (0.21 and 0.12). The noise share, both orders minus random order, is
  0.025 synthetic and 0.003 original, so once the slot is random the second verdict per pair adds little.
  Fixing one random order per pair instead, so the slot bonus repeats across a pair's queries, costs a
  further 0.11 synthetic and 0.04 original. Of the ten signed-rank tests, Bonferroni keeps the three
  synthetic bias results, Holm adds one original, the original bias share and per-pair gap survive only a
  false-discovery-rate step, and the noise shares survive none. The actual roster sits inside the
  roster-permuted band (1000 draws) on every record except NanoNQ, where the roster is most aligned with
  the official scores (0.74). The records' own `spearman_ci95` resamples models, not queries, so it is
  not a query-level uncertainty for these gaps. Three records carry an exact tie in total wins that
  the Bradley-Terry stopping tolerance breaks arbitrarily; scoring the tie as a tie moves their rho by
  0.014 to 0.027 (NanoNQ synthetic 0.545 to 0.518, NanoQuora synthetic 0.238 to 0.224, NanoNFCorpus
  original 0.769 to 0.750).
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
sbatch scripts/regen_nano_397b.sbatch       # optional: re-judge the same queries with Qwen3.5-397B (8 GPUs, cached predictions; not run, see below)
TASKS=NFCorpus sbatch scripts/regen_nano_m27.sbatch   # second judge, MiniMax-M2.7 on 4 GPUs, same queries and predictions
```

The 397B pass was submitted on Sep 12 (job 14398) and cancelled while still pending for 8 GPUs; it has not
run, so every number here is from the 27B judge. A second-judge pass with `MiniMaxAI/MiniMax-M2.7` (FP8, 4
GPUs, a family different from the judge, the generator and every entrant) over the same NFCorpus queries and
predictions is running from `scripts/regen_nano_m27.sbatch`; the driver's `TASKS`, `ARMS`, `WORKERS`,
`JUDGE_MAX_TOKENS` and `JUDGE_TIMEOUT` switches exist for it. That model thinks before it answers (about 30
judge calls a minute on 4 H100s whether 16 or 48 calls are in flight, so the job runs 24 with a 900 s client
timeout; the package default of 120 s ended one arm after its 4 retries), and the gym keeps the last JSON
object of the reply. The pass spans several 12 h jobs chained with `--dependency=afterany`; the verdict cache
resumes at the row. Its records and a `SUMMARY_<judge>.jsonl`
land here as they complete. The synthetic arm is in: `results/records/NFCorpus__MiniMax-M2.7__*.json` and
`results/SUMMARY_MiniMaxAI_MiniMax-M2.7.jsonl`.

| NFCorpus, synthetic arm, 100 queries | Qwen3.6-27B | MiniMax-M2.7 |
|---|---|---|
| rho vs official nDCG@10 (p) | 0.818 (0.001) | 0.895 (8e-5) |
| top-10 rho, Kendall tau | 0.721, 0.636 | 0.855, 0.758 |
| commit rate, first-position rate | 0.567, 0.693 | 0.727, 0.630 |
| parse failures (scored as ties) | 0.02% | 1.99% |
| judge calls, wall time | 13,198, 73 min on 1 GPU | 13,198, 9.0 h on 4 GPUs |

The two judges rank the 12 entrants the same to rho 0.95; MiniMax places bge-large third, where the
official scores put it, while the 27B judge had it sixth. Same query set and judge prompt hash, so only the
verdicts differ. The original-query arm (S against the corpus qrels) is running.

The same three analyses on the MiniMax record (`analysis/NFCorpus_m27/`): seed documents as labels give
rho 0.671 against the judge's 0.895, so the judge is worth 0.22 here (0.15 under the 27B judge). Subsampling
the verdicts, 20 queries give 0.84, 40 give 0.875, 80 give 0.895; a quarter of the pairs gives 0.865 and half
gives 0.90. Position bias: the single-order refits give 0.671 and 0.860, random order per pair and query
0.900, one random order per pair 0.843, half the queries with both orders 0.879; bias share 0.134, noise
share -0.005, first-position rate 0.630, no win ties. So the less biased judge loses less to a fixed slot
and nothing to a single random order.

Two more checks on the judge comparison. A query-level bootstrap of rho (`synthesis/rho_query_bootstrap.md`,
1000 resamples of the queries, both orders kept) puts MiniMax at [0.77, 0.96] and the 27B judge at [0.69,
0.91] on NFCorpus; paired on the same query draws the difference is 0.063 with interval [-0.03, 0.18], so
the 0.077 gap does not clear zero. Across all 28 records that query interval is 2.1 to 2.5 times narrower
than the record's own `spearman_ci95`, which resamples models, and no noise share in the position-bias
controls exceeds one query-bootstrap sd. The 262 unparsed MiniMax orders (`synthesis/m27_parse_failures.md`)
are the last rows to finish in each pair file (median completion decile 95 against 48 for other rows),
correlate with pair wall time, and left no reasoning text, which fits replies cut at the 6144-token cap
while the model was still thinking; order 2 fails twice as often as order 1 (2.65 against 1.32 percent),
and the 22 generated queries that carry a non-ASCII character (mostly a non-breaking hyphen) fail at twice
the rate of the rest. Input length barely varies in this run and does not explain them. 81 percent of the
affected rows still got a committed verdict from the other order, and dropping the failed rows instead of
scoring them as ties leaves rho and the 12-model order unchanged.

`analysis/<task>/scaling.json`, `seed_baseline.json` and `query_stats.json` come from `analysis/` in
#57; `synthesis/position_bias.py` is the single-order refit, `synthesis/position_bias_controls.py` the
controls and `synthesis/rho_query_bootstrap.py` the query bootstrap. `synthesis/sweep_report.md` is a
cross-corpus summary of the three analyses, every number in it read from the JSON files under
`analysis/` and recomputed against them.
