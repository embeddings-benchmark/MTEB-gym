# Regeneration run, 2026-09-12
NFCorpus end to end plus the 13 NanoBEIR corpora, the check asked for in the channel on Sep 9. Judge Qwen/Qwen3.6-27B (vllm 0.22.1, thinking off, 1024 max tokens), generator openai/gpt-oss-20b (reasoning effort low), a different family from the judge and from every entrant. 12 entrants, mteb/baseline-bm25s, all-MiniLM-L6-v2, all-mpnet-base-v2, bge-small/base/large-en-v1.5, e5-base-v2, gte-small, gte-large, mxbai-embed-large-v1, snowflake-arctic-embed-l-v2.0 and multilingual-e5-small, no Qwen-family embedder. NFCorpus at full scale with 100 synthetic queries and its own 323, the 13 nano corpora with 40 synthetic and their own 50 (49 for NanoTouche2020), top_k 10, seed 0, the corpus's mteb task prompt injected into the judge. Everything here came from the scripts in scripts/ at commit 71b2cf4 (branch tejas/analysis-scripts, main at eeb7e05 plus #32 and #57), mteb 2.15.1, on one node with 2 GPUs (jobs 14359 and 14372, the second resumed from the first's caches). NFCorpus took 73 min on the synthetic arm and 3.6 h on the original arm. Main has since changed the judge prompt (#68), the verdict identity (#70) and the original-arm sampling (#69), so pin that commit to reproduce these numbers. scripts/regen_nano_397b.sbatch (Qwen3.5-397B, 8 GPUs) was never run, so no number here comes from it.

The numbers live in results/. SUMMARY.jsonl has 32 rows, 28 corpus and arm rows, 3 repeats from the resumed job with the same numbers, and 1 export row; the last row per corpus and arm is the one used. leaderboard_export.json is the leaderboard app's data file. level_check.md puts the run next to the paper's tables. records/ holds the 30 per-run records, 28 from the 27B judge and 2 MiniMax. The verdict and prediction caches stay on the cluster. In the table, rho is Spearman between the judge ranking and official nDCG@10 over the 12 entrants on the synthetic arm, S is 2p - 1 on committed verdicts against the corpus's own qrels on the original arm, and the paper values are in brackets. They come from results/level_check.md except ArguAna's 0.349, which is the second ArguAna row of the paper's human-baseline table (measured with the corpus's own instruction) and is not printed by scripts/level_check.py.

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

A second judge, MiniMaxAI/MiniMax-M2.7 (FP8, 4 GPUs, a different family from the 27B judge, the generator and every entrant), judged the same NFCorpus queries and predictions from scripts/regen_nano_m27.sbatch. It thinks before answering, so JUDGE_MAX_TOKENS was 6144 and the gym keeps the last JSON object of the reply. The job ran 24 workers with a 900 s client timeout; the 120 s default ended the first original-arm attempt, which is the run_failed row in results/SUMMARY_MiniMaxAI_MiniMax-M2.7.jsonl. The completed original arm's 261 s is a replay from the verdict cache. Records are results/records/NFCorpus__MiniMax-M2.7__*.json, export in results/leaderboard_export_MiniMaxAI_MiniMax-M2.7.json.

| NFCorpus, synthetic arm, 100 queries | Qwen3.6-27B | MiniMax-M2.7 |
|---|---|---|
| rho vs official nDCG@10 (p) | 0.818 (0.001) | 0.895 (8e-5) |
| top-10 rho, Kendall tau | 0.721, 0.636 | 0.855, 0.758 |
| commit rate, first-position rate | 0.567, 0.693 | 0.727, 0.630 |
| parse failures (scored as ties) | 0.02% | 1.99% |
| judge calls, wall time | 13,198, 73 min on 1 GPU | 13,198, 9.0 h on 4 GPUs |

| NFCorpus, original arm, 323 queries | Qwen3.6-27B | MiniMax-M2.7 |
|---|---|---|
| S against the qrels, committed verdicts (95% CI) | 0.486 [0.440, 0.532] | 0.471 [0.424, 0.521] |
| committed agreement, clear-winner agreement | 0.743, 0.830 | 0.736, 0.826 |
| tier | A | A |
| commit rate, first-position rate | 0.665, 0.606 | 0.777, 0.595 |
| parse failures (scored as ties) | 0.02% | 4.05% |
| judge calls, wall time | 42,636, 3.6 h on 1 GPU | 42,636, about 28 h on 4 GPUs over the chained jobs (from the Slurm logs; the record's `evaluation_time` is the 261 s cache replay) |

The analyses sit under analysis/ and synthesis/, one number each here. synthesis/position_bias.md refits every record from one presentation order; on the synthetic arm the winner flips with the order on 0.50 of the pairs decided in both orders. synthesis/position_bias_controls.md separates bias from noise on the raw verdict rows; once the slot is random the second verdict per pair adds 0.025 rho on the synthetic arm. synthesis/rho_query_bootstrap.md resamples queries; the paired MiniMax minus 27B difference on NFCorpus is [-0.028, 0.175]. synthesis/m27_parse_failures.md goes through the MiniMax failed orders; 81.2 percent of the affected rows still got a committed verdict from the other order. analysis/NFCorpus/seed_baseline.md uses seed documents as labels with no judge, rho 0.671. analysis/NFCorpus/query_stats.md compares the query sets, the synthetic queries average 13.880 words. analysis/NFCorpus/scaling.md subsamples the verdicts, 20 queries give rho 0.786. analysis/NFCorpus_m27/ and analysis/NFCorpus_m27_orig/ repeat the seed baseline, scaling, controls and bootstrap on the MiniMax records; the MiniMax original-arm ratings correlate 0.944 with the official scores. synthesis/sweep_report.md is the cross-corpus summary read from the analysis JSONs, and the same curves for every nano corpus are under analysis/<task>/. On the cluster the scripts run from the run root; the analysis outputs here were copied from analysis_out/<name>/ to analysis/<name>/ with the pointer line at the foot of each .md rewritten to this folder.

Environment notes, the first two in scripts/regen_nano.sbatch, then the rerun commands in order.
- The conda env's bin goes on PATH (ninja and nvcc for the kernels vllm builds at first use) and its lib on LIBRARY_PATH and LD_LIBRARY_PATH (-lcudart at the GDN kernel link step).
- VLLM_USE_FLASHINFER_SAMPLER=0, the sampling kernel build needs curand.h and the env lacks it.
- NovaSearch/stella_en_400M_v5 (needs xformers) and intfloat/multilingual-e5-large-instruct (Pooling config) do not load in that venv, so scripts/regen_nano.py leaves them off the roster.
- The judge answers with a thinking trace unless enable_thinking false is passed; scripts/regen_nano.py does that.

```bash
sbatch scripts/regen_nano.sbatch            # both arms, NFCorpus then the 13 nano corpora
TASKS=NFCorpus sbatch scripts/regen_nano_m27.sbatch   # second judge on 4 GPUs, same queries and predictions
bash scripts/run_analysis.sh NFCorpus       # seed baseline, query stats and scaling on the 27B record; the seed baseline step needs commit 71b2cf4, the script left #57 after #60
bash scripts/run_analysis.sh NFCorpus <run root>/results MiniMax-M2.7 NFCorpus_m27   # same on the MiniMax record
python scripts/level_check.py results/SUMMARY.jsonl <paper results.tex> --md results/level_check.md
python synthesis/position_bias.py <run root>                 # single-order refits
python synthesis/position_bias_controls.py --root <run root> # controls; --only <record substring> --out-dir <dir> for one record
python synthesis/rho_query_bootstrap.py --root <run root>    # query bootstrap, same switches
```
