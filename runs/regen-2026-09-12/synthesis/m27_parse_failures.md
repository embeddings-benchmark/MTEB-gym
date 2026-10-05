# Parse failures of the MiniMax-M2.7 judge on NFCorpus synthetic

Record `NFCorpus__MiniMax-M2.7__gpt-oss-20b__q100-s0-e3221360.json` (judge MiniMaxAI/MiniMax-M2.7, 12 models, 100 gpt-oss-20b queries, 6600 rows, 13198 judge calls) against `NFCorpus__Qwen3.6-27B__gpt-oss-20b__q100-s0-75c61651.json` (judge Qwen/Qwen3.6-27B, same queries, same models, same prediction files). Verdict files were located with `mteb_gym.reliability.verdict_file`; none were reversed. The counts below reproduce each record's `diagnostics.parse_failure_rate` exactly, and the baseline refit reproduces each record's ratings and `agreement.spearman_rho`.

A failed order is a judge answer with no winner in {A, B, tie}; it is scored as a tie for that order and flagged in `parsed_ok`. Every one of the 262 failed MiniMax orders left no reasoning text either (all 238 single failure rows keep exactly one reasoning segment, the 12 double failure rows keep none), so no reasoning text was recovered from any failed order. That fits a cut reply, though on its own it does not exclude a JSON object with an off-vocabulary winner and no reasoning text. The raw answers and finish reasons are not stored. What the rows do show is that the failed rows were the slowest calls in their pair file (see Completion order below). The run set `JUDGE_MAX_TOKENS=6144` because the judge emits a think block before its answer, passed `enable_thinking: false` as a chat template argument the MiniMax template may not honour, and served the model without a reasoning parser, so a reply that reaches the cap inside the think block carries no JSON object for `extract_json`. The SDK retry path either returns a normal answer or raises, so a generation that ran to the cap is the one mechanism in the run that makes a call both far slower than its neighbours and empty of JSON. The position evidence is consistent with cap-cut replies; it does not prove it.

## Counts

| | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| judge calls | 13198 | 13198 |
| failed orders | 262 (1.99 percent) | 3 (0.023 percent) |
| order 1 failures | 87 | 1 |
| order 2 failures | 175 | 2 |
| rows failed in order 1 only | 75 | 1 |
| rows failed in order 2 only | 163 | 2 |
| rows failed in both orders | 12 (independent expectation 2.3) | 0 |
| rows with any failure | 250 of 6599 judged (3.79 percent) | 3 of 6599 |
| identical result sets (no call) | 1 | 1 |

Order 2 (the pass that shows `model_b` as System A) fails twice as often as order 1, 2.65 against 1.32 percent over 6599 calls each. Twelve rows failed in both orders where independence predicts about two, so failures repeat within a row.

## Where the failures sit

By query. 63 of 100 queries have at least one failed order and 37 have none; the count per query is overdispersed (variance over mean 4.0, maximum 13 of 132 orders). The five worst queries hold 22 percent of the failures.

| qid | failed orders | order 1 | order 2 | both | non-ASCII characters in query |
|---|---|---|---|---|---|
| q52 | 13 | 4 | 9 | 1 | yes |
| q62 | 12 | 2 | 10 | 1 | no |
| q96 | 11 | 5 | 6 | 0 | no |
| q95 | 11 | 7 | 4 | 1 | no |
| q63 | 11 | 5 | 6 | 0 | yes |
| q33 | 9 | 2 | 7 | 1 | no |
| q91 | 9 | 0 | 9 | 0 | yes |
| q43 | 8 | 3 | 5 | 0 | no |
| q92 | 8 | 2 | 6 | 1 | no |

Twenty-two of the 100 generated queries contain a non-ASCII character (16 a non-breaking hyphen U+2011, 19 occurrences; 4 a curly apostrophe U+2019; 3 a narrow no-break space U+202F; 1 query carries two of these). Those queries fail at 3.24 percent of orders against 1.63 percent for plain ASCII queries (Fisher p = 2.3e-07 over calls; per-query Mann-Whitney p = 0.010, which is the honest test since calls within a query are not independent). They account for 94 of the 262 failures from 22 percent of the queries. Query form matters a little too ("how" and "does" questions fail at about 2.7 percent of orders, "what" questions at 1.2 percent). Query length correlates negatively with failures per query (Spearman -0.18, p = 0.08).

By model pair. Every one of the 66 pairs has at least one failure; the spread is what a uniform rate gives (variance over mean 0.94, maximum 9 of 200 orders). Nothing is pair-specific.

By model. Counting each failed order for both models it involves, the range is 28 (bm25s) to 54 (gte-small) of 2200 orders per model, variance over mean 1.45.

| model | failed orders | percent of its 2200 orders |
|---|---|---|
| gte-small | 54 | 2.45 |
| e5-base-v2 | 52 | 2.36 |
| mxbai-embed-large-v1 | 51 | 2.32 |
| gte-large | 49 | 2.23 |
| bge-base-en-v1.5 | 48 | 2.18 |
| bge-large-en-v1.5 | 45 | 2.05 |
| snowflake-arctic-embed-l-v2.0 | 43 | 1.95 |
| multilingual-e5-small | 43 | 1.95 |
| bge-small-en-v1.5 | 42 | 1.91 |
| all-MiniLM-L6-v2 | 35 | 1.59 |
| all-mpnet-base-v2 | 34 | 1.55 |
| baseline-bm25s | 28 | 1.27 |

Slot and roster. Which model is shown first is fixed by the roster (order 1 shows the roster-earlier model as System A, order 2 the roster-later one), so a model's System A rate in order 2 is confounded with the order 2 excess. Within the ten models that appear as System A in both orders, the order 2 rate is the higher one in eight. The design cannot separate "second call of the pair" from "roster-later model shown first".

| model | roster position | order 1 calls as System A | failed | order 2 calls as System A | failed |
|---|---|---|---|---|---|
| baseline-bm25s | 0 | 1100 | 14 | 0 | 0 |
| all-MiniLM-L6-v2 | 1 | 1000 | 10 | 100 | 3 |
| all-mpnet-base-v2 | 2 | 900 | 7 | 200 | 1 |
| bge-small-en-v1.5 | 3 | 800 | 13 | 300 | 3 |
| bge-base-en-v1.5 | 4 | 700 | 12 | 400 | 9 |
| bge-large-en-v1.5 | 5 | 599 | 5 | 500 | 7 |
| e5-base-v2 | 6 | 500 | 13 | 600 | 19 |
| gte-large | 7 | 400 | 6 | 700 | 21 |
| mxbai-embed-large-v1 | 8 | 300 | 3 | 799 | 22 |
| gte-small | 9 | 200 | 2 | 900 | 34 |
| snowflake-arctic-embed-l-v2.0 | 10 | 100 | 2 | 1000 | 22 |
| multilingual-e5-small | 11 | 0 | 0 | 1100 | 34 |

Completion order. `judge_pair_cached` writes a pair's .json sorted by query order once the pool has finished; the .jsonl twin is appended as each verdict completes and is the only file in completion order (all 66 .json files ascend in qid, 0 of the 66 .jsonl files do). Read in qid order, 9 pairs of neighbouring rows both failed against 9.1 expected from random placement (p = 0.56), which says nothing about time. Read in completion order, 74 neighbouring pairs both failed (0 of 2000 random placements reach that), the longest run is 5, and the failed rows per block of 10, 25 and 50 consecutive completions have variance over mean 3.35, 3.63 and 2.69 against a null 97.5th percentile of about 1.07. The structure is position rather than bursts. The failed rows are the last rows to complete in their file. Their completion-position deciles are 88, 90, 92, 94, 95, 96, 97, 98 and 99 of 100 (median 95, against 48 for clean rows); 211 sit in the last 10 positions, 246 in the last 15, 3 in the first half. Their input positions are spread over the file (qid index deciles 17 to 95), so these are rows started anywhere that finished last; the mean displacement from input position to completion position is +32.3 rows for failed rows against -1.3 for clean rows, with the 95th percentile of absolute displacement over all rows at 50. The one file judged partly in an earlier job, bm25s vs all-MiniLM-L6-v2, has its failures at positions 28, 42, 46, 81 and 96, the same signature over a shorter window. In the run log the wall time of a pair file rises with its failed-row count (Spearman 0.58, p = 6e-07, 64 pairs).

## Input length

The judge sees the query plus both top-10 lists with each document cut at 300 characters (`mteb_gym.judge._format`; the config carries no `doc_chars`). Most NFCorpus documents exceed 300 characters, so the input is close to constant, 5845 to 6265 characters, and almost all of the variation is the query. Failed rows are slightly shorter, not longer, median 6203 against 6206 characters (Mann-Whitney p = 7.5e-04), query median 83 against 86 characters. Within that 7 percent range longer inputs do not fail more; the run has no long inputs to speak about.

| input quartile | characters | rows | failed rows | percent |
|---|---|---|---|---|
| 1 | 5845 to 6196 | 1405 | 57 | 4.06 |
| 2 | 6196 to 6206 | 1818 | 88 | 4.84 |
| 3 | 6206 to 6225 | 1650 | 65 | 3.94 |
| 4 | 6225 to 6265 | 1726 | 40 | 2.32 |

## Overlap with the Qwen3.6-27B judge

The 27B judge failed 3 orders in 3 rows. One of those three rows (q10, bge-base-en-v1.5 vs gte-small) also failed under MiniMax; independence predicts 0.11 shared rows (Fisher p = 0.11). All three Qwen failure queries are among the 63 MiniMax failure queries, which is what a 63 percent base rate gives. Too few Qwen failures to say more.

| qid | pair | Qwen raw | MiniMax failed too | MiniMax raw |
|---|---|---|---|---|
| q10 | bge-base-en-v1.5 vs gte-small | ['tie', 'B'] | yes | ['tie', 'A'] |
| q78 | bge-large-en-v1.5 vs mxbai-embed-large-v1 | ['tie', 'tie'] | no | ['tie', 'tie'] |
| q89 | mxbai-embed-large-v1 vs all-mpnet-base-v2 | ['A', 'tie'] | no | ['A', 'A'] |

## Committed verdicts through the other order

203 of the 250 MiniMax failed rows (81.2 percent) ended with a committed score of 0.25 or 0.75 because the surviving order returned A or B; the surviving order was decisive in 85.3 percent of single failure rows against a decisive rate of 78.7 percent over all MiniMax orders, so a failed order does not mark a hard comparison. The score_a of failed rows splits 0.75 x 122, 0.25 x 81, 0.5 x 47 (12 double failures plus 35 rows whose surviving order said tie). For Qwen, 2 of 3 failed rows committed through the other order.

## Effect on the ranking

Refit with the control script's Bradley-Terry on the same rows.

| refit | MiniMax-M2.7 rho | Qwen3.6-27B rho |
|---|---|---|
| record, failed orders scored as ties | 0.8951 | 0.8182 |
| drop every row with a failed order (250 and 3 rows) | 0.8951 | 0.8182 |
| keep the surviving order alone, drop double failures (12 and 0 rows) | 0.8951 | 0.8182 |
| drop the same number of random clean rows (200 draws, mean and 2.5 to 97.5 band) | 0.8990 [0.8951, 0.9161] | 0.8182 [0.8180, 0.8182] |

The 12-model order is unchanged under both drops for both judges (Spearman between the record ratings and the dropped-row ratings 1.0). The largest rating movement is 5.8 points for MiniMax when the 250 rows are dropped and 3.3 when the surviving order is kept, against gaps between neighbouring models of 1.7 to 68 points; the one gap smaller than the shift (bge-small-en-v1.5 over e5-base-v2) did not swap. Scoring the failed orders as ties costs nothing measurable at this rate.

## Summary

The MiniMax-M2.7 judge left 262 of 13198 answers unparseable (1.99 percent), against 3 for Qwen3.6-27B on identical inputs. The failures are twice as common in the second presentation order, sit on 63 of the 100 queries with a query effect (queries carrying non-ASCII punctuation from the generator fail at twice the rate), are spread evenly over model pairs and nearly evenly over models, and are the last rows to complete in their pair file, which is consistent with replies cut at the 6144-token cap. Input length barely varies in this run (5845 to 6265 characters) and does not separate the failures. 81 percent of failed rows still produced a committed verdict through the other order. Dropping the failed rows or keeping only the surviving order leaves rho at 0.8951 and the ranking untouched.
