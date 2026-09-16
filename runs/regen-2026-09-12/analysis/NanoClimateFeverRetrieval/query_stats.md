| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 14.025 | 21.360 |
| median length (words) | 14.000 | 20.000 |
| question share | 1.000 | 0.020 |
| copied-word share, mean per query | 0.454 | null |
| copied-word share, pooled | 0.461 | null |
| quality score, mean | 4.850 | null |
| quality score counts | {"3": 1, "4": 4, "5": 35} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoClimateFeverRetrieval/query_stats.json and analysis/NanoClimateFeverRetrieval/query_stats.md
