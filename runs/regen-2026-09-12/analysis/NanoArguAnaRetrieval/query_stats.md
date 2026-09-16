| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 15.025 | 193.000 |
| median length (words) | 14.000 | 188.000 |
| question share | 0.900 | 0.060 |
| copied-word share, mean per query | 0.535 | null |
| copied-word share, pooled | 0.529 | null |
| quality score, mean | 4.850 | null |
| quality score counts | {"4": 6, "5": 34} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoArguAnaRetrieval/query_stats.json and analysis/NanoArguAnaRetrieval/query_stats.md
