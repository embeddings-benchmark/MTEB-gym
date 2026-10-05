| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 11.225 | 9.040 |
| median length (words) | 11.000 | 8.000 |
| question share | 0.975 | 1.000 |
| copied-word share, mean per query | 0.364 | null |
| copied-word share, pooled | 0.372 | null |
| quality score, mean | 4.525 | null |
| quality score counts | {"3": 2, "4": 15, "5": 23} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoQuoraRetrieval/query_stats.json and analysis/NanoQuoraRetrieval/query_stats.md
