| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 18.375 | 9.440 |
| median length (words) | 18.000 | 9.000 |
| question share | 1.000 | 0.060 |
| copied-word share, mean per query | 0.595 | null |
| copied-word share, pooled | 0.602 | null |
| quality score, mean | 4.950 | null |
| quality score counts | {"4": 2, "5": 38} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoSCIDOCSRetrieval/query_stats.json and analysis/NanoSCIDOCSRetrieval/query_stats.md
