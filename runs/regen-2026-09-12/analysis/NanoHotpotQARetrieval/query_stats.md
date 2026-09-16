| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 15.450 | 14.920 |
| median length (words) | 14.000 | 15.000 |
| question share | 0.975 | 0.980 |
| copied-word share, mean per query | 0.688 | null |
| copied-word share, pooled | 0.687 | null |
| quality score, mean | 4.925 | null |
| quality score counts | {"3": 1, "4": 1, "5": 38} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoHotpotQARetrieval/query_stats.json and analysis/NanoHotpotQARetrieval/query_stats.md
