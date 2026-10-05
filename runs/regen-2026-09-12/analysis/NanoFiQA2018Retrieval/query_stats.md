| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 17.200 | 10.200 |
| median length (words) | 16.000 | 10.000 |
| question share | 1.000 | 0.740 |
| copied-word share, mean per query | 0.571 | null |
| copied-word share, pooled | 0.575 | null |
| quality score, mean | 5.000 | null |
| quality score counts | {"5": 40} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoFiQA2018Retrieval/query_stats.json and analysis/NanoFiQA2018Retrieval/query_stats.md
