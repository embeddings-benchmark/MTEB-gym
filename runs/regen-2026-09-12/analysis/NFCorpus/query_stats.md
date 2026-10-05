| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 100 | 323 |
| mean length (words) | 13.880 | 3.294 |
| median length (words) | 14.000 | 2 |
| question share | 1.000 | 0.142 |
| copied-word share, mean per query | 0.630 | null |
| copied-word share, pooled | 0.628 | null |
| quality score, mean | 5.000 | null |
| quality score counts | {"5": 100} | null |
| n_generated | 160 | null |
| n_dropped | 60 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NFCorpus/query_stats.json and analysis/NFCorpus/query_stats.md
