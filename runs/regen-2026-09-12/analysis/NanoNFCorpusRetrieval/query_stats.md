| metric | synthetic (openai/gpt-oss-20b) | original |
|---|---|---|
| queries | 40 | 50 |
| mean length (words) | 13.625 | 3.260 |
| median length (words) | 12.000 | 2.000 |
| question share | 1.000 | 0.160 |
| copied-word share, mean per query | 0.624 | null |
| copied-word share, pooled | 0.617 | null |
| quality score, mean | 4.975 | null |
| quality score counts | {"4": 1, "5": 39} | null |
| n_generated | 64 | null |
| n_dropped | 24 | null |
| drop rate | 0.375 | null |

n_generated counts queries that passed the length and degeneracy heuristics during generation; the package does not record why a query was dropped after that, so the drop count folds together the quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to n_queries. Heuristic rejections during generation are not counted anywhere.
-> analysis/NanoNFCorpusRetrieval/query_stats.json and analysis/NanoNFCorpusRetrieval/query_stats.md
