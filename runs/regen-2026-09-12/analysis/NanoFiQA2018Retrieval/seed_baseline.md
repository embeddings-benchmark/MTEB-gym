# Seed-label baseline: NanoFiQA2018Retrieval (zeta-alpha-ai_NanoFiQA2018_4163ba032953d5044a7a6244261413f609c14342_default_train-openai_gpt-oss-20b-af75bf930fd3)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.3728 | 1102.8 (1) | 54.31 (4) |
| 2 | BAAI/bge-base-en-v1.5 | 0.3413 | 1040.6 (3) | 46.05 (9) |
| 3 | BAAI/bge-large-en-v1.5 | 0.3399 | 1022.6 (7) | 56.55 (2) |
| 4 | thenlper/gte-large | 0.3390 | 1027.1 (5) | 54.00 (5) |
| 5 | thenlper/gte-small | 0.3320 | 1048.5 (2) | 49.16 (6) |
| 6 | BAAI/bge-small-en-v1.5 | 0.3231 | 996.3 (8) | 48.84 (7) |
| 7 | mixedbread-ai/mxbai-embed-large-v1 | 0.3159 | 1036.1 (4) | 56.21 (3) |
| 8 | intfloat/e5-base-v2 | 0.3055 | 974.3 (9) | 46.03 (10) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.2998 | 1026.7 (6) | 58.81 (1) |
| 10 | sentence-transformers/all-MiniLM-L6-v2 | 0.2985 | 944.6 (10) | 47.74 (8) |
| 11 | intfloat/multilingual-e5-small | 0.2910 | 900.2 (11) | 43.53 (12) |
| 12 | mteb/baseline-bm25s | 0.2883 | 880.2 (12) | 43.74 (11) |

- coverage: 39/40 queries have a seed document in some model's top 10 (uncovered: q34)
- seed vs gym: Spearman 0.825 (p=0.00), Kendall 0.697 (p=0.00), n=12
- seed vs official: Spearman 0.469 (p=0.12), Kendall 0.364 (p=0.12), n=12
- gym vs official: Spearman 0.580 (p=0.05), Kendall 0.424 (p=0.06), n=12
- official ranking source: --truth analysis/NanoFiQA2018Retrieval/truth.json

-> analysis/NanoFiQA2018Retrieval/seed_baseline.json, analysis/NanoFiQA2018Retrieval/seed_baseline.md
