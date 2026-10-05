# Seed-label baseline: NanoQuoraRetrieval (zeta-alpha-ai_NanoQuoraRetrieval_2ab2d73e6c862026282808b913a34f4136928545_default_train-openai_gpt-oss-20b-00c675d7aece)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | thenlper/gte-large | 0.4255 | 1070.8 (2) | 92.73 (9) |
| 2 | BAAI/bge-large-en-v1.5 | 0.4170 | 1071.5 (1) | 96.01 (4) |
| 3 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4107 | 1060.7 (3) | 96.86 (2) |
| 4 | mixedbread-ai/mxbai-embed-large-v1 | 0.4084 | 1053.8 (4) | 95.55 (6) |
| 5 | BAAI/bge-base-en-v1.5 | 0.4061 | 1029.3 (6) | 95.67 (5) |
| 6 | sentence-transformers/all-mpnet-base-v2 | 0.4040 | 1053.8 (5) | 92.14 (10) |
| 7 | intfloat/e5-base-v2 | 0.3988 | 994.0 (9) | 91.79 (11) |
| 8 | BAAI/bge-small-en-v1.5 | 0.3938 | 1008.8 (7) | 96.18 (3) |
| 9 | thenlper/gte-small | 0.3933 | 1002.0 (8) | 93.03 (8) |
| 10 | sentence-transformers/all-MiniLM-L6-v2 | 0.3704 | 991.0 (10) | 93.68 (7) |
| 11 | intfloat/multilingual-e5-small | 0.3678 | 931.3 (11) | 97.28 (1) |
| 12 | mteb/baseline-bm25s | 0.2743 | 733.1 (12) | 86.28 (12) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.965 (p=0.00), Kendall 0.879 (p=0.00), n=12
- seed vs official: Spearman 0.161 (p=0.62), Kendall 0.121 (p=0.64), n=12
- gym vs official: Spearman 0.238 (p=0.46), Kendall 0.182 (p=0.46), n=12
- official ranking source: --truth analysis/NanoQuoraRetrieval/truth.json

-> analysis/NanoQuoraRetrieval/seed_baseline.json, analysis/NanoQuoraRetrieval/seed_baseline.md
