# Seed-label baseline: NanoFEVERRetrieval (zeta-alpha-ai_NanoFEVER_a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8_default_train-openai_gpt-oss-20b-2096e73b8085)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | mixedbread-ai/mxbai-embed-large-v1 | 0.4531 | 1078.2 (1) | 92.28 (5) |
| 2 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4448 | 961.8 (10) | 94.02 (3) |
| 3 | BAAI/bge-large-en-v1.5 | 0.4446 | 1043.2 (3) | 93.04 (4) |
| 4 | intfloat/e5-base-v2 | 0.4432 | 1017.5 (5) | 89.77 (8) |
| 5 | BAAI/bge-base-en-v1.5 | 0.4374 | 1027.4 (4) | 94.15 (2) |
| 6 | thenlper/gte-large | 0.4372 | 1051.8 (2) | 90.99 (6) |
| 7 | intfloat/multilingual-e5-small | 0.4305 | 972.8 (8) | 83.49 (9) |
| 8 | thenlper/gte-small | 0.4255 | 998.1 (6) | 90.03 (7) |
| 9 | mteb/baseline-bm25s | 0.4246 | 941.7 (12) | 80.94 (10) |
| 10 | sentence-transformers/all-MiniLM-L6-v2 | 0.4218 | 960.7 (11) | 79.33 (11) |
| 11 | BAAI/bge-small-en-v1.5 | 0.4161 | 980.9 (7) | 94.20 (1) |
| 12 | sentence-transformers/all-mpnet-base-v2 | 0.4082 | 965.9 (9) | 75.87 (12) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.573 (p=0.05), Kendall 0.424 (p=0.06), n=12
- seed vs official: Spearman 0.476 (p=0.12), Kendall 0.424 (p=0.06), n=12
- gym vs official: Spearman 0.490 (p=0.11), Kendall 0.333 (p=0.15), n=12
- official ranking source: --truth analysis/NanoFEVERRetrieval/truth.json

-> analysis/NanoFEVERRetrieval/seed_baseline.json, analysis/NanoFEVERRetrieval/seed_baseline.md
