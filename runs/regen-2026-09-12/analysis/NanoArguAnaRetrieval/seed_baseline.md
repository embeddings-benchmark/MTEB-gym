# Seed-label baseline: NanoArguAnaRetrieval (zeta-alpha-ai_NanoArguAna_8f4a982d470a32c45817738b9d29042ca55d75ad_default_train-openai_gpt-oss-20b-08d2d5c1f6c2)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4236 | 1071.8 (2) | 65.64 (3) |
| 2 | BAAI/bge-large-en-v1.5 | 0.4072 | 1015.6 (7) | 66.09 (2) |
| 3 | mixedbread-ai/mxbai-embed-large-v1 | 0.3998 | 1051.3 (4) | 66.72 (1) |
| 4 | BAAI/bge-base-en-v1.5 | 0.3973 | 1031.5 (6) | 64.02 (4) |
| 5 | thenlper/gte-large | 0.3952 | 1073.7 (1) | 58.11 (8) |
| 6 | BAAI/bge-small-en-v1.5 | 0.3924 | 979.9 (9) | 62.51 (5) |
| 7 | thenlper/gte-small | 0.3863 | 1042.2 (5) | 60.79 (6) |
| 8 | intfloat/e5-base-v2 | 0.3745 | 964.6 (10) | 60.68 (7) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.3635 | 1070.2 (3) | 52.33 (10) |
| 10 | intfloat/multilingual-e5-small | 0.3517 | 929.1 (11) | 44.54 (12) |
| 11 | mteb/baseline-bm25s | 0.3277 | 756.3 (12) | 48.04 (11) |
| 12 | sentence-transformers/all-MiniLM-L6-v2 | 0.3222 | 1013.7 (8) | 54.87 (9) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.587 (p=0.04), Kendall 0.424 (p=0.06), n=12
- seed vs official: Spearman 0.881 (p=0.00), Kendall 0.697 (p=0.00), n=12
- gym vs official: Spearman 0.420 (p=0.17), Kendall 0.303 (p=0.20), n=12
- official ranking source: --truth analysis/NanoArguAnaRetrieval/truth.json

-> analysis/NanoArguAnaRetrieval/seed_baseline.json, analysis/NanoArguAnaRetrieval/seed_baseline.md
