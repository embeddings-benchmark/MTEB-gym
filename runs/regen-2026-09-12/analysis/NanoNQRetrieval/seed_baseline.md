# Seed-label baseline: NanoNQRetrieval (zeta-alpha-ai_NanoNQ_77540146379abf95df8326a3c5bb9eb21c7146c3_default_train-openai_gpt-oss-20b-94c953488244)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | thenlper/gte-large | 0.4489 | 1048.9 (1) | 71.04 (3) |
| 2 | thenlper/gte-small | 0.4489 | 1022.1 (3) | 70.55 (5) |
| 3 | BAAI/bge-large-en-v1.5 | 0.4459 | 1009.3 (6) | 71.34 (2) |
| 4 | intfloat/e5-base-v2 | 0.4457 | 1035.2 (2) | 66.81 (8) |
| 5 | BAAI/bge-small-en-v1.5 | 0.4430 | 1009.3 (7) | 59.34 (10) |
| 6 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4399 | 1020.2 (4) | 73.03 (1) |
| 7 | mixedbread-ai/mxbai-embed-large-v1 | 0.4398 | 1008.9 (8) | 70.70 (4) |
| 8 | BAAI/bge-base-en-v1.5 | 0.4364 | 1013.7 (5) | 66.72 (9) |
| 9 | intfloat/multilingual-e5-small | 0.4357 | 972.8 (10) | 68.62 (6) |
| 10 | sentence-transformers/all-mpnet-base-v2 | 0.4169 | 993.6 (9) | 67.34 (7) |
| 11 | sentence-transformers/all-MiniLM-L6-v2 | 0.4066 | 962.9 (11) | 59.04 (11) |
| 12 | mteb/baseline-bm25s | 0.3867 | 903.1 (12) | 50.07 (12) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.876 (p=0.00), Kendall 0.748 (p=0.00), n=12
- seed vs official: Spearman 0.616 (p=0.03), Kendall 0.473 (p=0.03), n=12
- gym vs official: Spearman 0.545 (p=0.07), Kendall 0.424 (p=0.06), n=12
- official ranking source: --truth analysis/NanoNQRetrieval/truth.json

-> analysis/NanoNQRetrieval/seed_baseline.json, analysis/NanoNQRetrieval/seed_baseline.md
