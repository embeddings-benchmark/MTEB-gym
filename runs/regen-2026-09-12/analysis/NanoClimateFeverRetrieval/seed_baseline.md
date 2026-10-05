# Seed-label baseline: NanoClimateFeverRetrieval (zeta-alpha-ai_NanoClimateFEVER_96741bfa30b9f56db8c9eb7d08e775ed6474f206_default_train-openai_gpt-oss-20b-26ba590d02f6)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.2594 | 994.1 (7) | 41.17 (2) |
| 2 | thenlper/gte-large | 0.2494 | 1114.9 (1) | 37.70 (5) |
| 3 | intfloat/multilingual-e5-small | 0.2394 | 991.8 (8) | 30.64 (9) |
| 4 | mteb/baseline-bm25s | 0.2320 | 792.9 (12) | 31.77 (8) |
| 5 | thenlper/gte-small | 0.2311 | 1052.1 (4) | 26.75 (11) |
| 6 | mixedbread-ai/mxbai-embed-large-v1 | 0.2244 | 1102.6 (2) | 39.81 (4) |
| 7 | BAAI/bge-small-en-v1.5 | 0.2171 | 975.8 (10) | 34.76 (6) |
| 8 | BAAI/bge-large-en-v1.5 | 0.2157 | 1012.3 (5) | 43.15 (1) |
| 9 | intfloat/e5-base-v2 | 0.2142 | 978.1 (9) | 32.72 (7) |
| 10 | sentence-transformers/all-MiniLM-L6-v2 | 0.2130 | 906.5 (11) | 29.58 (10) |
| 11 | BAAI/bge-base-en-v1.5 | 0.2060 | 999.8 (6) | 40.01 (3) |
| 12 | sentence-transformers/all-mpnet-base-v2 | 0.1934 | 1079.0 (3) | 26.18 (12) |

- coverage: 30/40 queries have a seed document in some model's top 10 (uncovered: q10, q14, q17, q19, q2, q24, q28, q31, q5, q6)
- seed vs gym: Spearman 0.063 (p=0.85), Kendall 0.061 (p=0.84), n=12
- seed vs official: Spearman 0.231 (p=0.47), Kendall 0.182 (p=0.46), n=12
- gym vs official: Spearman 0.168 (p=0.60), Kendall 0.091 (p=0.74), n=12
- official ranking source: --truth analysis/NanoClimateFeverRetrieval/truth.json

-> analysis/NanoClimateFeverRetrieval/seed_baseline.json, analysis/NanoClimateFeverRetrieval/seed_baseline.md
