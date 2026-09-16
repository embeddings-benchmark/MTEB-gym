# Seed-label baseline: NanoMSMARCORetrieval (zeta-alpha-ai_NanoMSMARCO_7b8ff22f2771dc65ac5b439f222eb19a1f56abda_default_train-openai_gpt-oss-20b-31a7f6478de0)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.3835 | 1051.9 (2) | 67.81 (1) |
| 2 | mixedbread-ai/mxbai-embed-large-v1 | 0.3687 | 1068.2 (1) | 66.07 (3) |
| 3 | BAAI/bge-large-en-v1.5 | 0.3673 | 1051.1 (3) | 67.77 (2) |
| 4 | thenlper/gte-large | 0.3672 | 1041.0 (4) | 59.67 (10) |
| 5 | sentence-transformers/all-MiniLM-L6-v2 | 0.3612 | 959.6 (11) | 55.40 (11) |
| 6 | BAAI/bge-base-en-v1.5 | 0.3588 | 1023.1 (5) | 63.61 (4) |
| 7 | thenlper/gte-small | 0.3578 | 1012.4 (6) | 63.38 (6) |
| 8 | intfloat/e5-base-v2 | 0.3569 | 996.1 (8) | 61.22 (9) |
| 9 | intfloat/multilingual-e5-small | 0.3508 | 977.2 (10) | 62.09 (8) |
| 10 | sentence-transformers/all-mpnet-base-v2 | 0.3471 | 1007.2 (7) | 63.45 (5) |
| 11 | BAAI/bge-small-en-v1.5 | 0.3422 | 980.9 (9) | 63.17 (7) |
| 12 | mteb/baseline-bm25s | 0.3325 | 831.3 (12) | 53.30 (12) |

- coverage: 38/40 queries have a seed document in some model's top 10 (uncovered: q29, q33)
- seed vs gym: Spearman 0.811 (p=0.00), Kendall 0.697 (p=0.00), n=12
- seed vs official: Spearman 0.573 (p=0.05), Kendall 0.424 (p=0.06), n=12
- gym vs official: Spearman 0.804 (p=0.00), Kendall 0.667 (p=0.00), n=12
- official ranking source: --truth analysis/NanoMSMARCORetrieval/truth.json

-> analysis/NanoMSMARCORetrieval/seed_baseline.json, analysis/NanoMSMARCORetrieval/seed_baseline.md
