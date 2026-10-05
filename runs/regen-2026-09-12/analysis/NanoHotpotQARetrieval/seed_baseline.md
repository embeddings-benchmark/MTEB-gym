# Seed-label baseline: NanoHotpotQARetrieval (zeta-alpha-ai_NanoHotpotQA_d79c0cdda980aba54842756770928035e1b61a51_default_train-openai_gpt-oss-20b-e814e76e504f)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | BAAI/bge-large-en-v1.5 | 0.4974 | 1036.9 (2) | 90.17 (1) |
| 2 | intfloat/e5-base-v2 | 0.4937 | 1048.8 (1) | 83.03 (5) |
| 3 | mixedbread-ai/mxbai-embed-large-v1 | 0.4925 | 1035.8 (3) | 87.19 (2) |
| 4 | thenlper/gte-large | 0.4844 | 1019.4 (4) | 80.81 (8) |
| 5 | BAAI/bge-small-en-v1.5 | 0.4760 | 1007.7 (5) | 83.97 (4) |
| 6 | BAAI/bge-base-en-v1.5 | 0.4740 | 1005.2 (6) | 86.79 (3) |
| 7 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4708 | 1002.3 (7) | 77.20 (10) |
| 8 | thenlper/gte-small | 0.4699 | 996.1 (9) | 78.54 (9) |
| 9 | mteb/baseline-bm25s | 0.4671 | 980.4 (10) | 82.77 (6) |
| 10 | intfloat/multilingual-e5-small | 0.4504 | 996.4 (8) | 81.01 (7) |
| 11 | sentence-transformers/all-MiniLM-L6-v2 | 0.4403 | 925.1 (12) | 59.60 (12) |
| 12 | sentence-transformers/all-mpnet-base-v2 | 0.4285 | 945.8 (11) | 62.65 (11) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.965 (p=0.00), Kendall 0.879 (p=0.00), n=12
- seed vs official: Spearman 0.769 (p=0.00), Kendall 0.576 (p=0.01), n=12
- gym vs official: Spearman 0.755 (p=0.00), Kendall 0.576 (p=0.01), n=12
- official ranking source: --truth analysis/NanoHotpotQARetrieval/truth.json

-> analysis/NanoHotpotQARetrieval/seed_baseline.json, analysis/NanoHotpotQARetrieval/seed_baseline.md
