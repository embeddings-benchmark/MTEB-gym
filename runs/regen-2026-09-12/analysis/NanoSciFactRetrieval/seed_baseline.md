# Seed-label baseline: NanoSciFactRetrieval (zeta-alpha-ai_NanoSciFact_309f1d1ae3ae2e092444a8a0c25bed59b82318bc_default_train-openai_gpt-oss-20b-be18a1957320)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | intfloat/e5-base-v2 | 0.4649 | 1047.1 (4) | 73.93 (7) |
| 2 | thenlper/gte-small | 0.4649 | 1028.9 (5) | 79.61 (2) |
| 3 | mixedbread-ai/mxbai-embed-large-v1 | 0.4649 | 1070.2 (1) | 78.83 (3) |
| 4 | thenlper/gte-large | 0.4649 | 1054.2 (3) | 78.65 (4) |
| 5 | BAAI/bge-base-en-v1.5 | 0.4649 | 1024.8 (6) | 81.72 (1) |
| 6 | BAAI/bge-large-en-v1.5 | 0.4634 | 1060.6 (2) | 78.22 (5) |
| 7 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4622 | 971.3 (7) | 73.51 (8) |
| 8 | BAAI/bge-small-en-v1.5 | 0.4621 | 968.7 (8) | 76.23 (6) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.4606 | 968.0 (9) | 69.67 (12) |
| 10 | intfloat/multilingual-e5-small | 0.4559 | 950.1 (11) | 72.46 (10) |
| 11 | mteb/baseline-bm25s | 0.4532 | 901.6 (12) | 70.99 (11) |
| 12 | sentence-transformers/all-MiniLM-L6-v2 | 0.4448 | 954.6 (10) | 72.65 (9) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.808 (p=0.00), Kendall 0.657 (p=0.00), n=12
- seed vs official: Spearman 0.773 (p=0.00), Kendall 0.594 (p=0.01), n=12
- gym vs official: Spearman 0.741 (p=0.01), Kendall 0.545 (p=0.01), n=12
- official ranking source: --truth analysis/NanoSciFactRetrieval/truth.json

-> analysis/NanoSciFactRetrieval/seed_baseline.json, analysis/NanoSciFactRetrieval/seed_baseline.md
