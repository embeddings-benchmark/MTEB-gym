# Seed-label baseline: NanoDBPediaRetrieval (zeta-alpha-ai_NanoDBPedia_438f1c25129f05db6238699b5afdc9c6b58d2096_default_train-openai_gpt-oss-20b-071fb2782138)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | intfloat/e5-base-v2 | 0.4548 | 1030.0 (3) | 64.48 (2) |
| 2 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4539 | 1019.0 (5) | 66.33 (1) |
| 3 | BAAI/bge-base-en-v1.5 | 0.4494 | 1025.6 (4) | 60.00 (9) |
| 4 | BAAI/bge-large-en-v1.5 | 0.4489 | 1033.3 (2) | 61.34 (5) |
| 5 | mixedbread-ai/mxbai-embed-large-v1 | 0.4473 | 1063.0 (1) | 63.56 (3) |
| 6 | intfloat/multilingual-e5-small | 0.4446 | 984.0 (8) | 60.54 (7) |
| 7 | BAAI/bge-small-en-v1.5 | 0.4437 | 1016.4 (7) | 61.24 (6) |
| 8 | mteb/baseline-bm25s | 0.4404 | 958.3 (9) | 61.38 (4) |
| 9 | thenlper/gte-large | 0.4315 | 1018.3 (6) | 60.17 (8) |
| 10 | sentence-transformers/all-MiniLM-L6-v2 | 0.4138 | 948.3 (11) | 55.01 (11) |
| 11 | thenlper/gte-small | 0.4110 | 947.5 (12) | 57.77 (10) |
| 12 | sentence-transformers/all-mpnet-base-v2 | 0.4038 | 956.4 (10) | 53.84 (12) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.811 (p=0.00), Kendall 0.576 (p=0.01), n=12
- seed vs official: Spearman 0.776 (p=0.00), Kendall 0.606 (p=0.01), n=12
- gym vs official: Spearman 0.671 (p=0.02), Kendall 0.424 (p=0.06), n=12
- official ranking source: --truth analysis/NanoDBPediaRetrieval/truth.json

-> analysis/NanoDBPediaRetrieval/seed_baseline.json, analysis/NanoDBPediaRetrieval/seed_baseline.md
