# Seed-label baseline: NanoSCIDOCSRetrieval (zeta-alpha-ai_NanoSCIDOCS_484eb90549fc3f0b9c42b3551e80ceb999515537_default_train-openai_gpt-oss-20b-5bcb5f0b7e3c)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | BAAI/bge-small-en-v1.5 | 0.4695 | 963.8 (9) | 43.04 (5) |
| 2 | BAAI/bge-base-en-v1.5 | 0.4693 | 1063.2 (3) | 39.08 (9) |
| 3 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.4693 | 966.5 (8) | 41.42 (8) |
| 4 | intfloat/multilingual-e5-small | 0.4691 | 913.0 (11) | 34.38 (11) |
| 5 | mteb/baseline-bm25s | 0.4652 | 822.0 (12) | 33.51 (12) |
| 6 | sentence-transformers/all-MiniLM-L6-v2 | 0.4649 | 1059.7 (4) | 43.28 (4) |
| 7 | intfloat/e5-base-v2 | 0.4649 | 970.4 (7) | 38.09 (10) |
| 8 | mixedbread-ai/mxbai-embed-large-v1 | 0.4612 | 1149.8 (1) | 44.82 (3) |
| 9 | BAAI/bge-large-en-v1.5 | 0.4578 | 1118.3 (2) | 42.98 (6) |
| 10 | thenlper/gte-large | 0.4532 | 1034.0 (5) | 47.36 (1) |
| 11 | thenlper/gte-small | 0.4474 | 920.7 (10) | 42.68 (7) |
| 12 | sentence-transformers/all-mpnet-base-v2 | 0.4430 | 1018.7 (6) | 45.50 (2) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman -0.260 (p=0.42), Kendall -0.154 (p=0.49), n=12
- seed vs official: Spearman -0.530 (p=0.08), Kendall -0.339 (p=0.13), n=12
- gym vs official: Spearman 0.573 (p=0.05), Kendall 0.424 (p=0.06), n=12
- official ranking source: --truth analysis/NanoSCIDOCSRetrieval/truth.json

-> analysis/NanoSCIDOCSRetrieval/seed_baseline.json, analysis/NanoSCIDOCSRetrieval/seed_baseline.md
