# Seed-label baseline: NanoNFCorpusRetrieval (zeta-alpha-ai_NanoNFCorpus_dd542a7efb9ad2136b9e00768b60fca9038f8156_default_train-openai_gpt-oss-20b-91edcd46e609)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | thenlper/gte-small | 0.4006 | 1055.3 (3) | 35.26 (7) |
| 2 | sentence-transformers/all-MiniLM-L6-v2 | 0.4006 | 944.3 (10) | 33.22 (9) |
| 3 | BAAI/bge-base-en-v1.5 | 0.3974 | 1021.4 (7) | 39.28 (1) |
| 4 | sentence-transformers/all-mpnet-base-v2 | 0.3966 | 1047.7 (5) | 31.80 (11) |
| 5 | intfloat/e5-base-v2 | 0.3942 | 1020.3 (8) | 35.32 (6) |
| 6 | BAAI/bge-small-en-v1.5 | 0.3926 | 967.0 (9) | 35.84 (5) |
| 7 | thenlper/gte-large | 0.3913 | 1091.6 (1) | 34.62 (8) |
| 8 | mixedbread-ai/mxbai-embed-large-v1 | 0.3880 | 1060.7 (2) | 38.52 (2) |
| 9 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.3852 | 1037.0 (6) | 38.51 (3) |
| 10 | BAAI/bge-large-en-v1.5 | 0.3780 | 1051.1 (4) | 37.01 (4) |
| 11 | intfloat/multilingual-e5-small | 0.3771 | 892.4 (11) | 28.82 (12) |
| 12 | mteb/baseline-bm25s | 0.3625 | 811.2 (12) | 32.50 (10) |

- coverage: 40/40 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.207 (p=0.52), Kendall 0.168 (p=0.45), n=12
- seed vs official: Spearman 0.105 (p=0.75), Kendall 0.046 (p=0.84), n=12
- gym vs official: Spearman 0.399 (p=0.20), Kendall 0.273 (p=0.25), n=12
- official ranking source: --truth analysis/NanoNFCorpusRetrieval/truth.json

-> analysis/NanoNFCorpusRetrieval/seed_baseline.json, analysis/NanoNFCorpusRetrieval/seed_baseline.md
