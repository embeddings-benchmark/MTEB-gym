# Seed-label baseline: NFCorpus (mteb_nfcorpus_ec0fa4fe99da2ff19ca1214b7966684033a58814_default_test-openai_gpt-oss-20b-6e3b1f28fe89)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.3964 | 1041.8 (5) | 35.08 (6) |
| 2 | thenlper/gte-large | 0.3911 | 1095.5 (2) | 38.17 (2) |
| 3 | thenlper/gte-small | 0.3899 | 1048.9 (3) | 34.84 (7) |
| 4 | BAAI/bge-base-en-v1.5 | 0.3897 | 1046.5 (4) | 37.37 (4) |
| 5 | intfloat/e5-base-v2 | 0.3894 | 990.2 (9) | 35.42 (5) |
| 6 | BAAI/bge-large-en-v1.5 | 0.3866 | 1041.4 (6) | 38.06 (3) |
| 7 | mixedbread-ai/mxbai-embed-large-v1 | 0.3866 | 1096.1 (1) | 38.67 (1) |
| 8 | BAAI/bge-small-en-v1.5 | 0.3836 | 1011.5 (8) | 34.26 (8) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.3813 | 1030.3 (7) | 33.29 (9) |
| 10 | intfloat/multilingual-e5-small | 0.3725 | 866.5 (11) | 31.10 (12) |
| 11 | sentence-transformers/all-MiniLM-L6-v2 | 0.3713 | 925.1 (10) | 31.59 (11) |
| 12 | mteb/baseline-bm25s | 0.3448 | 806.2 (12) | 32.10 (10) |

- coverage: 100/100 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.741 (p=0.01), Kendall 0.576 (p=0.01), n=12
- seed vs official: Spearman 0.671 (p=0.02), Kendall 0.455 (p=0.04), n=12
- gym vs official: Spearman 0.818 (p=0.00), Kendall 0.636 (p=0.00), n=12
- official ranking source: --truth analysis/NFCorpus/truth.json

-> analysis/NFCorpus/seed_baseline.json, analysis/NFCorpus/seed_baseline.md
