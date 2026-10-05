# Seed-label baseline: NFCorpus (mteb_nfcorpus_ec0fa4fe99da2ff19ca1214b7966684033a58814_default_test-openai_gpt-oss-20b-6e3b1f28fe89)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.3964 | 1025.0 (6) | 35.08 (6) |
| 2 | thenlper/gte-large | 0.3911 | 1091.4 (2) | 38.17 (2) |
| 3 | thenlper/gte-small | 0.3899 | 1033.2 (5) | 34.84 (7) |
| 4 | BAAI/bge-base-en-v1.5 | 0.3897 | 1043.6 (4) | 37.37 (4) |
| 5 | intfloat/e5-base-v2 | 0.3894 | 998.9 (9) | 35.42 (5) |
| 6 | BAAI/bge-large-en-v1.5 | 0.3866 | 1054.5 (3) | 38.06 (3) |
| 7 | mixedbread-ai/mxbai-embed-large-v1 | 0.3866 | 1098.3 (1) | 38.67 (1) |
| 8 | BAAI/bge-small-en-v1.5 | 0.3836 | 1000.6 (8) | 34.26 (8) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.3813 | 1016.0 (7) | 33.29 (9) |
| 10 | intfloat/multilingual-e5-small | 0.3725 | 884.4 (11) | 31.10 (12) |
| 11 | sentence-transformers/all-MiniLM-L6-v2 | 0.3713 | 931.3 (10) | 31.59 (11) |
| 12 | mteb/baseline-bm25s | 0.3448 | 822.9 (12) | 32.10 (10) |

- coverage: 100/100 queries have a seed document in some model's top 10
- seed vs gym: Spearman 0.664 (p=0.02), Kendall 0.455 (p=0.04), n=12
- seed vs official: Spearman 0.671 (p=0.02), Kendall 0.455 (p=0.04), n=12
- gym vs official: Spearman 0.895 (p=0.00), Kendall 0.758 (p=0.00), n=12
- official ranking source: --truth <RUNS>/analysis_out/NFCorpus_m27/truth.json

-> <RUNS>/analysis_out/NFCorpus_m27/seed_baseline.json, <RUNS>/analysis_out/NFCorpus_m27/seed_baseline.md
