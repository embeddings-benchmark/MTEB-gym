# Seed-label baseline: NanoTouche2020Retrieval (zeta-alpha-ai_NanoTouche2020_0d2f26ed8c5ad309f95c7f9499c70a40e140fccd_default_train-openai_gpt-oss-20b-31031965e09c)

| Rank | Model | Seed nDCG@10 | Gym rating (rank) | Official (rank) |
|---:|---|---:|---:|---:|
| 1 | thenlper/gte-large | 0.2921 | 1028.5 (3) | 54.54 (3) |
| 2 | mixedbread-ai/mxbai-embed-large-v1 | 0.2840 | 967.5 (10) | 53.20 (5) |
| 3 | BAAI/bge-large-en-v1.5 | 0.2790 | 955.8 (11) | 53.19 (6) |
| 4 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.2774 | 1001.5 (7) | 53.31 (4) |
| 5 | BAAI/bge-base-en-v1.5 | 0.2749 | 1006.2 (6) | 51.77 (7) |
| 6 | thenlper/gte-small | 0.2631 | 984.7 (8) | 51.68 (8) |
| 7 | sentence-transformers/all-MiniLM-L6-v2 | 0.2575 | 1011.0 (5) | 47.48 (12) |
| 8 | BAAI/bge-small-en-v1.5 | 0.2474 | 1049.1 (2) | 55.07 (2) |
| 9 | sentence-transformers/all-mpnet-base-v2 | 0.2210 | 1055.8 (1) | 48.48 (11) |
| 10 | intfloat/multilingual-e5-small | 0.2075 | 984.4 (9) | 49.54 (9) |
| 11 | intfloat/e5-base-v2 | 0.2050 | 1027.4 (4) | 49.21 (10) |
| 12 | mteb/baseline-bm25s | 0.1850 | 928.1 (12) | 69.53 (1) |

- coverage: 38/40 queries have a seed document in some model's top 10 (uncovered: q14, q21)
- seed vs gym: Spearman -0.049 (p=0.88), Kendall -0.091 (p=0.74), n=12
- seed vs official: Spearman 0.238 (p=0.46), Kendall 0.242 (p=0.31), n=12
- gym vs official: Spearman -0.280 (p=0.38), Kendall -0.182 (p=0.46), n=12
- official ranking source: --truth analysis/NanoTouche2020Retrieval/truth.json

-> analysis/NanoTouche2020Retrieval/seed_baseline.json, analysis/NanoTouche2020Retrieval/seed_baseline.md
