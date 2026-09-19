# MTEB-Gym regeneration sweep, cross-corpus synthesis

Inputs: `analysis/<task>/{query_stats.json, seed_baseline.json, scaling.json}` for 14 tasks and `results/level_check.md`. NFCorpus is the full corpus with 100 synthetic queries; the 13 Nano* tasks use 40 synthetic queries and the corpus's own 50 queries (NanoTouche2020 has 49). Every number below is read from those JSON files.

On all 14 tasks `scaling.full.rho` equals `seed_baseline.gym_vs_truth.spearman_rho`; 12 models, 66 pairs, k=10, 200 draws per grid point, tolerance 0.05; quality `n_unscored` = 0; copied-word `n_missing_seed_docs` = 0; 66/66 pairs found and 0 kept-but-unjudged queries in every record. Generator is `openai/gpt-oss-20b` with seed docs given (`docs_given: true`) everywhere. "Official" and "truth" both mean the official MTEB ranking loaded from `truth.json`.

## Table 1. Synthetic queries vs the corpus's own queries (`query_stats.json`)

Word counts are means; copied-word share is the per-query mean share of a synthetic query's words that appear in its seed document (undefined for the original queries, which have no seed document, so it is omitted for them by construction). Quality is the generator-side 1..5 score of the kept queries. Generated/kept/dropped come from `synthetic.filter`.

| corpus | syn n | orig n | syn mean words | orig mean words | syn question share | orig question share | syn copied-word share | quality dist (score:count) | quality mean | generated | kept | dropped | drop rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NFCorpus | 100 | 323 | 13.88 | 3.29 | 1.00 | 0.14 | 0.630 | 5:100 | 5.000 | 160 | 100 | 60 | 0.375 |
| NanoArguAnaRetrieval | 40 | 50 | 15.03 | 193.00 | 0.90 | 0.06 | 0.535 | 4:6, 5:34 | 4.850 | 64 | 40 | 24 | 0.375 |
| NanoClimateFeverRetrieval | 40 | 50 | 14.03 | 21.36 | 1.00 | 0.02 | 0.454 | 3:1, 4:4, 5:35 | 4.850 | 64 | 40 | 24 | 0.375 |
| NanoDBPediaRetrieval | 40 | 50 | 12.22 | 5.52 | 1.00 | 0.22 | 0.753 | 5:40 | 5.000 | 64 | 40 | 24 | 0.375 |
| NanoFEVERRetrieval | 40 | 50 | 13.03 | 7.94 | 1.00 | 0.00 | 0.767 | 5:40 | 5.000 | 64 | 40 | 24 | 0.375 |
| NanoFiQA2018Retrieval | 40 | 50 | 17.20 | 10.20 | 1.00 | 0.74 | 0.571 | 5:40 | 5.000 | 64 | 40 | 24 | 0.375 |
| NanoHotpotQARetrieval | 40 | 50 | 15.45 | 14.92 | 0.97 | 0.98 | 0.688 | 3:1, 4:1, 5:38 | 4.925 | 64 | 40 | 24 | 0.375 |
| NanoMSMARCORetrieval | 40 | 50 | 13.65 | 5.74 | 1.00 | 0.66 | 0.544 | 4:2, 5:38 | 4.950 | 64 | 40 | 24 | 0.375 |
| NanoNFCorpusRetrieval | 40 | 50 | 13.62 | 3.26 | 1.00 | 0.16 | 0.624 | 4:1, 5:39 | 4.975 | 64 | 40 | 24 | 0.375 |
| NanoNQRetrieval | 40 | 50 | 12.18 | 9.04 | 1.00 | 0.88 | 0.636 | 4:1, 5:39 | 4.975 | 64 | 40 | 24 | 0.375 |
| NanoQuoraRetrieval | 40 | 50 | 11.22 | 9.04 | 0.97 | 1.00 | 0.364 | 3:2, 4:15, 5:23 | 4.525 | 64 | 40 | 24 | 0.375 |
| NanoSCIDOCSRetrieval | 40 | 50 | 18.38 | 9.44 | 1.00 | 0.06 | 0.595 | 4:2, 5:38 | 4.950 | 64 | 40 | 24 | 0.375 |
| NanoSciFactRetrieval | 40 | 50 | 14.45 | 13.54 | 1.00 | 0.00 | 0.681 | 5:40 | 5.000 | 64 | 40 | 24 | 0.375 |
| NanoTouche2020Retrieval | 40 | 49 | 19.02 | 6.55 | 1.00 | 1.00 | 0.506 | 5:40 | 5.000 | 64 | 40 | 24 | 0.375 |

Across the 14 corpora (mean of per-corpus means): synthetic 14.53 words vs original 22.35 (original median 9.04; the original mean is pulled by ArguAna's 193.00-word queries); synthetic question share 0.989 vs original 0.423; copied-word share 0.596 (min 0.364 NanoQuoraRetrieval, max 0.767 NanoFEVERRetrieval). Pooled quality over all 620 kept queries: 3:4, 4:32, 5:584 (94.2% scored 5). Every run generated exactly 1.6x the requested count (64 for 40, 160 for 100) and kept exactly the requested count, so the drop rate is 0.375 by construction on all 14.

Caveats carried from the files' own notes: `n_generated` counts queries that passed the generation-time length/degeneracy heuristics and the package does not record why a query was dropped afterwards, so the drop folds together the quality gate (min score 3), near-duplicate removal (Jaccard >= 0.8) and the cut to `n_queries`; a quality score of 3 may be a defaulted value written when the scorer's answer could not be parsed (4 such 3s exist in total, on NanoClimateFever, NanoHotpotQA and NanoQuora).

## Table 2. Judge (gym) vs seed-document baseline against the official ranking (`seed_baseline.json`)

Spearman rho over the 12-model roster. `gym vs official` = the LLM-judged gym rating vs the official MTEB score; `seed vs official` = nDCG@10 computed with the seed documents as the only relevant labels, vs the official score; `seed vs gym` = seed nDCG vs gym rating. `judge - seed` = (gym vs official) - (seed vs official). p is the two-sided Spearman p-value from the file. Seed coverage = `coverage.fraction`, the fraction of kept queries whose seed document lands in at least one model's top 10 (the sibling `seed_baseline.md` states it that way). It is not a label-completeness figure: every kept query has a seed qrel and is scored for every model (`n_queries_scored` equals the kept count on all 14).

| corpus | gym vs official rho (p) | seed vs official rho (p) | seed vs gym rho | judge - seed | seed coverage |
|---|---|---|---|---|---|
| NFCorpus | +0.818 (0.001) | +0.671 (0.017) | +0.741 | +0.147 | 1.000 |
| NanoArguAnaRetrieval | +0.420 (0.175) | +0.881 (0.000) | +0.587 | -0.462 | 1.000 |
| NanoClimateFeverRetrieval | +0.168 (0.602) | +0.231 (0.471) | +0.063 | -0.063 | 0.750 |
| NanoDBPediaRetrieval | +0.671 (0.017) | +0.776 (0.003) | +0.811 | -0.105 | 1.000 |
| NanoFEVERRetrieval | +0.490 (0.106) | +0.476 (0.118) | +0.573 | +0.014 | 1.000 |
| NanoFiQA2018Retrieval | +0.580 (0.048) | +0.469 (0.124) | +0.825 | +0.112 | 0.975 |
| NanoHotpotQARetrieval | +0.755 (0.005) | +0.769 (0.003) | +0.965 | -0.014 | 1.000 |
| NanoMSMARCORetrieval | +0.804 (0.002) | +0.573 (0.051) | +0.811 | +0.231 | 0.950 |
| NanoNFCorpusRetrieval | +0.399 (0.199) | +0.105 (0.745) | +0.207 | +0.294 | 1.000 |
| NanoNQRetrieval | +0.545 (0.067) | +0.616 (0.033) | +0.876 | -0.071 | 1.000 |
| NanoQuoraRetrieval | +0.238 (0.457) | +0.161 (0.618) | +0.965 | +0.077 | 1.000 |
| NanoSCIDOCSRetrieval | +0.573 (0.051) | -0.530 (0.076) | -0.260 | +1.103 | 1.000 |
| NanoSciFactRetrieval | +0.741 (0.006) | +0.773 (0.003) | +0.808 | -0.031 | 1.000 |
| NanoTouche2020Retrieval | -0.280 (0.379) | +0.238 (0.457) | -0.049 | -0.517 | 0.950 |

**Summary over 14 corpora.** judge - seed: mean +0.051, median +0.000; the judge wins (difference > 0) on 7 of 14 and loses on 7; |difference| <= 0.10 on 6 of 14 and <= 0.05 on 3. Column means/medians: gym vs official 0.495/0.559, seed vs official 0.443/0.524, seed vs gym 0.566/0.775. Over the 13 nano tasks only: judge - seed mean +0.044, median -0.014, judge wins 6 of 13. gym vs official has p < 0.05 on 6 of 14 (NFCorpus, NanoDBPedia, NanoFiQA2018, NanoHotpotQA, NanoMSMARCO, NanoSciFact); seed vs official has p < 0.05 on 6 of 14 (NFCorpus, NanoArguAna, NanoDBPedia, NanoHotpotQA, NanoNQ, NanoSciFact). Seed coverage is below 1.0 on 4 corpora (NanoClimateFever 0.750, NanoMSMARCO 0.950, NanoTouche2020 0.950, NanoFiQA2018 0.975).

## Table 3. Subsampling, smallest budget whose mean rho is within 0.05 of the full-data rho (`scaling.json`)

All values are **means over 200 random draws** per grid point (seed 0); the criterion is |mean rho at the point - full-data rho| <= 0.05, and the smallest grid point satisfying it is reported. `within` is the fraction of the 200 draws that individually landed within 0.05, a much stricter statement than the mean criterion. Grids: queries {5, 10, 20, 40} (NFCorpus adds 80, 100); pair fraction {0.25, 0.5, 1.0}; models 4..12. `(full)` marks a corpus that only meets the criterion at the full grid point, where it holds trivially. `(nm)` marks an axis that is not monotone, i.e. some larger budget above the reported one falls back outside 0.05. At pair fraction 0.25 only 142 or 143 of the 200 draws produced a defined rho (`n_valid`), so those means rest on fewer draws.

| corpus | full rho | queries: smallest n (mean rho, within) | pairs: smallest fraction (mean rho, within) | models: smallest n (mean rho, within) |
|---|---|---|---|---|
| NFCorpus | +0.818 | 20 (+0.786, 0.41) | 0.25 (+0.806, 0.64) | 4 (+0.785, 0.46) |
| NanoArguAnaRetrieval | +0.420 | 5 (+0.384, 0.19) | 0.25 (+0.416, 0.41) | 9 (+0.379, 0.23) |
| NanoClimateFeverRetrieval | +0.168 | 40 (full) (+0.168, 1.00) | 0.5 (+0.139, 0.76) | 4 (+0.141, 0.10) |
| NanoDBPediaRetrieval | +0.671 | 40 (full) (+0.671, 1.00) | 1.0 (full) (+0.671, 1.00) | 10 (+0.634, 0.56) |
| NanoFEVERRetrieval | +0.490 | 40 (full) (+0.490, 1.00) | 0.25 (+0.449, 0.33) | 5 (nm) (+0.464, 0.13) |
| NanoFiQA2018Retrieval | +0.580 | 10 (nm) (+0.544, 0.21) | 0.25 (+0.627, 0.46) | 5 (+0.552, 0.12) |
| NanoHotpotQARetrieval | +0.755 | 40 (full) (+0.755, 1.00) | 0.25 (+0.706, 0.38) | 8 (+0.714, 0.34) |
| NanoMSMARCORetrieval | +0.804 | 40 (full) (+0.804, 1.00) | 0.5 (+0.779, 0.79) | 5 (+0.766, 0.04) |
| NanoNFCorpusRetrieval | +0.399 | 5 (+0.364, 0.22) | 0.25 (+0.371, 0.38) | 5 (nm) (+0.359, 0.07) |
| NanoNQRetrieval | +0.545 | 40 (full) (+0.545, 1.00) | 0.25 (+0.528, 0.33) | 5 (+0.537, 0.14) |
| NanoQuoraRetrieval | +0.238 | 40 (full) (+0.238, 1.00) | 1.0 (full) (+0.238, 1.00) | 4 (nm) (+0.201, 0.14) |
| NanoSCIDOCSRetrieval | +0.573 | 10 (+0.565, 0.34) | 0.25 (+0.612, 0.64) | 4 (+0.525, 0.07) |
| NanoSciFactRetrieval | +0.741 | 20 (+0.701, 0.49) | 0.25 (+0.706, 0.52) | 9 (+0.704, 0.36) |
| NanoTouche2020Retrieval | -0.280 | 40 (full) (-0.280, 1.00) | 0.25 (-0.246, 0.32) | 4 (nm) (-0.237, 0.00) |

**Summary.** Queries axis: 8 of 14 corpora only reach the criterion at the full query count; the rest reach it at 5 (NanoArguAna, NanoNFCorpus), 10 (NanoFiQA2018, NanoSCIDOCS) or 20 (NFCorpus, NanoSciFact). The per-draw `within` fraction at 20 queries (40 for NFCorpus, whose grid steps 40 -> 80 with no 50) averages 0.352 (range 0.185 to 0.520), and at 5 queries 0.172 (max 0.275). Pairs axis: 0.25 suffices on 10 of 14, 0.5 on 2, and 2 (NanoDBPedia, NanoQuora) need all pairs. Models axis: smallest count ranges 4 to 10, median 5, mean 5.79; the axis is non-monotone on 4 corpora (NanoFEVERRetrieval, NanoNFCorpusRetrieval, NanoQuoraRetrieval, NanoTouche2020Retrieval), and the queries axis is non-monotone on NanoFiQA2018Retrieval (within at 10, outside at 20).

## Level check (paper vs regen, from `results/level_check.md`)

`results/level_check.md` is one table over all 14 corpus rows. The file says it is not a replication (nano corpora, 12-model roster, rewritten judge prompt with the mteb task description injected, gpt-oss-20b generator). It reports Spearman +0.492 between paper rho and regen rho over 13 shared corpora, and +0.351 between paper kappa and regen S over 9 shared corpora. NFCorpus appears twice in that table: +0.818 at full scale (tier A) and +0.399 at nano tier (tier B); those are the same two numbers as `gym vs official` in Table 2 for NFCorpus and NanoNFCorpusRetrieval.

## Cross-corpus observations the numbers support

1. Synthetic queries are longer than the corpus's own queries on 12 of 14 corpora (exceptions: NanoArguAna, whose originals average 193.00 words, and NanoClimateFever at 21.36), and are almost always questions (share 1.00 on 11 of 14, minimum 0.90 on NanoArguAna) regardless of the original style, which spans 0.00 (NanoFEVER, NanoSciFact) to 1.00 (NanoQuora, NanoTouche2020). The synthetic question share does not track the original one.
2. On average 59.6% of a synthetic query's words come from its seed document (NanoQuora 0.364 to NanoFEVER 0.767). Across the 14 corpora, copied-word share and gym-vs-official rho have Spearman 0.547 (p 0.043); with n = 14 this is an association, not a mechanism.
3. The quality gate is nearly inert on the kept set: 584 of 620 kept queries scored 5, 32 scored 4, 4 scored 3, none lower; 6 corpora are all 5s. The 0.375 drop rate is identical on every run because each run generated 1.6x the request and kept exactly the request; the files cannot say how many of the 24 (or 60) drops were quality, dedup or the cut.
4. Judge vs seed baseline is a wash on aggregate: mean judge - seed +0.051, median +0.000, judge wins 7 of 14, and 6 of 14 are within 0.10 of each other. The mean is carried by NanoSCIDOCS (+1.103, where the seed labels correlate negatively with the official ranking at -0.530); the largest judge losses are NanoTouche2020 (-0.517, judge rho -0.280) and NanoArguAna (-0.462, seed rho +0.881).
5. Only 6 of 14 gym-vs-official correlations reach p < 0.05, and only 4 corpora are significant on both judge and seed (NFCorpus, NanoDBPedia, NanoHotpotQA, NanoSciFact). Seed vs gym agreement is high on most corpora (median 0.775) but ranges from -0.260 (NanoSCIDOCS) to +0.965 (NanoHotpotQA, NanoQuora).
6. Same corpus, different tier: NFCorpus at full scale with 100 queries gives gym-vs-official +0.818 while NanoNFCorpus with 40 queries gives +0.399; the seed baseline moves the same way (+0.671 vs +0.105).
7. On the queries axis the mean rho is outside 0.05 of the full-data value at every sub-full grid point on 8 of 14 corpora, so no smaller query count is shown to be enough there; at 20 queries (40 for NFCorpus) only about a third of individual draws (0.352 on average) land within 0.05 of the full rho, and at 5 queries about a sixth (0.172).
8. The two corpora that meet the mean criterion at 5 queries (NanoArguAna 0.384 vs full 0.420; NanoNFCorpus 0.364 vs 0.399) do so because their curves are flat and low, not because they converge: only 0.19 and 0.22 of draws are within tolerance at that point.
9. Pairs are the cheapest axis: 25% of the 66 pairs already puts the mean within 0.05 on 10 of 14 corpora (with 142 or 143 valid draws of 200). The models axis is the least well behaved: the smallest count ranges 4 to 10 and the within-0.05 flag flips back off at a larger count on 4 corpora (NanoFEVERRetrieval, NanoNFCorpusRetrieval, NanoQuoraRetrieval, NanoTouche2020Retrieval).
10. Seed coverage is incomplete on 4 corpora (NanoClimateFever 0.750, 30 of 40, is the lowest). Those queries are still labelled and still scored for all 12 models; their seed document just never reaches any model's top 10, so they contribute 0 nDCG to every model and no ranking signal, leaving those seed-vs-official rows on fewer informative queries than the judge rows.

## Files

- per-task inputs: `analysis/<task>/query_stats.json`, `seed_baseline.json`, `scaling.json`
- level check: `results/level_check.md`
