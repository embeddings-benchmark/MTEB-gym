# Position bias controls (regeneration sweep)

Controls that separate de-biasing from de-noising in the two-order judge design.

position_bias.py reports, per record, rho_both (Bradley-Terry refit from the two-order averaged
score_a, Spearman vs official nDCG@10), rho_order1 (refit from O[w1] only) and rho_order2 (refit
from 1 - O[w2] only). The single-order refits are not comparable with each other because both use
the same pair enumeration, so order 1 always hands the first-position bonus to each pair file's
model_a and order 2 to its model_b, and the roster (config.models) correlates with quality. Two
effects are confounded in "rho_both beats a single-order refit": de-biasing (the two-order average
cancels the slot bonus) and de-noising (a single-order refit rests on one verdict per pair instead
of two). Three controls computed from the raw verdict rows separate them.

  C1 random-order refit. For every verdict row (one model pair on one query) draw independently
     which order's verdict to use, O[w1] or 1 - O[w2], refit, Spearman vs truth. One verdict per pair
     like a single-order refit, but the slot bonus lands on a random model per pair. The draw is per
     row (pair x query), not once per model pair: a per-pair coin would leave every query of that pair
     with the same slot bonus, and C1 would no longer be unbiased in expectation.
  C2 roster-permuted fixed-slot refit. Per draw, permute the roster; for every row use O[w1] if
     model_a precedes model_b in the permuted roster, else 1 - O[w2]. The fixed-slot design with a
     random roster. The identity permutation reproduces rho_order1 and the reversed roster
     rho_order2 when every pair file is oriented along the roster (checked and reported). rho_order1
     and rho_order2 are then compared with the C2 2.5 to 97.5 band per record.
  C3 half-queries both-order refit. A random half of the queries (by qid), both orders averaged,
     refit. The same number of judge verdicts as a single-order refit over all queries, but unbiased.

Decomposition per record
  bias share  = mean(C1) - mean(rho_order1, rho_order2)
  noise share = rho_both - mean(C1)
  both minus C3 = rho_both - mean(C3)   (the pure noise cost of halving the verdict count)

Arm-level tests are two-sided Wilcoxon signed-rank over the records (scipy default, exact at n = 14)
and the mean over its standard error across records.

Scoring conventions follow position_bias.py. O = {A: 1, tie: 0.5, B: 0}; raw == ["identical"] and any
raw token outside {A, B, tie} score 0.5 in every refit and are excluded from decisive counts. The
refit is the package's Bradley-Terry (mteb_gym.rank._bradley_terry on the same win matrix that
mteb_gym.rank.rate builds), which reproduces the stored record ratings. The script fails loudly on
any record whose recomputed a_first_rate differs from the record, whose recomputed rho_both differs
from the record where the record carries one (synthetic arm; original records carry no
agreement.spearman_rho), or whose recomputed rho_both, rho_order1, rho_order2 or a_first_rate
differs from position_bias.json, beyond 1e-9.

Tie-break. The design is balanced in every record (each model plays the same number of games), so
the Bradley-Terry order is the order of total fractional wins. Where two models have exactly equal
total wins the fitted strengths differ only by the MM stopping tolerance (a rating gap of about 1e-8)
and spearmanr scores that as a strict rank difference. rho_both_tie_aware gives tied models the mean
of their ratings before correlating; it is reported alongside rho_both and does not enter the
decomposition.

Draws per control 200, seed 0. Each control cell is mean (sd over draws) [2.5, 97.5 percentiles over draws]. n_q is the number of queries in the record; C3 uses n_q // 2 of them per draw. The checks column reports the position_bias.json check, whether the record itself carries a rho to check against, the C2 identity and reversal checks, and any exact tie in total wins with the tie-aware rho_both.

## synthetic arm (14 records)

| task | n_q | rho_both | rho_order1 | rho_order2 | C1 mean (sd) [2.5, 97.5] | C2 mean (sd) [2.5, 97.5] | C3 mean (sd) [2.5, 97.5] | bias share | noise share | both minus C3 | checks |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| NFCorpus | 100 | 0.818 | 0.371 | 0.720 | 0.830 (sd 0.034) [0.762, 0.888] | 0.560 (sd 0.172) [0.210, 0.874] | 0.824 (sd 0.057) [0.699, 0.909] | 0.285 | -0.012 | -0.005 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoArguAnaRetrieval | 40 | 0.420 | 0.112 | 0.503 | 0.391 (sd 0.062) [0.273, 0.518] | 0.371 (sd 0.105) [0.147, 0.560] | 0.389 (sd 0.106) [0.188, 0.581] | 0.083 | 0.029 | 0.031 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoClimateFeverRetrieval | 40 | 0.168 | 0.182 | 0.028 | 0.137 (sd 0.043) [0.049, 0.196] | 0.110 (sd 0.071) [-0.028, 0.210] | 0.078 (sd 0.132) [-0.238, 0.301] | 0.032 | 0.031 | 0.090 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoDBPediaRetrieval | 40 | 0.671 | -0.182 | 0.385 | 0.582 (sd 0.105) [0.391, 0.770] | 0.123 (sd 0.298) [-0.455, 0.644] | 0.552 (sd 0.135) [0.314, 0.790] | 0.481 | 0.089 | 0.119 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoFEVERRetrieval | 40 | 0.490 | -0.273 | 0.301 | 0.448 (sd 0.103) [0.244, 0.616] | 0.131 (sd 0.303) [-0.497, 0.706] | 0.392 (sd 0.135) [0.077, 0.616] | 0.434 | 0.041 | 0.098 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoFiQA2018Retrieval | 40 | 0.580 | 0.573 | 0.497 | 0.632 (sd 0.062) [0.545, 0.784] | 0.566 (sd 0.150) [0.280, 0.804] | 0.640 (sd 0.104) [0.412, 0.839] | 0.097 | -0.052 | -0.060 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoHotpotQARetrieval | 40 | 0.755 | 0.063 | 0.133 | 0.699 (sd 0.095) [0.517, 0.846] | 0.129 (sd 0.277) [-0.476, 0.623] | 0.683 (sd 0.110) [0.440, 0.853] | 0.602 | 0.056 | 0.072 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoMSMARCORetrieval | 40 | 0.804 | 0.084 | 0.545 | 0.754 (sd 0.055) [0.636, 0.846] | 0.323 (sd 0.225) [-0.105, 0.706] | 0.742 (sd 0.080) [0.594, 0.881] | 0.440 | 0.050 | 0.062 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoNFCorpusRetrieval | 40 | 0.399 | 0.091 | 0.406 | 0.362 (sd 0.085) [0.203, 0.545] | 0.324 (sd 0.221) [-0.099, 0.693] | 0.363 (sd 0.109) [0.154, 0.581] | 0.114 | 0.037 | 0.036 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoNQRetrieval | 40 | 0.545 | -0.734 | 0.783 | 0.510 (sd 0.120) [0.273, 0.727] | 0.176 (sd 0.264) [-0.441, 0.644] | 0.486 (sd 0.110) [0.294, 0.713] | 0.485 | 0.036 | 0.060 | prior match; record match; c2 id=o1; c2 rev=o2; win tie bge-large-en-v1.5 = bge-small-en-v1.5, tie-aware rho_both 0.518 |
| NanoQuoraRetrieval | 40 | 0.238 | -0.056 | 0.371 | 0.173 (sd 0.075) [0.014, 0.308] | 0.147 (sd 0.243) [-0.330, 0.686] | 0.182 (sd 0.110) [-0.042, 0.392] | 0.016 | 0.065 | 0.055 | prior match; record match; c2 id=o1; c2 rev=o2; win tie mxbai-embed-large-v1 = all-mpnet-base-v2, tie-aware rho_both 0.224 |
| NanoSCIDOCSRetrieval | 40 | 0.573 | 0.573 | 0.259 | 0.606 (sd 0.035) [0.545, 0.678] | 0.524 (sd 0.159) [0.217, 0.797] | 0.617 (sd 0.074) [0.426, 0.762] | 0.190 | -0.033 | -0.044 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoSciFactRetrieval | 40 | 0.741 | -0.028 | 0.517 | 0.711 (sd 0.047) [0.615, 0.783] | 0.321 (sd 0.253) [-0.184, 0.748] | 0.703 (sd 0.081) [0.524, 0.826] | 0.466 | 0.030 | 0.038 | prior match; record match; c2 id=o1; c2 rev=o2 |
| NanoTouche2020Retrieval | 40 | -0.280 | -0.245 | 0.147 | -0.261 (sd 0.077) [-0.392, -0.111] | -0.224 (sd 0.188) [-0.553, 0.113] | -0.230 (sd 0.193) [-0.623, 0.112] | -0.212 | -0.019 | -0.050 | prior match; record match; c2 id=o1; c2 rev=o2 |

Over the 14 synthetic records the mean (median) rho_both is 0.495 (0.559), rho_order1 0.038 (0.073), rho_order2 0.400 (0.395), so the mean of the two single-order refits is 0.219 (0.201). C1, one random-order verdict per pair, averages 0.470 (0.546) with a within-record sd of 0.071; C2, the fixed-slot design under a random roster, averages 0.256 (0.248) with a within-record sd of 0.209; C3, both orders on half the queries, averages 0.459 (0.519) with a within-record sd of 0.110. The bias share, mean(C1) minus the single-order mean, is 0.251 on average (median 0.238), positive in 13 of 14 records. The noise share, rho_both minus mean(C1), is 0.025 on average (median 0.033), positive in 10 of 14. rho_both minus mean(C3), the cost of halving the verdict count with both orders kept, is 0.036 on average (median 0.047), positive in 10 of 14; rho_both lies inside the C3 2.5 to 97.5 band in 14 of 14 records, and mean(C1) lies inside the C2 band in 12 of 14. The mean a_first_rate is 0.719.

## original arm (14 records)

| task | n_q | rho_both | rho_order1 | rho_order2 | C1 mean (sd) [2.5, 97.5] | C2 mean (sd) [2.5, 97.5] | C3 mean (sd) [2.5, 97.5] | bias share | noise share | both minus C3 | checks |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| NFCorpus | 323 | 0.902 | 0.762 | 0.867 | 0.914 (sd 0.013) [0.888, 0.944] | 0.810 (sd 0.069) [0.657, 0.909] | 0.913 (sd 0.027) [0.860, 0.958] | 0.099 | -0.012 | -0.011 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoArguAnaRetrieval | 50 | 0.720 | 0.720 | 0.685 | 0.706 (sd 0.060) [0.580, 0.805] | 0.700 (sd 0.070) [0.559, 0.839] | 0.665 (sd 0.104) [0.461, 0.839] | 0.003 | 0.014 | 0.055 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoClimateFeverRetrieval | 50 | 0.182 | 0.245 | 0.168 | 0.185 (sd 0.045) [0.084, 0.287] | 0.175 (sd 0.109) [-0.028, 0.406] | 0.197 (sd 0.109) [-0.021, 0.399] | -0.021 | -0.003 | -0.015 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoDBPediaRetrieval | 50 | 0.636 | 0.406 | 0.594 | 0.642 (sd 0.016) [0.594, 0.671] | 0.432 (sd 0.154) [0.147, 0.706] | 0.608 (sd 0.068) [0.468, 0.727] | 0.142 | -0.006 | 0.029 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoFEVERRetrieval | 50 | 0.350 | 0.070 | 0.210 | 0.401 (sd 0.055) [0.315, 0.518] | 0.171 (sd 0.261) [-0.329, 0.616] | 0.405 (sd 0.085) [0.251, 0.574] | 0.261 | -0.051 | -0.055 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoFiQA2018Retrieval | 50 | 0.818 | 0.671 | 0.790 | 0.824 (sd 0.038) [0.734, 0.881] | 0.742 (sd 0.112) [0.517, 0.909] | 0.785 (sd 0.098) [0.538, 0.923] | 0.093 | -0.006 | 0.033 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoHotpotQARetrieval | 50 | 0.951 | 0.769 | 0.434 | 0.946 (sd 0.022) [0.902, 0.979] | 0.601 (sd 0.146) [0.266, 0.832] | 0.925 (sd 0.040) [0.846, 0.986] | 0.345 | 0.005 | 0.026 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoMSMARCORetrieval | 50 | 0.650 | -0.112 | 0.503 | 0.599 (sd 0.083) [0.448, 0.741] | 0.278 (sd 0.260) [-0.238, 0.748] | 0.561 (sd 0.142) [0.286, 0.811] | 0.403 | 0.052 | 0.090 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoNFCorpusRetrieval | 50 | 0.769 | 0.657 | 0.657 | 0.740 (sd 0.043) [0.657, 0.805] | 0.665 (sd 0.099) [0.482, 0.860] | 0.704 (sd 0.089) [0.510, 0.860] | 0.082 | 0.030 | 0.066 | prior match; no record rho; c2 id=o1; c2 rev=o2; win tie mxbai-embed-large-v1 = gte-large, tie-aware rho_both 0.750 |
| NanoNQRetrieval | 50 | 0.797 | -0.140 | 0.804 | 0.791 (sd 0.032) [0.748, 0.867] | 0.434 (sd 0.213) [-0.022, 0.811] | 0.786 (sd 0.064) [0.643, 0.902] | 0.459 | 0.006 | 0.011 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoQuoraRetrieval | 50 | -0.133 | -0.196 | 0.469 | -0.100 (sd 0.102) [-0.301, 0.084] | 0.172 (sd 0.244) [-0.273, 0.645] | -0.051 (sd 0.133) [-0.315, 0.182] | -0.236 | -0.033 | -0.081 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoSCIDOCSRetrieval | 50 | 0.741 | 0.727 | 0.622 | 0.715 (sd 0.041) [0.615, 0.783] | 0.671 (sd 0.110) [0.454, 0.860] | 0.689 (sd 0.087) [0.503, 0.832] | 0.040 | 0.026 | 0.052 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoSciFactRetrieval | 50 | 0.713 | 0.357 | 0.748 | 0.719 (sd 0.042) [0.643, 0.797] | 0.554 (sd 0.166) [0.154, 0.860] | 0.720 (sd 0.075) [0.580, 0.853] | 0.166 | -0.006 | -0.007 | prior match; no record rho; c2 id=o1; c2 rev=o2 |
| NanoTouche2020Retrieval | 49 | -0.077 | -0.126 | -0.021 | -0.100 (sd 0.061) [-0.218, 0.021] | -0.105 (sd 0.201) [-0.455, 0.309] | -0.121 (sd 0.146) [-0.392, 0.148] | -0.026 | 0.023 | 0.044 | prior match; no record rho; c2 id=o1; c2 rev=o2 |

Over the 14 original records the mean (median) rho_both is 0.573 (0.717), rho_order1 0.344 (0.381), rho_order2 0.538 (0.608), so the mean of the two single-order refits is 0.441 (0.526). C1, one random-order verdict per pair, averages 0.570 (0.711) with a within-record sd of 0.047; C2, the fixed-slot design under a random roster, averages 0.450 (0.494) with a within-record sd of 0.158; C3, both orders on half the queries, averages 0.556 (0.677) with a within-record sd of 0.091. The bias share, mean(C1) minus the single-order mean, is 0.129 on average (median 0.096), positive in 11 of 14 records. The noise share, rho_both minus mean(C1), is 0.003 on average (median 0.001), positive in 7 of 14. rho_both minus mean(C3), the cost of halving the verdict count with both orders kept, is 0.017 on average (median 0.027), positive in 9 of 14; rho_both lies inside the C3 2.5 to 97.5 band in 14 of 14 records, and mean(C1) lies inside the C2 band in 12 of 14. The mean a_first_rate is 0.638.

## What the controls show

De-biasing accounts for almost all of the rho_both advantage over a fixed-slot single order. The bias share, mean(C1) minus the single-order mean, averages 0.251 on the synthetic arm and 0.129 on the original arm, positive in 13 of 14 and 11 of 14 records (Wilcoxon p = 0.002 and 0.017; mean over standard error 3.9 and 2.6). Anchoring the fixed-slot baseline on the C2 mean instead of on the actual roster and its reverse gives mean(C1) minus mean(C2) of 0.214 and 0.120, so the conclusion does not rest on the roster. The negative bias shares fall on records where rho_both is at or below zero or where the judge shows no first-slot preference (a_first_rate below 0.5): synthetic NanoTouche2020Retrieval (rho_both -0.280, a_first_rate 0.575); original NanoClimateFeverRetrieval (rho_both 0.182, a_first_rate 0.440); NanoQuoraRetrieval (rho_both -0.133, a_first_rate 0.792); NanoTouche2020Retrieval (rho_both -0.077, a_first_rate 0.584). Where the judge does not track truth a higher rho is not a less biased rho, so the sign of the bias share carries no bias reading on those records. That grouping describes where the negative values fall; it was not fixed in advance.

Once the slot is random the second verdict per pair adds little. The noise share, rho_both minus mean(C1), averages 0.025 on the synthetic arm (positive in 10 of 14, Wilcoxon p = 0.04, mean over standard error 2.3) and 0.003 on the original arm (7 of 14, p = 0.67, mean over standard error 0.4). No record's absolute noise share exceeds its own C1 draw sd (largest ratio 0.94 synthetic, 0.94 original), so the per-record signs in that column are not individually meaningful; the arm means are. On the synthetic arm the cost of halving the verdicts is small, a tenth of the bias share, and only nominally significant; on the original arm nothing is detectable. The records' own bootstrap interval for rho_both (spearman_ci95 in the agreement block) is far wider than any noise share.

The p values are nominal. Eight signed-rank tests were run (four quantities on two arms) with no correction, the 14 records in an arm share the roster, the judge and the corpus set, and the two arms share the 14 corpora, so the effective sample is smaller than the counts suggest. The bias share survives any reasonable correction; the synthetic noise share and both minus C3 do not.

The actual roster is not special except on NanoNQRetrieval. rho_order1 lies inside the C2 2.5 to 97.5 band on 12 of 14 synthetic and 13 of 14 original records, rho_order2 on 12 of 14 and 14 of 14. Out of band, synthetic: NanoArguAnaRetrieval rho_order1 0.112 below 0.147; NanoNQRetrieval rho_order1 -0.734 below -0.441; NanoNQRetrieval rho_order2 0.783 above 0.644; NanoTouche2020Retrieval rho_order2 0.147 above 0.113; original: NanoNQRetrieval rho_order1 -0.140 below -0.022. The record whose roster is most aligned with truth (Spearman of roster position against official nDCG@10) is NanoNQRetrieval (0.741) on the synthetic arm and NanoNQRetrieval (0.741) on the original arm; a fixed roster-aligned slot injects truth into the slot bonus there, which is why its two single-order refits sit far apart and outside the roster-permuted band. The other out-of-band cases sit at the band edge, where a 2.5 percent edge from 200 draws rests on about 5 draws, so they may be band noise.

Monte Carlo error. With 200 draws the standard error of a control mean is at most 0.008 (C1), 0.021 (C2) and 0.014 (C3) per record on the synthetic arm and 0.007, 0.018 and 0.010 on the original arm. The original-arm noise share and both minus C3 are of the same order as these, so their per-record signs are not individually meaningful; the bias share is far above them. C1 draws the order per verdict row (one pair on one query), not once per model pair, and the headline bias share depends on that convention: drawing one order per model pair instead, so that a pair's slot bonus repeats across its queries, lowered C1 by 0.10 to 0.19 on the three records where it was re-derived independently.

Tie-break. The design is balanced in 28 of 28 records, so the Bradley-Terry order is the order of total fractional wins. In 3 records two models have exactly equal total wins and the stored rating gap is the MM stopping tolerance, which spearmanr scores as a strict rank difference: NanoNQRetrieval synthetic: bge-large-en-v1.5 = bge-small-en-v1.5 at 226.25 wins, rating gap 1.2e-08, rho_both 0.545 reported, 0.518 with tied ranks; NanoQuoraRetrieval synthetic: mxbai-embed-large-v1 = all-mpnet-base-v2 at 253 wins, rating gap 2.8e-08, rho_both 0.238 reported, 0.224 with tied ranks; NanoNFCorpusRetrieval original: mxbai-embed-large-v1 = gte-large at 343 wins, rating gap 1.9e-08, rho_both 0.769 reported, 0.750 with tied ranks. Giving tied models the mean rating moves the arm mean of rho_both from 0.495 to 0.492 (synthetic) and from 0.573 to 0.572 (original); the largest per-record shift is 0.027 synthetic and 0.020 original. The decomposition above uses the reported rho_both. No other record has a tie.

