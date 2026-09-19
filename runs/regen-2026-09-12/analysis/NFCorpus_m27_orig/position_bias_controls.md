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
  C1 per pair (c1_pair). As C1 but one random order per model pair per draw, applied to all of that
     pair's queries, so a pair's slot bonus repeats across its queries. This is the deployment where the
     presentation order is fixed per pair; C1 is the deployment where the order is drawn per query. Same
     number of draws and the same fields as C1, from its own random stream seeded from (seed, 2).
  C2 at 1000 draws (c2_1000). C2 redrawn --c2-draws times (default 1000) from its own random stream
     seeded from (seed, 1), so a 2.5 percent band edge rests on about 25 draws instead of 5. rho_order1
     and rho_order2 are compared with both bands and the cases that change status are listed. The C1, C2
     and C3 streams are untouched by the two additions, so their draws are identical to the 200-draw run.

Decomposition per record
  bias share  = mean(C1) - mean(rho_order1, rho_order2)
  noise share = rho_both - mean(C1)
  both minus C3 = rho_both - mean(C3)   (the pure noise cost of halving the verdict count)
  C1 minus C1 per pair = mean(C1) - mean(C1 per pair)   (the cost of fixing the order per pair)

Arm-level tests are two-sided Wilcoxon signed-rank over the records (scipy default, exact at n = 14)
and the mean over its standard error across records.

spearman_ci95. The interval in a record's agreement block comes from mteb_gym.agreement.correlate: it
resamples the shared models with replacement (bootstrap=1000 resamples, seed 0, as passed by
regen_nano.py; a resample with fewer than 3 distinct models or an undefined rho is dropped), recomputes
Spearman between the fixed point-estimate ratings and truth on each resample, and takes the 2.5 and
97.5 percentiles. Queries and verdicts are not resampled, so it is a model-roster interval, not a
query-level or judge-noise uncertainty. The script recomputes it with that function from the record
ratings and truth.json for every record that carries one and fails loudly on a mismatch beyond 1e-9.

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

Draws per control 200, seed 0; C2 is drawn again 1000 times (c2_1000) from a separate stream. Each control cell is mean (sd over draws) [2.5, 97.5 percentiles over draws]. n_q is the number of queries in the record; C3 uses n_q // 2 of them per draw. The checks column reports the position_bias.json check, whether the record itself carries a rho to check against, the spearman_ci95 reproduction (model bootstrap), the C2 identity and reversal checks, any single-order refit outside the C2 band at 200 or 1000 draws, and any exact tie in total wins with the tie-aware rho_both.

## original arm (1 records)

| task | n_q | rho_both | rho_order1 | rho_order2 | C1 mean (sd) [2.5, 97.5] | C1 per pair mean (sd) [2.5, 97.5] | C2 mean (sd) [2.5, 97.5] | C2 at 1000 mean (sd) [2.5, 97.5] | C3 mean (sd) [2.5, 97.5] | bias share | noise share | both minus C3 | C1 minus C1 per pair | checks |
|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| NFCorpus | 323 | 0.944 | 0.748 | 0.902 | 0.946 (sd 0.012) [0.930, 0.965] | 0.913 (sd 0.036) [0.846, 0.972] | 0.849 (sd 0.053) [0.741, 0.944] | 0.851 (sd 0.058) [0.734, 0.958] | 0.937 (sd 0.025) [0.881, 0.972] | 0.121 | -0.002 | 0.007 | 0.032 | prior match; no record rho; ci95 absent (record carries no agreement block); c2 id=o1; c2 rev=o2 |

Over the 1 original records the mean (median) rho_both is 0.944 (0.944), rho_order1 0.748 (0.748), rho_order2 0.902 (0.902), so the mean of the two single-order refits is 0.825 (0.825). C1, one random-order verdict per pair, averages 0.946 (0.946) with a within-record sd of 0.012; C2, the fixed-slot design under a random roster, averages 0.849 (0.849) with a within-record sd of 0.053; C3, both orders on half the queries, averages 0.937 (0.937) with a within-record sd of 0.025. The bias share, mean(C1) minus the single-order mean, is 0.121 on average (median 0.121), positive in 1 of 1 records. The noise share, rho_both minus mean(C1), is -0.002 on average (median -0.002), positive in 0 of 1. rho_both minus mean(C3), the cost of halving the verdict count with both orders kept, is 0.007 on average (median 0.007), positive in 1 of 1; rho_both lies inside the C3 2.5 to 97.5 band in 1 of 1 records, and mean(C1) lies inside the C2 band in 0 of 1 (1 of 1 against the 1000-draw band). C1 per pair, one random order per model pair held across its queries, averages 0.913 (0.913) with a within-record sd of 0.036, so C1 minus C1 per pair is 0.032 on average (median 0.032, range 0.032 to 0.032), positive in 1 of 1. C2 at 1000 draws averages 0.851 (0.851) with a within-record sd of 0.058. The mean a_first_rate is 0.595.

## What the controls show

This run covers the original arm only (1 record), so the two-arm comparison is not written and the numbers below are this arm's own.

The bias share, mean(C1) minus the single-order mean, averages 0.121 (range 0.121 to 0.121), positive in 1 of 1; mean(C1) minus mean(C2) averages 0.097. The noise share, rho_both minus mean(C1), averages -0.002 (range -0.002 to -0.002), positive in 0 of 1; rho_both minus mean(C3) averages 0.007 (range 0.007 to 0.007), positive in 1 of 1. The largest absolute noise share over its own C1 draw sd is 0.15. C1 per pair averages 0.913, so C1 minus C1 per pair averages 0.032 (range 0.032 to 0.032), positive in 1 of 1. Signed-rank tests need more than one record and are not read here. Negative bias shares: none. The spearman_ci95 interval in a record's agreement block is a model bootstrap, not a query-level uncertainty. mteb_gym.agreement.correlate resamples the shared models (all 12 roster models on every record here) with replacement 1000 times (seed 0; a resample with fewer than 3 distinct models or an undefined rho is dropped), recomputes Spearman between the fixed point-estimate ratings and truth on each resample, and takes the 2.5 and 97.5 percentiles, so it measures how much rho depends on which models are in the roster; the queries and the verdicts are never resampled, and the judge noise that the controls above measure does not enter it. The original records carry no agreement block, so no interval is reported there.

Against the 200-draw C2 band, rho_order1 lies inside on 1 of 1 records and rho_order2 on 1 of 1; out of band: none. Against the 1000-draw band, rho_order1 lies inside on 1 of 1 and rho_order2 on 1 of 1; out of band: none. Cases whose status differs between the two bands: none. The record whose roster is most aligned with truth is NFCorpus (0.266); its single-order refits are 0.748 and 0.902.

Monte Carlo error. With 200 draws the standard error of a control mean is at most 0.001 (C1), 0.004 (C2), 0.002 (C3) and 0.003 (C1 per pair) per record; with 1000 draws the C2 standard error is at most 0.002. Each per-record C1 minus C1 per pair gap carries a Monte Carlo standard error of at most 0.003.

Tie-break. The design is balanced in 1 of 1 records. Exact ties in total wins: none. Giving tied models the mean rating moves the arm mean of rho_both from 0.944 to 0.944; the largest per-record shift is 0.000.

