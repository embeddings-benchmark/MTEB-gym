# Position bias of the judge (regeneration sweep)

Position-bias analysis of the judge over the regeneration sweep.

For every record under results/records/ (synthetic and original arms):
  rho_both   : Spearman(truth, ratings refit from the stored score_a)      [check vs record agreement.spearman_rho]
  rho_order1 : Spearman(truth, ratings refit from score_a = O[w1])          (model_a shown first)
  rho_order2 : Spearman(truth, ratings refit from score_a = 1 - O[w2])      (model_b shown first)
  split_rate : among pairs decisive in BOTH orders, fraction where the two orders favour different MODELS
               (w1 == w2 as letters: the judge picked the same presented slot both times)
  a_first_rate : over all decisive single-order verdicts, fraction won by the model shown first
               [check vs record diagnostics.a_first_rate]
Identical result sets (raw == ["identical"]) count as ties (0.5) for every rating refit and are excluded
from the order-specific counts. A raw token outside {A, B, tie, identical} is scored 0.5 in the
order-specific refits and excluded from decisive counts. The package coerces unparseable judge answers
before it writes the verdict file, so the invalid count is 0 on this sweep even though the records
report parse failures (`diagnostics.parse_failure_rate`, 61 calls of 210,140).

## How to read the two single-order columns

`rho_order1` and `rho_order2` are refit over the same pairs, so order 1 always hands the first-position
bonus to each pair file's `model_a` and order 2 always to its `model_b`. The roster is not random with
respect to quality: recovering the implied roster from `rank(rating_order1 - rating_order2)` gives the
same ordering on every record but the two NanoClimateFever ones (mean Spearman 0.90 against the
NFCorpus one), that ordering
correlates with the official scores (mean rho +0.24 over the 28 records, up to +0.74 on NanoNQ), and
the alignment predicts the gap between the two columns (rho -0.89 over the 28 records between
`rho(roster index, truth)` and `rho_order1 - rho_order2`). So the two single-order columns are biased in
opposite directions along the roster and their gap measures the roster, not the judge. The two
NanoClimateFever records, where the implied roster matches least, are also the two with the weakest
first-position effect (0.44 and 0.52).

What is roster-independent and can be read directly: `a_first_rate` (mean 0.72 on the synthetic arms,
0.64 on the original arms, 0.5 would be no bias), `split_rate` (among pairs decisive in both orders, the
winner flips with the order on 0.50 of them on the synthetic arms and 0.39 on the original arms), and
`rho_both` against
the two single-order refits taken together (mean of the pair 0.22 against 0.50, synthetic; 0.44 against
0.57, original). The swing between the two single-order columns on one corpus (NFCorpus 0.37 and 0.72
against 0.82 from both orders; NanoNQ -0.73 and 0.78 against 0.55) is how far a fixed slot assignment
can move the ranking, which is why both orders are judged. Part of the `rho_both` advantage is
de-noising rather than de-biasing, since a single-order refit rests on one verdict per pair; separating
the two needs a half-split control on the raw verdict rows.

## synthetic arm (14 records)

| task | n_q | n_pairs | rho_both | rec rho | rho_order1 | rho_order2 | split_rate (n_split/n_both_decisive) | a_first_rate | rec a_first | checks |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| NFCorpus | 100 | 6600 | 0.818 | 0.818 | 0.371 | 0.720 | 0.434 (2671/6158) | 0.693 | 0.693 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoArguAnaRetrieval | 40 | 2640 | 0.420 | 0.420 | 0.112 | 0.503 | 0.366 (749/2044) | 0.557 | 0.557 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoClimateFeverRetrieval | 40 | 2640 | 0.168 | 0.168 | 0.182 | 0.028 | 0.249 (584/2346) | 0.522 | 0.522 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoDBPediaRetrieval | 40 | 2640 | 0.671 | 0.671 | -0.182 | 0.385 | 0.743 (1775/2388) | 0.874 | 0.874 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoFEVERRetrieval | 40 | 2640 | 0.490 | 0.490 | -0.273 | 0.301 | 0.633 (1100/1738) | 0.822 | 0.822 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoFiQA2018Retrieval | 40 | 2640 | 0.580 | 0.580 | 0.573 | 0.497 | 0.385 (975/2531) | 0.625 | 0.625 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoHotpotQARetrieval | 40 | 2640 | 0.755 | 0.755 | 0.063 | 0.133 | 0.741 (1371/1850) | 0.877 | 0.877 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoMSMARCORetrieval | 40 | 2640 | 0.804 | 0.804 | 0.084 | 0.545 | 0.569 (1210/2128) | 0.768 | 0.768 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoNFCorpusRetrieval | 40 | 2640 | 0.399 | 0.399 | 0.091 | 0.406 | 0.450 (989/2199) | 0.703 | 0.703 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoNQRetrieval | 40 | 2640 | 0.545 | 0.545 | -0.734 | 0.783 | 0.698 (1276/1829) | 0.855 | 0.855 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoQuoraRetrieval | 40 | 2640 | 0.238 | 0.238 | -0.056 | 0.371 | 0.538 (1108/2061) | 0.768 | 0.768 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoSCIDOCSRetrieval | 40 | 2640 | 0.573 | 0.573 | 0.573 | 0.259 | 0.355 (803/2259) | 0.656 | 0.656 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoSciFactRetrieval | 40 | 2640 | 0.741 | 0.741 | -0.028 | 0.517 | 0.551 (1148/2083) | 0.774 | 0.774 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |
| NanoTouche2020Retrieval | 40 | 2640 | -0.280 | -0.280 | -0.245 | 0.147 | 0.310 (684/2207) | 0.575 | 0.575 | rho=rec; af=rec; ratings maxdiff 0.0e+00 |

mean / median over records: rho_both 0.495 / 0.559, rho_order1 0.038 / 0.073, rho_order2 0.400 / 0.395, split_rate 0.502 / 0.494, a_first_rate 0.719 / 0.735

## original arm (14 records)

| task | n_q | n_pairs | rho_both | rec rho | rho_order1 | rho_order2 | split_rate (n_split/n_both_decisive) | a_first_rate | rec a_first | checks |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| NFCorpus | 323 | 21318 | 0.902 | n/a | 0.762 | 0.867 | 0.312 (5829/18712) | 0.606 | 0.606 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoArguAnaRetrieval | 50 | 3300 | 0.720 | n/a | 0.720 | 0.685 | 0.384 (1180/3073) | 0.526 | 0.526 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoClimateFeverRetrieval | 50 | 3300 | 0.182 | n/a | 0.245 | 0.168 | 0.275 (795/2896) | 0.440 | 0.440 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoDBPediaRetrieval | 50 | 3300 | 0.636 | n/a | 0.406 | 0.594 | 0.321 (1010/3148) | 0.651 | 0.651 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoFEVERRetrieval | 50 | 3300 | 0.350 | n/a | 0.070 | 0.210 | 0.490 (1417/2891) | 0.739 | 0.739 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoFiQA2018Retrieval | 50 | 3300 | 0.818 | n/a | 0.671 | 0.790 | 0.305 (989/3242) | 0.577 | 0.577 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoHotpotQARetrieval | 50 | 3300 | 0.951 | n/a | 0.769 | 0.434 | 0.444 (1203/2711) | 0.729 | 0.729 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoMSMARCORetrieval | 50 | 3300 | 0.650 | n/a | -0.112 | 0.503 | 0.492 (1522/3096) | 0.738 | 0.738 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoNFCorpusRetrieval | 50 | 3300 | 0.769 | n/a | 0.657 | 0.657 | 0.281 (775/2755) | 0.595 | 0.595 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoNQRetrieval | 50 | 3300 | 0.797 | n/a | -0.140 | 0.804 | 0.484 (1516/3132) | 0.733 | 0.733 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoQuoraRetrieval | 50 | 3300 | -0.133 | n/a | -0.196 | 0.469 | 0.579 (1700/2937) | 0.792 | 0.792 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoSCIDOCSRetrieval | 50 | 3300 | 0.741 | n/a | 0.727 | 0.622 | 0.297 (868/2920) | 0.581 | 0.581 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoSciFactRetrieval | 50 | 3300 | 0.713 | n/a | 0.357 | 0.748 | 0.398 (1022/2570) | 0.644 | 0.644 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |
| NanoTouche2020Retrieval | 49 | 3234 | -0.077 | n/a | -0.126 | -0.021 | 0.331 (982/2967) | 0.584 | 0.584 | rho no-record; af=rec; ratings maxdiff 0.0e+00 |

mean / median over records: rho_both 0.573 / 0.717, rho_order1 0.344 / 0.381, rho_order2 0.538 / 0.608, split_rate 0.385 / 0.357, a_first_rate 0.638 / 0.625

