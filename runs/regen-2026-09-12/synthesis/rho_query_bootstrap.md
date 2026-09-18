# Query bootstrap of rho_both

Query-level bootstrap of rho_both for every record.

rho_both is the Spearman correlation between a record's Bradley-Terry ratings (two-order averaged
score_a, the package fit) and the official nDCG@10. The interval a record carries, agreement.spearman_ci95,
resamples the models, not the queries, so it says how much rho depends on the roster and nothing about how
much it depends on which queries were drawn. This script adds the query-level interval.

Query bootstrap. The record's queries (qids) are resampled with replacement, every verdict row of a drawn
query is kept with both orders (the stored score_a) and counted once per draw of its query, the Bradley-Terry
fit is redone with the package function on the weighted win matrix, and Spearman against truth is taken.
Resampling queries with multiplicity is the same as duplicating their verdict rows, so the fit is the package
fit on the resampled verdict list. Reported per record: rho_both (point, all weights one, which reproduces the
record's rho bit for bit), bootstrap mean, sd, 2.5 and 97.5 percentiles.

Model bootstrap. The models are resampled with replacement the way mteb_gym.agreement.correlate does it
(same draw scheme, resamples and seed as regen_nano.py; a resample with fewer than 3 distinct models or an
undefined rho is dropped). Where the record carries agreement.spearman_ci95 the recomputed interval must
match it to 1e-9 or the script fails.

Relation to the controls. From position_bias_controls.json each record's noise share (rho_both minus mean of
C1, the random-order single-verdict refit) and both minus C3 (rho_both minus the half-queries both-order
refit) are divided by the query-bootstrap sd, so a control difference can be read against the query-level
uncertainty of the quantity it is a difference of.

Scoring conventions follow position_bias_controls.py, whose helpers load the rows and build the fit.

Query resamples 1000, seed 0; model bootstrap 1000 resamples, seed 0 (the record's own setting). Each interval cell is mean (sd over resamples) [2.5, 97.5 percentiles]. n_q is the number of queries in the record. noise share and both minus C3 are taken from the controls run; the columns after them divide by the query-bootstrap sd of this run.

## synthetic (Qwen3.6-27B) (14 records)

| task | n_q | rho_both | query mean (sd) [2.5, 97.5] | query width | model mean (sd) [2.5, 97.5] | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | checks |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| NFCorpus | 100 | 0.818 | 0.818 (sd 0.062) [0.685, 0.909] | 0.224 | 0.786 (sd 0.143) [0.462, 0.957] | 0.495 | 2.21 | -0.012 | -0.19 | -0.005 | -0.09 | model ci95 match; point = record; controls from controls |
| NanoArguAnaRetrieval | 40 | 0.420 | 0.394 (sd 0.110) [0.203, 0.629] | 0.427 | 0.392 (sd 0.301) [-0.274, 0.871] | 1.145 | 2.68 | 0.029 | 0.26 | 0.031 | 0.28 | model ci95 match; point = record; controls from controls |
| NanoClimateFeverRetrieval | 40 | 0.168 | 0.072 (sd 0.132) [-0.210, 0.322] | 0.532 | 0.170 (sd 0.313) [-0.500, 0.743] | 1.243 | 2.34 | 0.031 | 0.23 | 0.090 | 0.68 | model ci95 match; point = record; controls from controls |
| NanoDBPediaRetrieval | 40 | 0.671 | 0.570 (sd 0.146) [0.266, 0.818] | 0.552 | 0.625 (sd 0.186) [0.122, 0.860] | 0.738 | 1.33 | 0.089 | 0.61 | 0.119 | 0.82 | model ci95 match; point = record; controls from controls |
| NanoFEVERRetrieval | 40 | 0.490 | 0.405 (sd 0.142) [0.091, 0.637] | 0.546 | 0.444 (sd 0.266) [-0.160, 0.861] | 1.021 | 1.87 | 0.041 | 0.29 | 0.098 | 0.69 | model ci95 match; point = record; controls from controls |
| NanoFiQA2018Retrieval | 40 | 0.580 | 0.632 (sd 0.111) [0.434, 0.839] | 0.406 | 0.548 (sd 0.272) [-0.160, 0.904] | 1.064 | 2.62 | -0.052 | -0.47 | -0.060 | -0.54 | model ci95 match; point = record; controls from controls |
| NanoHotpotQARetrieval | 40 | 0.755 | 0.684 (sd 0.104) [0.462, 0.853] | 0.392 | 0.719 (sd 0.162) [0.314, 0.958] | 0.644 | 1.64 | 0.056 | 0.54 | 0.072 | 0.69 | model ci95 match; point = record; controls from controls |
| NanoMSMARCORetrieval | 40 | 0.804 | 0.729 (sd 0.086) [0.552, 0.874] | 0.322 | 0.780 (sd 0.180) [0.314, 0.993] | 0.679 | 2.11 | 0.050 | 0.58 | 0.062 | 0.73 | model ci95 match; point = record; controls from controls |
| NanoNFCorpusRetrieval | 40 | 0.399 | 0.370 (sd 0.108) [0.161, 0.587] | 0.427 | 0.371 (sd 0.290) [-0.312, 0.848] | 1.160 | 2.72 | 0.037 | 0.34 | 0.036 | 0.33 | model ci95 match; point = record; controls from controls |
| NanoNQRetrieval | 40 | 0.545 | 0.500 (sd 0.114) [0.280, 0.713] | 0.434 | 0.524 (sd 0.232) [-0.025, 0.889] | 0.915 | 2.11 | 0.036 | 0.31 | 0.060 | 0.52 | model ci95 match; point = record; controls from controls |
| NanoQuoraRetrieval | 40 | 0.238 | 0.171 (sd 0.116) [-0.063, 0.378] | 0.441 | 0.225 (sd 0.349) [-0.500, 0.806] | 1.307 | 2.96 | 0.065 | 0.56 | 0.055 | 0.48 | model ci95 match; point = record; controls from controls |
| NanoSCIDOCSRetrieval | 40 | 0.573 | 0.613 (sd 0.073) [0.461, 0.755] | 0.294 | 0.551 (sd 0.229) [0.039, 0.892] | 0.853 | 2.90 | -0.033 | -0.45 | -0.044 | -0.60 | model ci95 match; point = record; controls from controls |
| NanoSciFactRetrieval | 40 | 0.741 | 0.701 (sd 0.080) [0.524, 0.832] | 0.308 | 0.706 (sd 0.168) [0.321, 0.941] | 0.621 | 2.02 | 0.030 | 0.38 | 0.038 | 0.48 | model ci95 match; point = record; controls from controls |
| NanoTouche2020Retrieval | 40 | -0.280 | -0.230 (sd 0.189) [-0.601, 0.140] | 0.741 | -0.265 (sd 0.347) [-0.843, 0.475] | 1.318 | 1.78 | -0.019 | -0.10 | -0.050 | -0.26 | model ci95 match; point = record; controls from controls |

## synthetic (MiniMax-M2.7) (1 record)

| task | n_q | rho_both | query mean (sd) [2.5, 97.5] | query width | model mean (sd) [2.5, 97.5] | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | checks |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| NFCorpus | 100 | 0.895 | 0.881 (sd 0.049) [0.769, 0.958] | 0.189 | 0.867 (sd 0.108) [0.598, 1.000] | 0.402 | 2.13 | -0.005 | -0.09 | 0.016 | 0.33 | model ci95 match; point = record; controls from NFCorpus_m27/controls |

## original (Qwen3.6-27B) (14 records)

| task | n_q | rho_both | query mean (sd) [2.5, 97.5] | query width | model mean (sd) [2.5, 97.5] | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | checks |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| NFCorpus | 323 | 0.902 | 0.913 (sd 0.027) [0.860, 0.958] | 0.098 | 0.865 (sd 0.085) [0.640, 0.978] | 0.338 | 3.45 | -0.012 | -0.44 | -0.011 | -0.40 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoArguAnaRetrieval | 50 | 0.720 | 0.673 (sd 0.107) [0.455, 0.846] | 0.392 | 0.694 (sd 0.207) [0.204, 0.978] | 0.774 | 1.98 | 0.014 | 0.13 | 0.055 | 0.51 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoClimateFeverRetrieval | 50 | 0.182 | 0.181 (sd 0.113) [-0.042, 0.399] | 0.441 | 0.172 (sd 0.320) [-0.482, 0.712] | 1.194 | 2.71 | -0.003 | -0.03 | -0.015 | -0.13 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoDBPediaRetrieval | 50 | 0.636 | 0.612 (sd 0.070) [0.462, 0.741] | 0.280 | 0.606 (sd 0.252) [-0.025, 0.942] | 0.968 | 3.46 | -0.006 | -0.08 | 0.029 | 0.41 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoFEVERRetrieval | 50 | 0.350 | 0.409 (sd 0.091) [0.238, 0.587] | 0.350 | 0.325 (sd 0.320) [-0.350, 0.864] | 1.214 | 3.47 | -0.051 | -0.57 | -0.055 | -0.60 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoFiQA2018Retrieval | 50 | 0.818 | 0.775 (sd 0.105) [0.524, 0.916] | 0.392 | 0.778 (sd 0.125) [0.484, 0.949] | 0.465 | 1.19 | -0.006 | -0.05 | 0.033 | 0.31 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoHotpotQARetrieval | 50 | 0.951 | 0.926 (sd 0.038) [0.839, 0.986] | 0.147 | 0.934 (sd 0.086) [0.691, 1.000] | 0.309 | 2.10 | 0.005 | 0.13 | 0.026 | 0.69 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoMSMARCORetrieval | 50 | 0.650 | 0.554 (sd 0.132) [0.287, 0.790] | 0.503 | 0.621 (sd 0.217) [0.082, 0.934] | 0.852 | 1.69 | 0.052 | 0.39 | 0.090 | 0.68 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoNFCorpusRetrieval | 50 | 0.769 | 0.706 (sd 0.080) [0.545, 0.832] | 0.287 | 0.729 (sd 0.190) [0.226, 0.957] | 0.731 | 2.55 | 0.030 | 0.37 | 0.066 | 0.82 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoNQRetrieval | 50 | 0.797 | 0.776 (sd 0.071) [0.615, 0.909] | 0.294 | 0.776 (sd 0.199) [0.248, 0.993] | 0.745 | 2.54 | 0.006 | 0.09 | 0.011 | 0.16 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoQuoraRetrieval | 50 | -0.133 | -0.059 (sd 0.138) [-0.315, 0.196] | 0.510 | -0.134 (sd 0.352) [-0.785, 0.539] | 1.324 | 2.59 | -0.033 | -0.24 | -0.081 | -0.59 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoSCIDOCSRetrieval | 50 | 0.741 | 0.692 (sd 0.093) [0.483, 0.846] | 0.364 | 0.703 (sd 0.198) [0.222, 0.957] | 0.735 | 2.02 | 0.026 | 0.28 | 0.052 | 0.56 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoSciFactRetrieval | 50 | 0.713 | 0.711 (sd 0.073) [0.566, 0.846] | 0.280 | 0.665 (sd 0.153) [0.293, 0.892] | 0.599 | 2.14 | -0.006 | -0.08 | -0.007 | -0.09 | model ci95 absent (no agreement block); no record rho; controls from controls |
| NanoTouche2020Retrieval | 49 | -0.077 | -0.114 (sd 0.131) [-0.364, 0.133] | 0.497 | -0.068 (sd 0.303) [-0.669, 0.513] | 1.182 | 2.38 | 0.023 | 0.17 | 0.044 | 0.34 | model ci95 absent (no agreement block); no record rho; controls from controls |

## Group means

| group | n | rho_both | query mean | query sd | query width | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | records with abs noise share above one query sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic (Qwen3.6-27B) | 14 | 0.495 | 0.459 | 0.112 | 0.432 | 0.943 | 2.24 | 0.025 | 0.21 | 0.036 | 0.30 | 0 |
| synthetic (MiniMax-M2.7) | 1 | 0.895 | 0.881 | 0.049 | 0.189 | 0.402 | 2.13 | -0.005 | -0.09 | 0.016 | 0.33 | 0 |
| original (Qwen3.6-27B) | 14 | 0.573 | 0.554 | 0.091 | 0.345 | 0.816 | 2.45 | 0.003 | 0.01 | 0.017 | 0.19 | 0 |

## Results

Over the 14 records in the synthetic (Qwen3.6-27B) group the query interval (2.5 to 97.5 percentiles over 1000 query resamples) is 0.432 wide on average with a bootstrap sd of 0.112, against a model interval 0.943 wide (sd 0.246); the model interval is wider on 14 of 14 records, by a factor of 2.24 on average. The bootstrap mean sits -0.035 from the point rho_both on average and the point lies inside the query interval on 14 of 14. The noise share divided by the query sd averages 0.206 (largest absolute value 0.61); it exceeds one query sd in absolute value on 0 of 14 records. Both minus C3 over the query sd averages 0.300 (largest absolute value 0.82), above one in absolute value on 0 of 14. The C3 draw sd is 0.98 query sds on average. The model bootstrap reproduces the record interval on 14 of 14 records.

Over the 1 record in the synthetic (MiniMax-M2.7) group the query interval (2.5 to 97.5 percentiles over 1000 query resamples) is 0.189 wide on average with a bootstrap sd of 0.049, against a model interval 0.402 wide (sd 0.108); the model interval is wider on 1 of 1 records, by a factor of 2.13 on average. The bootstrap mean sits -0.014 from the point rho_both on average and the point lies inside the query interval on 1 of 1. The noise share divided by the query sd averages -0.095 (largest absolute value 0.09); it exceeds one query sd in absolute value on 0 of 1 records. Both minus C3 over the query sd averages 0.325 (largest absolute value 0.33), above one in absolute value on 0 of 1. The C3 draw sd is 0.99 query sds on average. The model bootstrap reproduces the record interval on 1 of 1 records.

Over the 14 records in the original (Qwen3.6-27B) group the query interval (2.5 to 97.5 percentiles over 1000 query resamples) is 0.345 wide on average with a bootstrap sd of 0.091, against a model interval 0.816 wide (sd 0.215); the model interval is wider on 14 of 14 records, by a factor of 2.45 on average. The bootstrap mean sits -0.019 from the point rho_both on average and the point lies inside the query interval on 14 of 14. The noise share divided by the query sd averages 0.006 (largest absolute value 0.57); it exceeds one query sd in absolute value on 0 of 14 records. Both minus C3 over the query sd averages 0.190 (largest absolute value 0.82), above one in absolute value on 0 of 14. The C3 draw sd is 1.00 query sds on average. None of the 14 records carries an agreement block, so the model interval is computed here for the first time and there is nothing to check it against.

Records whose absolute noise share exceeds one query-bootstrap sd: none. Records whose absolute both minus C3 exceeds one query-bootstrap sd: none.

On NFCorpus the MiniMax-M2.7 record has rho_both 0.895 with query interval [0.769, 0.958] and model interval [0.598, 1.000]; the Qwen3.6-27B record has rho_both 0.818 with query interval [0.685, 0.909] and model interval [0.462, 0.957]. The query intervals overlap; the model intervals overlap. The difference in rho_both, 0.077, is 1.37 pooled query sds. Both records are judged on the same 100 queries and the two bootstraps share their resampled query sets draw for draw (same seed), so the paired difference MiniMax-M2.7 minus Qwen3.6-27B has bootstrap mean 0.063, sd 0.052 and interval [-0.028, 0.175].

