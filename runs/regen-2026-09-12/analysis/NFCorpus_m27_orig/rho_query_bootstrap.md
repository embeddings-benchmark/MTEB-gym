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

## original (MiniMax-M2.7) (1 record)

| task | n_q | rho_both | query mean (sd) [2.5, 97.5] | query width | model mean (sd) [2.5, 97.5] | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | checks |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| NFCorpus | 323 | 0.944 | 0.939 (sd 0.025) [0.881, 0.972] | 0.091 | 0.923 (sd 0.087) [0.689, 1.000] | 0.311 | 3.42 | -0.002 | -0.07 | 0.007 | 0.28 | model ci95 absent (no agreement block); no record rho; controls from NFCorpus_m27_orig/controls |

## Group means

| group | n | rho_both | query mean | query sd | query width | model width | model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | records with abs noise share above one query sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original (MiniMax-M2.7) | 1 | 0.944 | 0.939 | 0.025 | 0.091 | 0.311 | 3.42 | -0.002 | -0.07 | 0.007 | 0.28 | 0 |

## Results

Over the 1 record in the original (MiniMax-M2.7) group the query interval (2.5 to 97.5 percentiles over 1000 query resamples) is 0.091 wide on average with a bootstrap sd of 0.025, against a model interval 0.311 wide (sd 0.087); the model interval is wider on 1 of 1 records, by a factor of 3.42 on average. The bootstrap mean sits -0.005 from the point rho_both on average and the point lies inside the query interval on 1 of 1. The noise share divided by the query sd averages -0.068 (largest absolute value 0.07); it exceeds one query sd in absolute value on 0 of 1 records. Both minus C3 over the query sd averages 0.284 (largest absolute value 0.28), above one in absolute value on 0 of 1. The C3 draw sd is 1.00 query sds on average. None of the 1 records carries an agreement block, so the model interval is computed here for the first time and there is nothing to check it against.

Records whose absolute noise share exceeds one query-bootstrap sd: none. Records whose absolute both minus C3 exceeds one query-bootstrap sd: none.

