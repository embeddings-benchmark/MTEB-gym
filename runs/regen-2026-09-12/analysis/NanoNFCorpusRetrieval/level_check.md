# Level check: paper (old package) vs regen (mteb_gym main; NFCorpus at full scale, the rest nano tier)

Not a replication: nano corpora, a 12-model roster, the rewritten judge prompt with the mteb task
description injected, and a gpt-oss-20b generator. The question is only whether the per-corpus
level and ordering survive the rewrite.

| corpus | paper rho (n) | regen rho (n) | paper kappa (n) | regen S (n models) | regen tier |
|---|---|---|---|---|---|
| NFCorpus | +0.670 (25) | +0.818 (12) | 0.565 (25) | +0.486 | A |
| NFCorpus | +0.670 (25) | +0.399 (12) | 0.565 (25) | +0.453 | B |
| SciFact | +0.743 (25) | +0.741 (12) | 0.590 (25) | +0.436 | B |
| FiQA-2018 | +0.709 (25) | +0.580 (12) | 0.552 (25) | +0.470 | B |
| ArguAna | +0.288 (25) | +0.420 (12) | 0.151 (25) | +0.353 | C |
| SCIDOCS | +0.706 (22) | +0.573 (12) | n/a | +0.283 | C |
| Quora | +0.573 (24) | +0.238 (12) | 0.334 (25) | +0.316 | C |
| DBPedia-entity | +0.423 (23) | +0.671 (12) | 0.318 (25) | +0.546 | A |
| HotpotQA | +0.386 (24) | +0.755 (12) | n/a | +0.904 | A |
| FEVER | +0.117 (22) | +0.490 (12) | n/a | +0.784 | A |
| Climate-FEVER | +0.259 (22) | +0.168 (12) | 0.058 (23) | +0.179 | C |
| NQ | -0.286 (8) | +0.545 (12) | 0.405 (24) | +0.581 | A |
| Touch\'e-2020 | +0.063 (15) | -0.280 (12) | n/a | +0.011 | C |
| NanoMSMARCORetrieval | n/a | +0.804 (12) | n/a | +0.236 | C |

Spearman between paper rho and regen rho over 13 shared corpora: +0.492
Spearman between paper kappa and regen S over 9 shared corpora: +0.351
