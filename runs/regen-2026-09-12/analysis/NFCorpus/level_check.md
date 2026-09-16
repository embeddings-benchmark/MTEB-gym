# Level check: paper (old package) vs regen (mteb_gym main; NFCorpus at full scale, the rest nano tier)

Not a replication: nano corpora, a 12-model roster, the rewritten judge prompt with the mteb task
description injected, and a gpt-oss-20b generator. The question is only whether the per-corpus
level and ordering survive the rewrite.

| corpus | paper rho (n) | regen rho (n) | paper kappa (n) | regen S (n models) | regen tier |
|---|---|---|---|---|---|
| NFCorpus | +0.670 (25) | +0.818 (12) | 0.565 (25) | +0.486 | A |
