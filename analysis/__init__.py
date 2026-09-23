"""Offline analyses over a gym output folder; each module is a script with importable functions.

scaling      ranking agreement as queries, pairs and models are subsampled from a record
query_stats  descriptive statistics of a run's synthetic queries next to the corpus's own
arena        the judge against the MTEB Arena's human votes

The no-judge label baseline (nDCG@10 against the dataset qrels or the seed documents) is part of
the package: run() writes it into every record and the leaderboard exports it.
"""
