# MTEB-Gym leaderboard

Label-free embedding-model rankings with a first-class reliability report. Live at
https://huggingface.co/spaces/tejasnaladala/mteb-gym-leaderboard; this directory is the app's source
and the script that builds its data file from gym records.

Each corpus tab is a Bradley-Terry ranking built from pairwise judge verdicts over frozen synthetic
queries; no relevance labels enter the ranking. The **Reliability** tab reports, for every corpus, the
judge's chance-corrected agreement (κ) with the official qrels measured on that corpus's own human
queries (`mteb_gym.reliability`). A ranking is only as trustworthy as its κ, and the app refuses to
show a ranked corpus without one.

## Regenerating the data file

Run the gym twice per corpus, once on synthetic queries (the ranking) and once on the dataset's own
queries (the reliability check), then score the second run against the labels and export:

```python
import mteb_gym as gym
from mteb_gym.reliability import reliability_all

judge, generator = gym.LLM("...", base_url="..."), gym.LLM("...", base_url="...")
models = ["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"]
gym.run("SciFact", models, judge, generator, output_folder="results")  # ranking
gym.run("SciFact", models, judge, queries="original", output_folder="results")  # reliability arm
reliability_all("results")  # writes record["reliability"] into every original-arm record
```

```bash
python -m leaderboard.export --output-folder results --out leaderboard/data/leaderboard_export.json
```

`export.py` pairs each corpus's synthetic record with its scored original-arm record. It stops on a
ranked corpus with no reliability row (`--allow-missing` drops such corpora and lists them in
`meta.dropped`), and on a corpus with more than one synthetic record (`--pin TASK=HASH` picks one;
`--judge` / `--generator` filter first).

## Running the app locally

```bash
pip install -r leaderboard/requirements.txt
python leaderboard/app.py
```

## Data provenance

The checked-in `data/leaderboard_export.json` was produced from the paper's runs at commit `b5327e9`
of this repository, before the package became `mteb_gym`; those records predate `export.py` and were
converted from the earlier result layout. Every rating traces to per-verdict JSONL files retained per
run. Regenerating under the current package changes the verdict identities (the judge prompt and the
task-description default changed), so a refreshed export should come from a full rerun, not a merge.
