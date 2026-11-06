<h1 align="center">MTEB Gym</h1>

<h3 align="center">Evaluate embedding models on any corpus, without relevance labels</h3>

<p align="center">
  <a href="https://pypi.org/project/mteb-gym/"><img src="https://img.shields.io/pypi/v/mteb-gym.svg" alt="PyPI"></a>
  <a href="https://github.com/embeddings-benchmark/MTEB-gym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/embeddings-benchmark/MTEB-gym.svg" alt="License"></a>
  <a href="https://pepy.tech/project/mteb-gym"><img src="https://static.pepy.tech/badge/mteb-gym" alt="Downloads"></a>
</p>

<h4 align="center">
  <a href="#installation">Installation</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="#how-it-works">How it works</a> |
  <a href="#usage">Usage</a> |
  <a href="#output">Output</a> |
  <a href="#choosing-the-judge-and-generator">Judge and generator</a> |
  <a href="#citing">Citing</a>
</h4>

MTEB Gym ranks embedding models on your own documents. You don't need labels or queries.

## Installation

```bash
pip install mteb-gym
```

## Quickstart

```python
import mteb_gym as gym

result = gym.run(
    corpus="path/to/docs",
    models=["mteb/baseline-bm25s", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"],
    judge="autotrust/JEV-27B",
    judge_url="http://localhost:8001",
    generator="deepseek-ai/DeepSeek-V4-Flash",
    generator_url="http://localhost:8000/v1",
)
print(result.leaderboard)
```

The judge and the generator can be [any model](#choosing-the-judge-and-generator), served by you or through an API.

## How it works

1. **Queries.** A generator writes queries from your documents. You can use your own instead.
2. **Retrieval.** Each model retrieves its top 10 documents for each query, through `mteb`.
3. **Judging.** A judge scores what each model retrieved, in one of two ways.
4. **Ranking.** The models are ranked by those scores, with confidence intervals.

### Pointwise and pairwise judging

| | Pointwise | Pairwise |
|---|---|---|
| **The judge sees** | One query and one document | One query and two models' lists |
| **The judge answers** | A grade from 0 to 5 | Which list is better, or a tie |
| **Each model's rating** | nDCG over the grades, 0 to 100 | Bradley-Terry, centred on 1000 |
| **Judge calls** | One per retrieved document, shared across models | Two per query per model pair, one in each order |
| **Adding a model** | Grades only the documents it adds | Judges only its new pairs |
| **Default for** | A decision model | An LLM |

Either kind of judge works either way. Set it with `judging="pointwise"` or `judging="pairwise"`. Pairwise costs more: 100 queries and 10 models take 9,000 calls, which `pairs_per_query` reduces.

## Usage

### Your documents

| Corpus | Example |
|---|---|
| A folder of `.txt` or `.md` files | `corpus="path/to/docs"` |
| A `.jsonl` file with `id` and `text` | `corpus="docs.jsonl"` |
| An MTEB task | `corpus="NFCorpus"` |

### Describe the task

Say what a good result is. The generator and the judge both get it. It's optional, but recommended:

```python
gym.run(..., task_instruction="Given a support ticket, retrieve the help article that resolves it")
```

### Use your own queries

If you have real queries, use them. You then need no generator:

```python
gym.run(..., queries="my_queries.jsonl")  # with id and text; or a .txt file, or a list of strings
gym.run(..., queries="original")  # an MTEB task's own queries
```

### Encode once

Keep each model's vectors, so a new run doesn't encode your documents again:

```python
gym.run(..., embedding_cache="path/to/vectors")
```

### All options

| Option | Default | |
|---|---|---|
| `queries` | `"synthetic"` | `"synthetic"`, `"original"`, or your own |
| `task_instruction` | `None` | One sentence on what a good result is |
| `judging` | `"auto"` | `"pointwise"`, `"pairwise"`, or the judge's default |
| `pairs_per_query` | `None` | Pairwise: judge this many random model pairs per query, not all |
| `n_queries` | `100` | Queries to generate or sample |
| `top_k` | `10` | Documents judged per query |
| `doc_chars` | `None` | Characters of each document shown: 16,000, with a pairwise judge on 4,000 |
| `filter_queries` | `False` | Add an LLM quality filter and deduplication of generated queries |
| `embedding_cache` | `None` | A folder for each model's vectors |
| `max_seq_length` | `None` | Cap a model's length. By default, each model runs as `mteb` loads it |
| `output_folder` | `"results"` | Where records go |
| `cache_folder` | `None` | Queries, predictions and judgments: `$MTEB_GYM_CACHE`, else `~/.cache/mteb_gym` |
| `seed` | `0` | Sampling and intervals |

## Output

`print(result.leaderboard)` shows the ranking:

```text
==============================================================================
Rank Model                                   Rating    CI±    W    L    T
------------------------------------------------------------------------------
1    bge-small-en-v1.5                           84      2   61   39    0
2    e5-base-v2                                  81      2   55   45    0
3    baseline-bm25s                              72      3   34   66    0
==============================================================================
```

Each run also writes a record to `results/<task>/`. Read them back with `gym.load_results("results/").to_dataframe()`.

<details>
<summary>Record format</summary>

```jsonc
{
  "task_name": "NFCorpus",
  "dataset": {"path": "...", "revision": "..."},
  "mteb_version": "...", "gym_version": "...", "gym_revision": "...",
  "config": {
    "judging": "pointwise", "judge_model": "...", "generator_model": "...",
    "query_set": "...", "n_queries": 100, "top_k": 10, "doc_chars": 16000, "seed": 0,
    "models": ["..."], "model_revisions": {"...": "..."}
  },
  "diagnostics": {"n_grades": 4563, "parse_failure_rate": 0.0},
  "ratings": [{"model": "...", "revision": "...", "rating": 84.1, "ci_low": 82.0, "ci_high": 86.2, "ndcg_at_10": 0.34}]
}
```

`ndcg_at_10` is each model's score against the task's labels or the documents a query was written from: a baseline that needs no judge.

</details>

<details>
<summary>Cache layout</summary>

A rerun reuses everything here. The same configuration judges nothing new.

```text
~/.cache/mteb_gym/
├── queries/<task>/<query set>.json
├── predictions/<task>/<model>@<revision>/<query set>/
├── grades/<task>/<judge>-<key>.jsonl     # pointwise
└── verdicts/<task>/<pair>-<key>.jsonl    # pairwise
```

</details>

<details>
<summary>Comparing with MTEB</summary>

For an MTEB task, compare the ranking with the official scores:

```python
result.agreement()
```

</details>

<details>
<summary>Submitting results</summary>

Records go to the [results repository](https://github.com/embeddings-benchmark/gym-results), and the files behind them to the [runs dataset](https://huggingface.co/datasets/mteb/gym-runs):

```bash
gh auth login && hf auth login
mteb-gym submit              # commit to a local clone; nothing is uploaded
mteb-gym submit --create-pr  # open a pull request on each
```

</details>

## Choosing the judge and generator

Change a name to change a model. Give a URL for a model you serve yourself:

```python
gym.run(..., judge="gpt-5.4", generator="gpt-5.4-mini")  # with OPENAI_API_KEY set
gym.run(..., judge="autotrust/JEV-27B", judge_url="http://localhost:8001")
```

- **Generator:** any OpenAI-compatible chat model.
- **Judge:** a chat model, or a decision model with a `/v1/decide` endpoint such as [JEV-27B](https://huggingface.co/autotrust/JEV-27B). A decision model is found by its endpoint.
- **Models to compare:** any model `mteb` can load.

## Command line

```bash
mteb-gym run --corpus path/to/docs --models mteb/baseline-bm25s BAAI/bge-small-en-v1.5 \
    --judge autotrust/JEV-27B --judge-url http://localhost:8001 \
    --generator deepseek-ai/DeepSeek-V4-Flash --generator-url http://localhost:8000/v1
```

## Contributing

```bash
make install
make test
make lint
```

Tests need no API key or GPU.

## Citing

MTEB Gym builds on MTEB. A citation for MTEB Gym will follow with the paper.

<details>
<summary>BibTeX</summary>

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```

</details>

## License

Apache-2.0.
