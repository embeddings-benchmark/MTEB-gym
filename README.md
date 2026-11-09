<h1 align="center">MTEB Gym</h1>

<h3 align="center">Evaluate embedding models on any corpus, without relevance labels</h3>

<p align="center">
  <a href="https://github.com/embeddings-benchmark/MTEB-gym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/embeddings-benchmark/MTEB-gym.svg" alt="License"></a>
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

You can install MTEB Gym with pip:

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym.git"
```

Some embedding models need an extra of `mteb`, which their error names, and a few older ones need transformers 4.

## Quickstart

The judge here, decider-4b, needs the `decider` extra: `pip install "mteb-gym[decider] @ git+https://github.com/embeddings-benchmark/MTEB-gym.git"`.

```python
import mteb_gym as gym

result = gym.run(
    corpus="NFCorpus",  # an MTEB task, or "path/to/docs" for your own documents
    models=["mteb/baseline-bm25s", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"],
    judge="Mapika/decider-4b",
    generator="Qwen/Qwen3.5-4B",
    judging="pointwise",  # or "pairwise" to compare whole result lists
    n_queries=20,  # 100 for a real evaluation
)
print(result.leaderboard)
```

Runs on one GPU with about 20 GB of memory.

The judge and the generator both load from Hugging Face. [Change them or serve them](#choosing-the-judge-and-generator) by changing a name.

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

Pointwise is the default. Set `judging="pairwise"` to compare lists. It costs more: 100 queries and 10 models take 9,000 calls, which `pairs_per_query` reduces.

## Usage

### Your documents

| Corpus | Example |
|---|---|
| A folder of `.txt` or `.md` files | `corpus="path/to/docs"` |
| A `.jsonl` file with `id` and `text` | `corpus="docs.jsonl"` |
| An MTEB task | `corpus="NFCorpus"` |

### Describe the task

Describe the retrieval task in one sentence. The generator and the judge both get it. It's optional, but recommended:

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
| `task_instruction` | `None` | One sentence describing the retrieval task |
| `judging` | `"pointwise"` | `"pointwise"` or `"pairwise"` |
| `pairs_per_query` | All pairs | Pairwise: judge this many random model pairs per query instead |
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

Each run also writes a record to `results/<task>/`. Read them back with `gym.load_results("results/").to_dataframe()`, or read the shared results repository with `gym.load_results()`, which fetches it into the cache as `mteb.load_results()` does.

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

Give a name. The model loads from Hugging Face and runs in your Python process, and the run prints what it found:

```text
judge: Mapika/decider-4b (decision model)
generator: Qwen/Qwen3.5-4B (chat model)
```

**Judge.** A decision model answers in one forward pass, in both judging modes. A chat model also works, and is slower.

- In your Python process: the judges listed below.
- On your own server: any decision model with a `/v1/systemone` endpoint, or any chat model.

These judges work by name, at pinned versions, which `gym.get_model_metas()` lists:

- `autotrust/JEV-9B`
- `autotrust/JEV-27B`, used in the paper
- `autotrust/JEV-35B`
- `autotrust/GEV-26B-Decide`
- `Mapika/decider-4b`

The first four need the `transformers5` extra and decider-4b the `decider` extra. Their error says how to install it.

**Generator.** Any chat model, such as `Qwen/Qwen3.5-4B`, or `gpt-5.4-mini` through an API with `OPENAI_API_KEY` set.

**Models to compare.** Any model `mteb` can load, by name, or the model itself: a `ModelMeta` or a loaded mteb model. A variant, such as a wrapper, is ranked as its own model when its `mteb_model_meta` has its own name or `experiment_kwargs`, as mteb's wrappers do. Two different models with one name are refused.

To use a model on your own server, add its URL. If a model doesn't fit in your process, the error links its model card, which says how to serve it.

```python
gym.run(..., judge="autotrust/JEV-27B", judge_url="http://localhost:8001/v1")
```

<details>
<summary>Serving with vLLM</summary>

A chat model:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000
```

A decision model, with its decision module:

```bash
hf download autotrust/JEV-9B --revision 7ac669b8fb476ba926491e3d1c39f9330d68dbed --local-dir JEV-9B
vllm serve JEV-9B --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-9B/adapter_vllm \
    --logprobs-mode processed_logprobs --port 8001
```

vLLM takes most of a GPU's memory. To share one with a second model, add `--gpu-memory-utilization` with `0.5` for the decision model or `0.4` for the chat model.

</details>

<details>
<summary>Other decision models</summary>

These run on a server, as their model cards describe: `LiquidAI/d1-3B`, `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` and `autotrust/AutoTrust-JEV-27B-v1.1`. Pass the name and `judge_url`.

A decision model on a server with `/v1/systemone`: pass its name and `judge_url` if its repo has a `decision_config.json` or `decider_config.json`. If not, say it is one:

```python
judge = gym.DecisionModel("org/my-model", base_url="http://localhost:8000/v1")
```

For a format with no wrapper yet, add one in [`mteb_gym/models/`](mteb_gym/models), or pass your own object:

```python
class MyJudge:
    model = "org/my-model"

    def decide(self, kind, state, question, options=None):
        # kind is "score", with grades 0 to 5, or "choice"; a probability per option
        return {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1, "5": 0.0}
```

</details>

## Leaderboard

A benchmark is a fixed set of tasks judged one way. `gym.get_benchmarks()` lists them. The main one, `Gym(eng, v1)`, is 28 tasks on synthetic queries judged by JEV-27B. Its table, from the shared results repository, or from a folder with `gym.load_results("path/to/results")`:

```python
results = gym.load_results()
results.get_benchmark_result("Gym(eng, v1)")
```

The leaderboard app needs the `leaderboard` extra:

```bash
pip install "mteb-gym[leaderboard] @ git+https://github.com/embeddings-benchmark/MTEB-gym.git"
python -m mteb_gym.leaderboard
```

## Command line

```bash
mteb-gym run --corpus path/to/docs --models mteb/baseline-bm25s BAAI/bge-small-en-v1.5 \
    --judge autotrust/JEV-9B --generator Qwen/Qwen3.5-4B
```

## Contributing

```bash
make install
make test
make lint
```

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
