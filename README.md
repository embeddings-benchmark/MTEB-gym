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
  <a href="#usage">Usage</a> |
  <a href="#results">Results</a> |
  <a href="#serving-the-models">Serving the models</a> |
  <a href="#citing">Citing</a>
</h4>

MTEB Gym ranks embedding models on your own documents. You don't need labels or queries.

It works in four steps: write queries for your documents, let each model retrieve, have a judge grade the results, and rank the models.

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
    generator=gym.LLM("deepseek-ai/DeepSeek-V4-Flash", base_url="http://localhost:8000/v1"),
    judge=gym.DecisionModel("autotrust/JEV-27B", base_url="http://localhost:8001"),
)
print(result.leaderboard)
```

This needs a generator and a judge running. See [Serving the models](#serving-the-models).

## Usage

### Your documents

`corpus` can be any of these:

| Corpus | Example |
|---|---|
| A folder of `.txt` or `.md` files | `corpus="path/to/docs"` |
| A `.jsonl` file with `id` and `text` | `corpus="docs.jsonl"` |
| An MTEB task | `corpus="NFCorpus"` |

### Describe the task

Say what a good result is. It's optional, but recommended:

```python
gym.run(..., task_instruction="Given a support ticket, retrieve the help article that resolves it")
```

### Use your own queries

If you have real queries, use them. You then need no generator:

```python
gym.run(..., queries="my_queries.jsonl")  # or a .txt file, or a list of strings
gym.run(..., queries="original")  # an MTEB task's own queries
```

### Choose how to judge

```python
gym.run(..., judging="pointwise")  # grade each retrieved document
gym.run(..., judging="pairwise", pairs_per_query=26)  # compare two models' lists: slower
```

### Encode once

Keep each model's vectors, so a new run doesn't encode your documents again:

```python
gym.run(..., embedding_cache="path/to/vectors")
```

`help(gym.run)` lists every option.

## Results

Each run writes a record to `results/<task>/`. Read them back with:

```python
gym.load_results("results/").to_dataframe()
```

A rerun reuses its queries, retrieval and grades. Adding a model only grades the documents it adds.

For an MTEB task, `result.agreement()` compares the ranking with the official MTEB scores.

## Serving the models

- **Models to compare:** any model `mteb` can load.
- **Generator:** any OpenAI-compatible chat model.
- **Judge:** a decision model with a `/v1/decide` endpoint, or an LLM through `gym.LLM`.

For example, with vLLM:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash --port 8000 --trust-remote-code \
    --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4

hf download autotrust/JEV-27B --local-dir JEV-27B
python JEV-27B/serve_decide.py --model JEV-27B --served-model-name autotrust/JEV-27B --port 8001 \
    --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-27B/adapter_vllm \
    --logprobs-mode processed_logprobs
```

To use an API instead, set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for other providers.

## Command line

```bash
mteb-gym run --corpus path/to/docs \
    --models mteb/baseline-bm25s BAAI/bge-small-en-v1.5 \
    --generator deepseek-ai/DeepSeek-V4-Flash --generator-url http://localhost:8000/v1 \
    --judge autotrust/JEV-27B --judge-url http://localhost:8001 --judge-decision
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
