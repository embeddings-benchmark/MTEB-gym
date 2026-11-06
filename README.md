<h1 align="center">MTEB Gym</h1>

<h3 align="center">Evaluate embedding models on any corpus, without relevance labels</h3>

<p align="center">
  <a href="https://pypi.org/project/mteb-gym/"><img src="https://img.shields.io/pypi/v/mteb-gym.svg" alt="PyPI"></a>
  <a href="https://github.com/embeddings-benchmark/MTEB-gym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/embeddings-benchmark/MTEB-gym.svg" alt="License"></a>
  <a href="https://pepy.tech/project/mteb-gym"><img src="https://static.pepy.tech/badge/mteb-gym" alt="Downloads"></a>
</p>

<h4 align="center">
  <a href="#installation">Installation</a> |
  <a href="#example-usage">Usage</a> |
  <a href="#models">Models</a> |
  <a href="#overview">Overview</a> |
  <a href="#citing">Citing</a>
</h4>

MTEB Gym writes queries for your corpus, lets each embedding model retrieve for them, has a judge grade what they retrieved, and ranks the models. You don't need labels or queries of your own, only your documents.

## Installation

```bash
pip install mteb-gym
```

## Example Usage

```python
import mteb_gym as gym

result = gym.run(
    corpus="path/to/docs",  # a folder of .txt/.md files, a .jsonl with id and text, or an MTEB task
    task_instruction="Given a support ticket, retrieve the help article that resolves it",  # optional, recommended
    models=["mteb/baseline-bm25s", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"],
    generator=gym.LLM("deepseek-ai/DeepSeek-V4-Flash", base_url="http://localhost:8000/v1"),
    judge=gym.DecisionModel("autotrust/JEV-27B", base_url="http://localhost:8001"),
)
print(result.leaderboard)
```

This expects the generator and the judge to be running, see [Models](#models). Or use the command line:

```bash
mteb-gym run --corpus path/to/docs \
    --task-instruction "Given a support ticket, retrieve the help article that resolves it" \
    --models mteb/baseline-bm25s BAAI/bge-small-en-v1.5 intfloat/e5-base-v2 \
    --generator deepseek-ai/DeepSeek-V4-Flash --generator-url http://localhost:8000/v1 \
    --judge autotrust/JEV-27B --judge-url http://localhost:8001 --judge-decision
```

## Models

- **Models to compare:** any model id that `mteb` can load.
- **Generator:** writes the queries. Any OpenAI-compatible chat model works, either from an API (set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for other providers) or served with vLLM, SGLang or Ollama.
- **Judge:** a decision model behind a `/v1/decide` endpoint, any object with a `decide(kind, state, question, options)` method, or an LLM (`gym.LLM`). By default a decision model grades each retrieved document, and an LLM compares the models' lists. `judging="pointwise"` or `"pairwise"` chooses either way for either kind; pairwise costs more.

<details>
<summary>Serving the generator and the judge with vLLM</summary>

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash --port 8000 --trust-remote-code --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4

hf download autotrust/JEV-27B --local-dir JEV-27B
python JEV-27B/serve_decide.py --model JEV-27B --served-model-name autotrust/JEV-27B --port 8001 \
    --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-27B/adapter_vllm \
    --logprobs-mode processed_logprobs --max-model-len 32768
```

</details>

## Overview

| Topic | |
|---|---|
| **Queries** | `queries="original"` uses an MTEB task's own queries; your own go in as a list, `.txt` or `.jsonl`. `help(gym.run)` lists every option |
| **Encoding once** | `embedding_cache="path"` keeps each model's vectors, and `max_seq_length=8192` caps very long models. By default models run as `mteb` loads them |
| **Outputs** | Each run writes a JSON record to `results/<task>/`. Read them back with `gym.load_results("results/").to_dataframe()` |
| **Caching** | A rerun reuses its queries, retrievals and grades. Adding a model only grades the documents it adds |
| **MTEB comparison** | For an MTEB task, `result.agreement()` compares the ranking with the official MTEB scores |
| **Submitting** | `mteb-gym submit --create-pr` opens pull requests to the [results repository](https://github.com/embeddings-benchmark/gym-results) and the [runs dataset](https://huggingface.co/datasets/mteb/gym-runs) |
| **Contributing** | `make install`, `make test` and `make lint`. Tests need no API key or GPU |

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
