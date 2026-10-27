# MTEB Gym

Evaluate embedding models on any corpus, without relevance labels.

MTEB Gym writes queries for your corpus, lets every model retrieve for them, has a judge grade what they retrieved, and ranks the models by nDCG over those grades.

```text
corpus → queries → retrieval → judging → ranking
```

## Installation

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym"
```

## Quickstart

You need two models behind HTTP endpoints: an LLM that writes the queries and a decision model that grades the results. Serve each with vLLM on its own GPU (or use an API for the generator, see below):

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000            # generator

hf download autotrust/JEV-27B --local-dir JEV-27B              # judge
python JEV-27B/serve_decide.py --model JEV-27B --served-model-name autotrust/JEV-27B --port 8001 \
    --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-27B/adapter_vllm \
    --logprobs-mode processed_logprobs --max-model-len 32768
```

Then rank models on a corpus:

```python
import mteb_gym as gym

result = gym.run(
    corpus="SciFact",  # an MTEB task, a folder of .txt/.md files, or a .jsonl with id and text
    models=["mteb/baseline-bm25s", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"],
    generator=gym.LLM("Qwen/Qwen3-4B-Instruct-2507", base_url="http://localhost:8000/v1"),
    judge=gym.DecisionModel("autotrust/JEV-27B", base_url="http://localhost:8001"),
)
print(result.leaderboard)
```

Or from the command line:

```bash
mteb-gym run --corpus SciFact --models mteb/baseline-bm25s BAAI/bge-small-en-v1.5 intfloat/e5-base-v2 \
    --generator Qwen/Qwen3-4B-Instruct-2507 --generator-url http://localhost:8000/v1 \
    --judge autotrust/JEV-27B --judge-url http://localhost:8001 --judge-decision
```

## Choosing models

- **Generator:** any OpenAI-compatible chat model. Use an API (set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for providers other than OpenAI) or serve an open model with vLLM, SGLang or Ollama. Larger generators write better queries.
- **Judge:** a decision model behind a `/v1/decide` endpoint, as above. Any object with a `decide(kind, state, question, options)` method works too. An LLM (`gym.LLM`) can also judge; it then compares the models' result lists pairwise, which is slower.

## Options

```python
gym.run(
    corpus, models, judge, generator,
    queries="synthetic",   # "original" for an MTEB task's own queries, or your own: list of strings, .txt or .jsonl
    n_queries=100,
    top_k=10,              # documents judged per query
    output_folder="results",
    cache_folder=None,     # default $MTEB_GYM_CACHE, else ~/.cache/mteb_gym
)
```

`help(gym.run)` lists the rest. A rerun with the same settings reuses the cached queries, retrievals and grades; adding a model grades only what it adds.

## Outputs

Each run writes one JSON record to `output_folder/<task>/`: the leaderboard with confidence intervals, the configuration and diagnostics.

```python
gym.Result.from_disk(path).leaderboard
gym.load_results("results/").to_dataframe()
```

For an MTEB task, `result.agreement()` compares the ranking with the official MTEB scores.

## Submitting results

Records go to the [results repository](https://github.com/embeddings-benchmark/gym-results); the queries, retrievals and grades behind them go to the [gym-runs dataset](https://huggingface.co/datasets/mteb/gym-runs).

```bash
gh auth login && hf auth login
mteb-gym submit --create-pr
```

## Development

```bash
make install   # uv sync with dev tools
make test      # pytest; no API key or GPU needed
make lint      # ruff
```

## Citation

Paper and citation information will follow with the release. MTEB Gym builds on MTEB:

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
