# MTEB Gym

Label-free, LLM-judged evaluation of embedding models.

Give it a corpus, either an MTEB task or your own documents. It generates queries for the corpus, or takes yours. Every model retrieves for the same queries, an LLM judge compares the retrieved lists pairwise, and Bradley–Terry turns the comparisons into a ranking. No relevance labels are needed.

```text
corpus → queries → retrieval → pairwise LLM judging → ranking
```

Started in the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068).

## Installation

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym-v2"
```

`[colbert]` adds late-interaction models.

## Quickstart

The mock LLM needs no API key or GPU. It answers deterministically, so this checks the install and does not rank models. The first run downloads a small dataset and MiniLM; allow a few minutes.

```python
import mteb_gym as gym

result = gym.run(
    corpus="NanoNFCorpusRetrieval",
    models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
    generator=gym.MockLLM(),
    judge=gym.MockLLM(),
    n_queries=20,
    output_folder="results/demo",
)
print(result.leaderboard)
```

With `OPENAI_API_KEY` set, `gym.LLM(model)` uses OpenAI. Any other OpenAI-compatible provider, such as OpenRouter, Together, Anthropic or Gemini, works with its URL in `OPENAI_BASE_URL` and its key.

```bash
export OPENAI_API_KEY=<your_api_key>
```

```python
import mteb_gym as gym

result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    generator=gym.LLM("gpt-5.4-mini"),  # writes the queries
    judge=gym.LLM("gpt-5.4"),  # compares what the models retrieve
    n_queries=100,
    output_folder="results/nfcorpus",
)
print(result.leaderboard)
```

Or from the command line:

```bash
mteb-gym --corpus NFCorpus \
    --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 intfloat/e5-base-v2 \
    --generator gpt-5.4-mini --judge gpt-5.4
```

## Open models

`gym.LLM` talks to any OpenAI-compatible server. To use an open model, serve it:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

Then point the client at it. Like the openai SDK, it reads these two variables, or takes them as the `base_url` and `api_key` arguments. A local server accepts any key.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
```

```python
import mteb_gym as gym

llm = gym.LLM("Qwen/Qwen3-4B-Instruct-2507")

result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    generator=llm,
    judge=llm,
    n_queries=100,
    output_folder="results/nfcorpus",
)
print(result.leaderboard)
```

SGLang and `transformers serve` expose the same endpoint. A judge and a generator from different model families are preferred.

## Usage

`gym.run` takes the corpus, the models and the two LLMs, as in the Quickstart. The other options, with their defaults:

- `queries="synthetic"`: `"original"` uses `n_queries` of the task's own queries, drawn with `seed`; your own go in as a `.jsonl` with `id` and `text`, a `.txt` with one per line, or a list of strings.
- `task_description=None`: one sentence on what counts as a good result, given to the generator and the judge. By default an MTEB task's own criterion, or plain relevance if it has none. Every run logs the one it used.
- `n_queries=100`, `top_k=10` documents judged per query, `doc_chars=2000` characters of each shown to the judge, `pairs_per_query=None` to judge only that many random model pairs per query, `seed=0`, `filter_queries=True` for the LLM quality filter and deduplication, `output_folder="results"`, `batch_size=32`, `workers=8` concurrent LLM calls.

`help(gym.run)` documents each one.

Everything is written under `output_folder`. The record holds the ratings, the configuration, the diagnostics, and each model's nDCG@10 against the seed documents or the dataset's labels, a baseline that needs no judge:

```text
results/nfcorpus/
├── records/NFCorpus__gpt-5.4__gpt-5.4-mini__q100-s0-<hash>.json
├── queries/       # generated queries with quality scores
├── predictions/   # mteb's retrieval output per model
└── verdicts/      # judge verdicts per model pair
```

- **Reruns.** The same configuration reuses all of it; adding a model judges only the new pairs.
- **Many models on one GPU.** Retrieval loads one model at a time, and a long roster can exhaust the GPU in one process. Run the models one per process so the operating system reclaims each, then judge:

  ```bash
  for m in "${MODELS[@]}"; do
      mteb-gym predict --corpus NFCorpus --model "$m" --generator gpt-5.4-mini --output-folder results/nfcorpus
  done
  mteb-gym run --corpus NFCorpus --models "${MODELS[@]}" --generator gpt-5.4-mini --judge gpt-5.4 --output-folder results/nfcorpus
  ```

  `run` finds the prediction files already written and only judges. The arguments that decide the query set, the corpus, generator, `queries`, `n_queries` and `seed`, must match between the two.
- **Cost.** Two judge calls per query per model pair: 100 queries and 10 models is 9,000 calls.
- **Reading back.** `gym.Result.from_disk(path)` for one run (`.leaderboard`, `.to_dataframe()`); `gym.load_results("results/")` for every run under a directory.

## Agreement with MTEB

For an MTEB task, compare the ranking with the official scores after the run. The labels never enter the pipeline. `queries="original"` runs the same comparison on the dataset's own queries instead of generated ones.

```python
result.agreement()  # one run
gym.load_results("results/").agreement()  # every run under a directory
```

- Official scores come from the MTEB results repository through mteb's cache; `agreement(evaluate_missing=True)` runs mteb for models that have none.
- Reported: Spearman, Kendall, top-10 Spearman and AP correlation with bootstrap intervals, and whether each score was official or self-run.

## How it works

1. **Generate queries**  
   Sample documents; the generator writes one query per sample.

2. **Filter queries**  
   Drop short, malformed, low-quality and near-duplicate queries.

3. **Retrieve**  
   Every model retrieves through `mteb.evaluate`.

4. **Judge pairwise**  
   The judge compares two models' top-k lists per query, in both orders. A split decision counts half.

5. **Rank models**  
   Bradley–Terry over all pairwise outcomes; confidence intervals from resampling queries.

6. **Record the run**  
   Queries, predictions, verdicts, model revisions and configuration are written to disk and cached, so a rerun repeats only what changed.

## Development

```bash
make install   # uv sync with dev tools
make test      # pytest, two end-to-end runs included
make lint      # ruff
```

Tests use the mock LLM: no key, no GPU. CI runs on Python 3.10 and 3.13.

```text
mteb_gym/
├── corpus.py     an MTEB task corpus or a local one
├── queries.py    query generation and filtering
├── task.py       corpus + queries as an mteb retrieval task
├── retrieval.py  mteb.evaluate per model; read its predictions
├── judge.py      pairwise judging, both orders
├── rank.py       Bradley–Terry with bootstrap CIs
├── results.py    Result, Results, load_results
├── agreement.py  agreement with official MTEB scores
├── llm.py        LLM and MockLLM
└── run.py        the pipeline
```

## Citation

Paper and citation information will be added with the MTEB Gym release.

MTEB Gym builds on MTEB:

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
