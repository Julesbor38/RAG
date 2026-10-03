*This project has been created as part of the 42 curriculum by jbordeli.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about
the [vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase. It:

1. **indexes** the repository (Python code, Markdown docs, text files) into
   chunks of at most 2000 characters, scored with BM25;
2. **retrieves** the top-k source locations (file path + character range)
   for a question, alone or for a whole JSON dataset;
3. **generates** a grounded answer with the local `Qwen/Qwen3-0.6B` model;
4. **evaluates** its own retrieval with recall@k.

Every structure exchanged between stages is a pydantic model
(`src/models.py`), so every JSON output is valid by construction.

## Instructions

Requirements: Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). The vLLM
repository must be in `data/raw/vllm-0.10.1/` and the datasets in
`data/datasets/{AnsweredQuestions,UnansweredQuestions}/`.

```bash
make install         # uv sync
make run ARGS=index  # uv run python -m src index
make lint            # flake8 + mypy
make debug ARGS=...  # run under pdb
make clean           # remove caches
```

All commands are run as `uv run python -m src <command> [options]`:

| Command | Options |
|---------|---------|
| `index` | `--max_chunk_size` (default 2000, max 2000), `--raw_dir`, `--index_dir` |
| `search <query>` | `--k` (default 10), `--index_dir` |
| `search_dataset` | `--dataset_path`, `--save_directory`, `--k`, `--index_dir` |
| `answer <query>` | `--k`, `--context_sources` (default 5), `--index_dir` |
| `answer_dataset` | `--student_search_results_path`, `--save_directory`, `--context_sources` |
| `evaluate` | `--student_search_results_path`, `--dataset_path` |

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indices saved under data/processed/

uv run python -m src search "How do I enable LoRA adapters?" --k 5

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

```
data/raw/ ──► chunking.py ──► lexical.BM25Retriever ──► data/processed/bm25_index.pkl
                (per file type)     (inverted index)          (pickle, indexing.py)
                                                                   │
question ──► tokenize ──► BM25 top-k ──► MinimalSource list ◄──────┘
                                              │
                         read the k spans from disk (generation.py)
                                              │
                    Qwen3-0.6B prompt (token budget) ──► cleaned answer
                                              │
                         StudentSearchResultsAndAnswer JSON
```

- `src/__main__.py`: Fire CLI, input validation, error handling, JSON I/O
  and the `evaluate` command.
- `src/chunking.py`: the chunking strategies.
- `src/lexical.py`: tokenizer, BM25 (used) and TF-IDF (alternative) retrievers.
- `src/indexing.py`: walks the corpus, builds and persists the index.
- `src/generation.py`: prompt building, generation and answer cleaning.
- `src/models.py`: pydantic data models.

## Chunking strategy

Each chunker cuts the text into consecutive, non-overlapping spans of at most
`--max_chunk_size` characters, and looks for the *best* cut point in the
last three quarters of the window (so no tiny chunks):

- **Python** (`.py`): before a top-level `def`/`class`/decorator, then a
  nested one, then a blank line, a newline, a space. Decorators stay
  attached to their function.
- **Markdown / text** (`.md`, `.txt`, `.rst`): before a heading, then a
  blank line (paragraph), a newline, a space.
- **JSON** (`.json`): after a closed object or array line, then a newline,
  a comma, a space.

Character offsets are kept exactly, so a chunk is directly a valid
`MinimalSource`.

## Retrieval method

Okapi BM25 (k1 = 1.5, b = 0.75, Lucene-style idf) over an inverted index.
The tokenizer splits `camelCase` and `snake_case`, lowercases, drops a short
list of English stop words and strips plural "s". The tokens of the file
path (relative to the repository) are added to each chunk, because
questions often name the topic a file is called after (e.g.
`docs/serving/data_parallel_deployment.md`).

## Performance analysis

Scores from the provided moulinette on the public datasets (k = 10,
chunk size 2000):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required |
|---------|---------:|---------:|---------:|----------:|---------:|
| docs    | 0.63 | 0.81 | **0.84** | 0.87 | 0.80 |
| code    | 0.36 | 0.62 | **0.72** | 0.81 | 0.50 |

Effect of each improvement on Recall@5 (docs / code):

| Version | docs | code |
|---------|-----:|-----:|
| BM25, `.py/.md/.json` only, newline cuts in Markdown | 0.80 | 0.62 |
| + `.txt` files, heading-aware Markdown cuts, path tokens | 0.82 | 0.73 |
| + stop words and plural stemming | 0.84 | 0.72 |

Smaller chunks lower recall@5: 0.81 / 0.66 at 1000 characters and
0.81 / 0.68 at 1500, so the default stays at 2000. Changing k1 and b
(1.2–2.0, 0.5–0.9) gave no consistent gain.

Speed on a CPU laptop: indexing ≈ 1.5 s for ~14,700 chunks (limit 5 min);
searching 100 questions ≈ 1 s (limit 90 s for 200). Answering takes a few
seconds per question once the model is loaded.

## Design decisions

- **BM25 rather than TF-IDF**: it saturates term frequency and normalizes
  by length, which works better on chunks of uneven size. TF-IDF is still
  in `lexical.py` for comparison.
- **Pickled retriever**: loading it takes well under a second, so search
  starts quickly.
- **Lazy import of torch/transformers**: `index`, `search` and `evaluate`
  never pay the cost of loading the model.
- **The model only writes the answer text**: the JSON is built by pydantic,
  so it is always valid. Thinking mode is disabled, decoding is greedy with
  a repetition penalty, and the output is cleaned (no `<think>` block, no
  "[Source 1]" mentions, cut at the last full sentence if truncated).
- **Token budget**: the best sources are added in rank order until about
  3000 tokens, and the last one is truncated if needed.
- **Errors**: every invalid input (empty query, k < 1, missing file,
  malformed JSON, missing index) prints a one-line `Error:` and exits with
  code 1, never a traceback.

## Challenges faced

- **Docs recall stuck at 80 %**: looking at the misses showed that
  `CMakeLists.txt` was never indexed and that many questions match a file
  *name* more than its content. Indexing text files and path tokens fixed it.
- **Never exceeding 2000 characters**: one source that is too long makes
  the whole output invalid, so every chunker caps the window, and the CLI
  refuses a larger `--max_chunk_size`.
- **Small model behaviour**: Qwen3-0.6B tends to think aloud, cite
  "Source 2" or ramble; a strict system prompt plus post-processing keeps
  the answers short and grounded.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009).
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
  chapters 6 and 11.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020).
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Hugging Face transformers docs](https://huggingface.co/docs/transformers),
  [Pydantic docs](https://docs.pydantic.dev/),
  [Python Fire](https://github.com/google/python-fire).

**Use of AI**: an AI assistant (Claude) was used to review the code, to help
analyse the retrieval misses and tune the chunking/tokenization (text files,
heading-aware cuts, path tokens, stemming), and to draft this README. All
the code was reviewed, tested against the moulinette and understood before
being kept.
