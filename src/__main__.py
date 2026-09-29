"""Command-line interface of the RAG system, built with Python Fire.

Usage: uv run python -m src <command> [options]
"""
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
from pydantic import BaseModel
from tqdm import tqdm

from .indexing import load_chunks, load_index, save_index
from .lexical import BM25Retriever
from .models import (AnsweredQuestion, MinimalSearchResults,
                     MinimalSource, RagDataset, StudentSearchResults)

logger = logging.getLogger("src")

DEFAULT_RAW_DIR: str = "data/raw"
DEFAULT_INDEX_DIR: str = "data/processed"
MAX_CHUNK_SIZE: int = 2000
# A retrieved source matches a reference one when both are in the same
# file and their character ranges have at least this IoU.
MIN_IOU: float = 0.05
RECALL_KS: List[int] = [1, 3, 5, 10]


class CLIError(Exception):
    """An invalid input, reported to the user without a traceback."""


def check_int(value: Any, name: str, minimum: int,
              maximum: Optional[int] = None) -> int:
    """Return `value` if it is an int within bounds, else raise CLIError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CLIError(f"--{name} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (f"between {minimum} and {maximum}" if maximum is not None
                  else f"at least {minimum}")
        raise CLIError(f"--{name} must be {bounds}, got {value}")
    return value


def check_query(query: Any) -> str:
    """Return `query` as a non-empty string, else raise CLIError."""
    # Fire parses arguments as Python literals: `search 42` gives an int.
    text = str(query).strip() if query is not None else ""
    if not text:
        raise CLIError("The query must not be empty")
    return text


def existing_file(path: Any, name: str) -> Path:
    """Return `path` as a Path to an existing file, else raise CLIError."""
    file_path = Path(str(path))
    if not file_path.is_file():
        raise CLIError(f"--{name}: file not found: {file_path}")
    return file_path


def read_model(path: Path, model: type[BaseModel]) -> Any:
    """Parse the JSON file at `path` with the pydantic `model`."""
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise CLIError(f"{path} is not a valid {model.__name__} JSON "
                       f"file:\n{error}") from error


def write_model(data: BaseModel, directory: Any, file_name: str) -> Path:
    """Write `data` as JSON to `directory`/`file_name`, atomically."""
    output_dir = Path(str(directory))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / file_name
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(data.model_dump_json(indent=2),
                             encoding="utf-8")
        temporary.replace(output)
    except OSError as error:
        raise CLIError(f"Cannot write to {output_dir}: {error}") from error
    return output


def to_sources(retriever: BM25Retriever, query: str,
               k: int) -> List[MinimalSource]:
    """Return the top-`k` source locations of `query`."""
    return [MinimalSource(file_path=chunk.file_path,
                          first_character_index=chunk.first_character_index,
                          last_character_index=chunk.last_character_index)
            for chunk, _ in retriever.search(query, k)]


def is_match(retrieved: MinimalSource, reference: MinimalSource) -> bool:
    """Tell whether `retrieved` covers the region of `reference`."""
    if retrieved.file_path != reference.file_path:
        return False
    overlap = (min(retrieved.last_character_index,
                   reference.last_character_index)
               - max(retrieved.first_character_index,
                     reference.first_character_index))
    union = (max(retrieved.last_character_index,
                 reference.last_character_index)
             - min(retrieved.first_character_index,
                   reference.first_character_index))
    return overlap > 0 and union > 0 and bool(overlap / union >= MIN_IOU)


def recall_at(k: int, retrieved: List[MinimalSource],
              references: List[MinimalSource]) -> float:
    """Share of `references` matched by the first `k` retrieved sources."""
    found = sum(any(is_match(source, reference)
                    for source in retrieved[:k])
                for reference in references)
    return found / len(references)


class RAG:
    """Retrieval-Augmented Generation over the vLLM codebase."""

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE,
              raw_dir: str = DEFAULT_RAW_DIR,
              index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Ingest the corpus and build the index.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
            raw_dir: Directory holding the corpus to index.
            index_dir: Directory where the index is saved.
        """
        size = check_int(max_chunk_size, "max_chunk_size", 1, MAX_CHUNK_SIZE)
        root = Path(str(raw_dir))
        if not root.is_dir():
            raise CLIError(f"--raw_dir: directory not found: {root}")
        started = time.perf_counter()
        chunks = load_chunks(root, size)
        if not chunks:
            raise CLIError(f"No .py, .md or .json file to index in {root}")
        retriever = BM25Retriever(chunks)
        try:
            path = save_index(retriever, Path(str(index_dir)))
        except OSError as error:
            raise CLIError(f"Cannot save the index: {error}") from error
        logger.info("Indexed %d chunks in %.1fs into %s", len(chunks),
                    time.perf_counter() - started, path)
        print(f"Ingestion complete! Indices saved under {path.parent}/")

    def search(self, query: str, k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            index_dir: Directory where the index was saved.
        """
        text = check_query(query)
        count = check_int(k, "k", 1)
        retriever = load_index(Path(str(index_dir)))
        result = MinimalSearchResults(
            question_id="query", question=text,
            retrieved_sources=to_sources(retriever, text, count))
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10,
                       index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Search every question of a dataset.

        Writes a StudentSearchResults JSON file named like the dataset.

        Args:
            dataset_path: RagDataset JSON file of questions.
            save_directory: Directory where the results are written.
            k: Number of sources to return per question.
            index_dir: Directory where the index was saved.
        """
        count = check_int(k, "k", 1)
        path = existing_file(dataset_path, "dataset_path")
        dataset: RagDataset = read_model(path, RagDataset)
        retriever = load_index(Path(str(index_dir)))
        results = [MinimalSearchResults(
            question_id=item.question_id, question=item.question,
            retrieved_sources=to_sources(retriever, item.question, count))
            for item in tqdm(dataset.rag_questions, desc="Searching",
                             unit="question")]
        output = write_model(StudentSearchResults(search_results=results,
                                                  k=count),
                             save_directory, path.name)
        print(f"Saved student_search_results to {output}")

    def answer(self, query: str, k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR,
               context_sources: int = 5) -> None:
        """Answer a single query from the retrieved context.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve.
            index_dir: Directory where the index was saved.
            context_sources: How many of the best sources the model reads.
        """
        text = check_query(query)
        count = check_int(k, "k", 1)
        context = check_int(context_sources, "context_sources", 1)
        retriever = load_index(Path(str(index_dir)))
        search = StudentSearchResults(search_results=[MinimalSearchResults(
            question_id="query", question=text,
            retrieved_sources=to_sources(retriever, text, count))], k=count)
        # Imported lazily: loading torch and transformers is slow.
        from .generation import AnswerGenerator, answer_all
        result = answer_all(search, AnswerGenerator(), context)
        print(result.search_results[0].model_dump_json(indent=2))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str,
                       context_sources: int = 5) -> None:
        """Generate answers for every question of search results.

        Writes a StudentSearchResultsAndAnswer JSON file named like the
        search results file.

        Args:
            student_search_results_path: StudentSearchResults JSON file.
            save_directory: Directory where the answers are written.
            context_sources: How many of the best sources the model reads.
        """
        context = check_int(context_sources, "context_sources", 1)
        path = existing_file(student_search_results_path,
                             "student_search_results_path")
        search: StudentSearchResults = read_model(path, StudentSearchResults)
        print(f"Loaded {len(search.search_results)} questions from {path}")
        from .generation import AnswerGenerator, answer_all
        results = answer_all(search, AnswerGenerator(), context)
        output = write_model(results, save_directory, path.name)
        print(f"Saved student_search_results_and_answer to {output}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@k of search results against a ground truth.

        Args:
            student_search_results_path: StudentSearchResults JSON file.
            dataset_path: RagDataset JSON file of answered questions.
        """
        results_path = existing_file(student_search_results_path,
                                     "student_search_results_path")
        truth_path = existing_file(dataset_path, "dataset_path")
        search: StudentSearchResults = read_model(results_path,
                                                  StudentSearchResults)
        dataset: RagDataset = read_model(truth_path, RagDataset)
        references = [item for item in dataset.rag_questions
                      if isinstance(item, AnsweredQuestion) and item.sources]
        if not references:
            raise CLIError(f"{truth_path} has no question with sources")
        retrieved: Dict[str, List[MinimalSource]] = {
            result.question_id: result.retrieved_sources
            for result in search.search_results}
        missing = sum(item.question_id not in retrieved
                      for item in references)
        if missing:
            logger.warning("%d of %d questions have no search result "
                           "(counted as 0)", missing, len(references))
        ks = sorted({k for k in RECALL_KS if k <= search.k} | {search.k})
        scores = [
            sum(recall_at(k, retrieved.get(item.question_id, []),
                          item.sources) for item in references)
            / len(references)
            for k in ks]
        print(f"Evaluated {len(references)} questions (k={search.k})")
        print(" ".join(f"Recall@{k}: {score:.3f}"
                       for k, score in zip(ks, scores)))


def main() -> int:
    """Run the CLI and turn errors into messages and exit codes."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    try:
        fire.Fire(RAG)
    except (CLIError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
