"""Build, persist and load the lexical index of the corpus."""
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List

from tqdm import tqdm

from .chunking import chunk_json, chunk_markdown, chunk_python
from .lexical import BM25Retriever
from .models import Chunk

logger = logging.getLogger(__name__)

CHUNKERS: Dict[str, Callable[[str, str, int], List[Chunk]]] = {
    ".md": chunk_markdown,
    ".json": chunk_json,
    ".py": chunk_python,
}

INDEX_FILE: str = "bm25_index.pkl"


def load_chunks(root: Path, max_chunk_size: int) -> List[Chunk]:
    """Chunk every supported file under `root`.

    Args:
        root: Directory holding the raw corpus.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks of every readable file, in path order.
    """
    files: List[Path] = sorted(path for path in root.rglob("*")
                               if path.suffix in CHUNKERS and path.is_file())
    chunks: List[Chunk] = []
    for path in tqdm(files, desc="Chunking", unit="file"):
        try:
            text = path.read_text(encoding="utf-8")
            chunks.extend(CHUNKERS[path.suffix](text, str(path),
                                                max_chunk_size))
        except (OSError, UnicodeDecodeError):
            logger.warning("Skipping unreadable file %s", path)
    return chunks


def save_index(retriever: BM25Retriever, index_dir: Path) -> Path:
    """Pickle `retriever` under `index_dir` and return the file path."""
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / INDEX_FILE
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(retriever, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def load_index(index_dir: Path) -> BM25Retriever:
    """Load the retriever saved by `save_index`.

    Raises:
        FileNotFoundError: If no index was built in `index_dir`.
        ValueError: If the index file is corrupted.
    """
    path = index_dir / INDEX_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No index found at {path}: run the `index` command first")
    try:
        with path.open("rb") as handle:
            retriever = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError,
            ImportError) as error:
        raise ValueError(f"Corrupted index {path}: {error}") from error
    if not isinstance(retriever, BM25Retriever):
        raise ValueError(f"Corrupted index {path}: unexpected content")
    return retriever
