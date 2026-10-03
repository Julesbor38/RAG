"""Lexical retrievers: TF-IDF and Okapi BM25."""
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .models import Chunk

# Split camelCase before lowercasing so "chunkMarkdown" -> "chunk markdown";
# snake_case is split by the token pattern itself.
CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN = re.compile(r"[a-z0-9]+")


STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or "
    "the to what when where which who why with you your".split())


def stem(token: str) -> str:
    """Strip a plural "s" so "models" and "model" match."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase `text`, split it into tokens, drop stop words and stem."""
    return [stem(token) for token
            in TOKEN.findall(CAMEL_CASE.sub(" ", text).lower())
            if token not in STOP_WORDS]


def path_tokens(file_path: str) -> List[str]:
    """Return the tokens of the corpus-relative part of `file_path`."""
    parts = Path(file_path).parts
    # Drop the "data/raw/<repository>" prefix shared by every chunk.
    relative = parts[3:] if len(parts) > 3 and parts[:2] == ("data", "raw") \
        else parts
    return tokenize(" ".join(relative))


class TfidfRetriever:
    """Rank chunks by cosine similarity of their TF-IDF vectors."""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks: List[Chunk] = chunks
        counts: List[Counter[str]] = [Counter(tokenize(chunk.text))
                                      for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for count in counts:
            document_frequency.update(count.keys())
        total: int = len(chunks)
        # Smoothed idf, as in scikit-learn: never zero, never negative.
        self.idf: Dict[str, float] = {
            term: math.log((1 + total) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        # Inverted index: term -> [(chunk index, normalized weight)].
        self.index: Dict[str, List[Tuple[int, float]]] = {}
        for chunk_index, count in enumerate(counts):
            vector = self._vectorize(count)
            for term, weight in vector.items():
                self.index.setdefault(term, []).append((chunk_index, weight))

    def _vectorize(self, count: Counter[str]) -> Dict[str, float]:
        """Return the L2-normalized, log-scaled TF-IDF vector of `count`."""
        vector: Dict[str, float] = {
            term: (1 + math.log(frequency)) * self.idf[term]
            for term, frequency in count.items() if term in self.idf
        }
        norm: float = math.sqrt(sum(w * w for w in vector.values()))
        if norm == 0:
            return {}
        return {term: weight / norm for term, weight in vector.items()}

    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return the `k` best chunks for `query`, with their scores."""
        scores: Dict[int, float] = {}
        for term, query_weight in self._vectorize(
                Counter(tokenize(query))).items():
            for chunk_index, weight in self.index[term]:
                scores[chunk_index] = (scores.get(chunk_index, 0.0)
                                       + query_weight * weight)
        best = sorted(scores.items(), key=lambda item: item[1],
                      reverse=True)[:k]
        return [(self.chunks[i], score) for i, score in best]


class BM25Retriever:
    """Rank chunks with Okapi BM25."""

    def __init__(self, chunks: List[Chunk],
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: List[Chunk] = chunks
        self.k1: float = k1
        self.b: float = b
        # File path tokens are indexed too: a question often names the
        # topic a file is named after (e.g. "data parallel deployment").
        counts: List[Counter[str]] = [
            Counter(tokenize(chunk.text) + path_tokens(chunk.file_path))
            for chunk in chunks]
        self.lengths: List[int] = [sum(count.values()) for count in counts]
        self.average_length: float = (sum(self.lengths) / len(chunks)
                                      if chunks else 0.0)
        # Inverted index: term -> [(chunk index, term frequency)].
        self.index: Dict[str, List[Tuple[int, int]]] = {}
        for chunk_index, count in enumerate(counts):
            for term, frequency in count.items():
                self.index.setdefault(term, []).append(
                    (chunk_index, frequency))
        total: int = len(chunks)
        # Lucene-style idf: the "+ 1" keeps it positive for common terms.
        self.idf: Dict[str, float] = {
            term: math.log((total - len(postings) + 0.5)
                           / (len(postings) + 0.5) + 1)
            for term, postings in self.index.items()
        }

    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return the `k` best chunks for `query`, with their scores."""
        scores: Dict[int, float] = {}
        for term in set(tokenize(query)):
            if term not in self.index:
                continue
            idf: float = self.idf[term]
            for chunk_index, frequency in self.index[term]:
                length_ratio = self.lengths[chunk_index] / self.average_length
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio)
                scores[chunk_index] = (scores.get(chunk_index, 0.0)
                                       + idf * frequency * (self.k1 + 1)
                                       / denominator)
        best = sorted(scores.items(), key=lambda item: item[1],
                      reverse=True)[:k]
        return [(self.chunks[i], score) for i, score in best]
