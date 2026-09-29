import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from .models import Chunk

# Split camelCase before lowercasing so "chunkMarkdown" -> "chunk markdown";
# snake_case is split by the token pattern itself.
CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase `text` and split it into alphanumeric tokens."""
    return TOKEN.findall(CAMEL_CASE.sub(" ", text).lower())


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
        counts: List[Counter[str]] = [Counter(tokenize(chunk.text))
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
