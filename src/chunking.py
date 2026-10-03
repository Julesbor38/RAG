"""Chunking strategies: Markdown / text, JSON and Python code."""
import re
from typing import List, Pattern, Tuple
from .models import Chunk


# Cut points for Markdown / text, best first: before a heading, then
# after a blank line, a line end, or a space.
MARKDOWN_SEPARATORS: Tuple[Tuple[Pattern[str], int], ...] = (
    (re.compile(r"^#{1,6}\s", re.MULTILINE), 0),
    (re.compile(r"\n[ \t]*\n"), 1),
    (re.compile(r"\n"), 1),
    (re.compile(r" "), 1),
)


def _find_cut(text: str, start: int, end: int,
              separators: Tuple[Tuple[Pattern[str], int], ...]) -> int:
    """Return the last index in (start, end] matching the best separator."""
    for pattern, offset in separators:
        cut: int = -1
        for match in pattern.finditer(text, start, end):
            position: int = match.start() + offset
            if start < position <= end:
                cut = position
        if cut != -1:
            return cut
    return end


def chunk_markdown(text: str,
                   file_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split Markdown or plain text into chunks, preferably at headings.

    Args:
        text: Content of the file.
        file_path: Path of the file, stored in every chunk.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The non-blank chunks, covering the whole text in order.
    """
    chunks: List[Chunk] = []
    index: int = 0
    while index < len(text):
        end: int = min(index + max_chunk_size, len(text))
        if end == len(text):
            cut: int = end
        else:
            # Ignore separators too close to `index` to avoid tiny chunks.
            cut = _find_cut(text, index + max_chunk_size // 4, end,
                            MARKDOWN_SEPARATORS)
        chunk_text: str = text[index:cut]
        if chunk_text.strip():
            chunks.append(Chunk(file_path=file_path,
                                first_character_index=index,
                                last_character_index=cut,
                                text=chunk_text))
        index = cut
    return chunks


# Cut points for JSON, best first: after a closed object/array line,
# then any line end, then (for minified JSON) after a comma or space.
JSON_SEPARATORS = ("},\n", "],\n", "}\n", "]\n", "\n", ",", " ")


def _find_json_cut(text: str, start: int, end: int) -> int:
    """Return the best index in (start, end] at which to cut `text`."""
    for separator in JSON_SEPARATORS:
        position = text.rfind(separator, start, end)
        if position != -1:
            return position + len(separator)
    return end


def chunk_json(text: str,
               file_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split JSON into chunks, preferably after a closed object or array."""
    chunks: List[Chunk] = []
    index: int = 0
    while index < len(text):
        end: int = min(index + max_chunk_size, len(text))
        if end == len(text):
            cut: int = end
        else:
            # Ignore separators too close to `index` to avoid tiny chunks.
            cut = _find_json_cut(text, index + max_chunk_size // 4, end)
        chunk_text: str = text[index:cut]
        if chunk_text.strip():
            chunks.append(Chunk(file_path=file_path,
                                first_character_index=index,
                                last_character_index=cut,
                                text=chunk_text))
        index = cut
    return chunks


# Cut points for Python, best first, as (pattern, offset from match start):
# before a top-level def/class/decorator, then a nested one, then after
# a blank line, a line end, or a space.
PYTHON_SEPARATORS: Tuple[Tuple[Pattern[str], int], ...] = (
    (re.compile(r"^(?:@|(?:async\s+)?def\s|class\s)", re.MULTILINE), 0),
    (re.compile(r"^[ \t]+(?:@|(?:async\s+)?def\s|class\s)", re.MULTILINE), 0),
    (re.compile(r"\n[ \t]*\n"), 1),
    (re.compile(r"\n"), 1),
    (re.compile(r" "), 1),
)


def _follows_decorator(text: str, position: int) -> bool:
    """Return True if the line before `position` is a decorator."""
    previous_start: int = text.rfind("\n", 0, position - 1) + 1
    return text[previous_start:position].lstrip().startswith("@")


def _find_python_cut(text: str, start: int, end: int) -> int:
    """Return the best index in (start, end] at which to cut `text`."""
    for pattern, offset in PYTHON_SEPARATORS:
        cut: int = -1
        for match in pattern.finditer(text, start, end):
            position: int = match.start() + offset
            # Keep decorators attached to the def/class they decorate.
            if start < position <= end and not (
                    offset == 0 and _follows_decorator(text, position)):
                cut = position
        if cut != -1:
            return cut
    return end


def chunk_python(text: str,
                 file_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split JSON into chunks, preferably after a closed object or array."""
    chunks: List[Chunk] = []
    index: int = 0
    while index < len(text):
        end: int = min(index + max_chunk_size, len(text))
        if end == len(text):
            cut: int = end
        else:
            # Ignore separators too close to `index` to avoid tiny chunks.
            cut = _find_python_cut(text, index + max_chunk_size // 4, end)
        chunk_text: str = text[index:cut]
        if chunk_text.strip():
            chunks.append(Chunk(file_path=file_path,
                                first_character_index=index,
                                last_character_index=cut,
                                text=chunk_text))
        index = cut
    return chunks
