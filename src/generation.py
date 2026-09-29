"""Grounded answer generation with Qwen3-0.6B.

The model only ever writes the answer *text*: the JSON output is built
and validated by the Pydantic models, so it is always valid JSON whatever
the model produces.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import (MinimalAnswer, MinimalSource, StudentSearchResults,
                     StudentSearchResultsAndAnswer)

logger = logging.getLogger(__name__)

MODEL_NAME: str = "Qwen/Qwen3-0.6B"
NOT_FOUND: str = "I could not find the answer in the provided sources."

SYSTEM_PROMPT: str = (
    "You are a precise technical assistant for the vLLM codebase and "
    "documentation.\n"
    "Rules:\n"
    "1. Answer ONLY with facts stated in the sources given by the user. "
    "Never use outside knowledge and never invent names, flags, "
    "endpoints, functions or numbers.\n"
    "2. Answer the question directly in 1 to 3 short sentences. Quote "
    "exact identifiers (functions, classes, arguments, endpoints, "
    "commands) with backticks, copied exactly from the sources.\n"
    "3. Do not mention the sources, their numbers or file paths, and do "
    "not repeat the question.\n"
    f"4. If the sources do not contain the answer, reply exactly: "
    f"{NOT_FOUND}"
)

THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
SOURCE_MENTION = re.compile(
    r"\s*\((?:see |according to )?\[?sources? ?\d+(?:[,\s]+\d+)*\]?\)"
    r"|\s*\[sources? ?\d+(?:[,\s]+\d+)*\]", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?`)\]\"'](?=\s|$)")


def read_source(source: MinimalSource,
                cache: Dict[str, str]) -> str:
    """Return the text a source points to, reading its file once."""
    if source.file_path not in cache:
        try:
            cache[source.file_path] = Path(source.file_path).read_text(
                encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("Cannot read source %s", source.file_path)
            cache[source.file_path] = ""
    text = cache[source.file_path]
    return text[source.first_character_index:source.last_character_index]


def clean_answer(raw: str, truncated: bool) -> str:
    """Turn raw model output into a single clean answer string."""
    text = THINK_BLOCK.sub("", raw)
    text = SOURCE_MENTION.sub("", text).strip()
    # Drop a leading "Answer:" that small models like to add.
    text = re.sub(r"^(?:\*\*)?answer(?:\*\*)?\s*:\s*", "", text,
                  flags=re.IGNORECASE)
    if truncated:
        # Generation hit the token limit: cut at the last full sentence.
        ends = [match.end() for match in SENTENCE_END.finditer(text)]
        if ends:
            text = text[:ends[-1]]
    if text.count("```") % 2:
        text += "\n```"
    text = text.strip()
    if not text or NOT_FOUND.lower().rstrip(".") in text.lower():
        return NOT_FOUND
    return text


class AnswerGenerator:
    """Answer questions from retrieved sources with a small chat LLM."""

    def __init__(self, model_name: str = MODEL_NAME,
                 max_new_tokens: int = 192,
                 max_context_tokens: int = 3000) -> None:
        # transformers' type hints are incomplete: treat both as Any.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32)
        self.model.eval()
        self.max_new_tokens: int = max_new_tokens
        self.max_context_tokens: int = max_context_tokens

    def _build_context(self, texts: List[str]) -> str:
        """Number the sources, keeping the best ones within the budget."""
        parts: List[str] = []
        used: int = 0
        for number, text in enumerate(texts, start=1):
            ids = self.tokenizer.encode(text.strip(),
                                        add_special_tokens=False)
            remaining = self.max_context_tokens - used
            if remaining < 64:
                break
            if len(ids) > remaining:
                text = str(self.tokenizer.decode(ids[:remaining]))
                ids = ids[:remaining]
            used += len(ids)
            parts.append(f"[Source {number}]\n{text.strip()}")
        return "\n\n".join(parts)

    def generate(self, question: str, texts: List[str]) -> str:
        """Return an answer to `question` grounded in `texts`."""
        texts = [text for text in texts if text.strip()]
        if not texts:
            return NOT_FOUND
        user = (f"Sources:\n{self._build_context(texts)}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using only the sources above.")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.1,
                no_repeat_ngram_size=6,
                pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        truncated = (len(new_tokens) >= self.max_new_tokens
                     and new_tokens[-1] != self.tokenizer.eos_token_id)
        raw = str(self.tokenizer.decode(new_tokens,
                                        skip_special_tokens=True))
        return clean_answer(raw, truncated)


def answer_all(search: StudentSearchResults, generator: AnswerGenerator,
               context_sources: int) -> StudentSearchResultsAndAnswer:
    """Generate an answer for every question of `search`."""
    cache: Dict[str, str] = {}
    answers: List[MinimalAnswer] = []
    for result in tqdm(search.search_results, desc="Answering",
                       unit="question"):
        texts = [read_source(source, cache) for source
                 in result.retrieved_sources[:context_sources]]
        try:
            answer = generator.generate(result.question, texts)
        except Exception:
            logger.exception("Generation failed for %s", result.question_id)
            answer = NOT_FOUND
        answers.append(MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=answer))
    return StudentSearchResultsAndAnswer(search_results=answers,
                                         k=search.k)
