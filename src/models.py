from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """Location of a text excerpt inside a file of the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question asked by the user, without sources or answer."""

    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question together with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Sources retrieved by the search step for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for a question, plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset, with the k used for retrieval."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a whole dataset."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """A piece of a corpus file, with its location in that file."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
