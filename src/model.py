from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnsweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnsweredQuestion):
    sources: List[MinimalSource]
    answer: str
