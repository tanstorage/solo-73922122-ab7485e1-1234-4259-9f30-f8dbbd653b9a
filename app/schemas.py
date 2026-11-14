"""Request/response schemas for the corpus API."""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .db import MAX_ARTICLES, MAX_TEXT_LEN, MIN_TEXT_LEN

ArticleId = Annotated[str, Field(min_length=1, max_length=128)]
ArticleText = Annotated[str, Field(min_length=MIN_TEXT_LEN, max_length=MAX_TEXT_LEN)]


class ArticleUpsert(BaseModel):
    id: ArticleId
    side: Literal["train", "test"]
    text: ArticleText


class BatchRequest(BaseModel):
    """One atomic corpus modification: additions/corrections plus deletions.

    The whole batch is validated first and applied in a single transaction;
    if anything fails, nothing takes effect.
    """

    upserts: list[ArticleUpsert] = Field(default_factory=list, max_length=MAX_ARTICLES)
    deletes: list[ArticleId] = Field(default_factory=list, max_length=MAX_ARTICLES)


class BatchResponse(BaseModel):
    changed: bool
    generation: int
    published_generation: int
    upserted: int
    deleted: int
    articles: int
