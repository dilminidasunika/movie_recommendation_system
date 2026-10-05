from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

TagType = Literal[
    "cast_members", "director", "writers", "producers", "music_composer",
    "keywords", "production_countries", "spoken_languages",
]

class TagIn(BaseModel):
    tag_type: TagType
    name: str

class RecommendRequest(BaseModel):
    movie_ids: list[int] = Field(default_factory=list)
    tags: list[TagIn] = Field(default_factory=list)
    original_language: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    tag_weight: float = Field(0.5, ge=0, le=1)
    top_n: int = Field(20, ge=1, le=100)

    @field_validator("original_language", "genres")
    @classmethod
    def lower(cls, v): return [x.strip().lower() for x in v]

    @model_validator(mode="after")
    def check(self):
        if not self.movie_ids and not self.tags:
            raise ValueError("Provide at least one of movie_ids or tags")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        if not self.movie_ids:
            self.tag_weight = 1.0  # no plot centroid exists
        return self

    @property
    def has_filters(self) -> bool:
        return bool(self.original_language or self.genres or self.date_from or self.date_to)