from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

TagType = Literal[
    "cast_members", "director", "writers", "producers", "music_composer",
    "keywords", "production_countries", "spoken_languages","genres"
]

class TagIn(BaseModel):
    tag_type: TagType
    name: str

class RecommendRequest(BaseModel):
    movie_ids: list[int] = Field(default_factory=list)
    tags: list[TagIn] = Field(default_factory=list)
    plot_text: str | None = None
    original_language: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    tag_weight: float = Field(0.5, ge=0, le=1)
    top_n: int = Field(20, ge=1, le=100)

    @field_validator("original_language")
    @classmethod
    def lower(cls, v): return [x.strip().lower() for x in v]

    @field_validator("plot_text", mode="before")
    @classmethod
    def validate_plot_text(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("plot_text must be a string")
        collapsed = " ".join(v.split())
        if not collapsed:
            return None
        if len(collapsed) < 30:
            raise ValueError("Plot text must be at least 30 characters")
        if len(collapsed) > 1000:
            raise ValueError("Plot text must not exceed 1000 characters")
        return collapsed

    @model_validator(mode="after")
    def check(self):
        if not self.movie_ids and not self.tags and not self.plot_text:
            raise ValueError("Provide at least one of movie_ids, tags, or plot_text")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        if not self.movie_ids and not self.plot_text:
            self.tag_weight = 1.0  # no plot query exists
        return self

    @property
    def has_filters(self) -> bool:
        return bool(self.original_language or self.genres or self.date_from or self.date_to)


class MovieOut(BaseModel):
    id: int
    title: str
    release_date: date | None = None
    original_language: str | None = None
    genres: list[str] = []
    director: list[str] = []
    cast_members: list[str] = []
    overview: str | None = None
    poster_path: str | None = None

class ResultOut(BaseModel):
    movie_id: int
    final_score: float
    tag_score: float | None      # None = candidate had no tags
    plot_score: float | None     # None = no plot signal, or tags-only query
    movie: MovieOut
    explanation: str | None      # None if the LLM call failed

class RecommendResponse(BaseModel):
    results: list[ResultOut]
    warnings: list[str] = []
    pool_size: int
    timings: dict[str, float] | None = None
