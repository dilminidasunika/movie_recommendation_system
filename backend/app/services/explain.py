from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.db import queries

log = logging.getLogger(__name__)

PROMPT = (settings.BACKEND_DIR / "app/prompts/explain.txt").read_text() \
    if hasattr(settings, "BACKEND_DIR") else None

TAG_FIELDS = ["genres", "director", "cast_members", "keywords", "writers",
              "producers", "music_composer"]
MAX_PLOT_CHARS = 400


class Item(BaseModel):
    movie_id: int
    explanation: str


class Explanations(BaseModel):
    items: list[Item]


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    from app.config import BACKEND_DIR
    return (BACKEND_DIR / "app" / "prompts" / "explain.txt").read_text()


def _norm(vals) -> set[str]:
    return {str(v).strip().lower() for v in (vals or []) if str(v).strip()}


def _query_tags(selected_rows: list[dict], chips: list[dict]) -> dict[str, set[str]]:
    out = {f: set() for f in TAG_FIELDS}
    for row in selected_rows:
        for f in TAG_FIELDS:
            out[f] |= _norm(row.get(f))
    for c in chips:
        out.setdefault(c["tag_type"], set()).add(c["name"].strip().lower())
    return out


def _overlap(movie: dict, qtags: dict[str, set[str]]) -> dict[str, list[str]]:
    res = {}
    for f, qset in qtags.items():
        hit = sorted(_norm(movie.get(f)) & qset)
        if hit:
            res[f] = hit[:6]
    return res


def _build_payload(ranked, movies, selected_rows, q) -> str:
    qtags = _query_tags(selected_rows, q.chips)
    payload = {
        "selected_movies": [
            {"title": r["title"], "genres": r.get("genres") or [],
             "plot": (r.get("overview") or "")[:MAX_PLOT_CHARS]}
            for r in selected_rows
        ],
        "tag_chips": q.chips,
        "recommended": [
            {
                "movie_id": r["movie_id"],
                "title": movies[r["movie_id"]]["title"],
                "tag_score": None if r["tag_score"] is None else round(r["tag_score"], 2),
                "plot_score": None if r["plot_score"] is None else round(r["plot_score"], 2),
                "overlapping_tags": _overlap(movies[r["movie_id"]], qtags),
                "plot": (movies[r["movie_id"]].get("overview") or "")[:MAX_PLOT_CHARS],
            }
            for r in ranked if r["movie_id"] in movies
        ],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


async def explain(ranked: list[dict], movies: dict[int, dict], q, req) -> dict[int, str]:
    """One Gemini call for all top-N movies. Returns {movie_id: explanation}."""
    selected_rows = []
    if q.movie_ids:
        got = await asyncio.to_thread(queries.get_movies, q.movie_ids)
        selected_rows = [got[i] for i in q.movie_ids if i in got]

    contents = _build_payload(ranked, movies, selected_rows, q)
    valid_ids = {r["movie_id"] for r in ranked if r["movie_id"] in movies}

    resp = await asyncio.wait_for(
        _client().aio.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_system_prompt(),
                response_mime_type="application/json",
                response_schema=Explanations,
                temperature=0.3,
                max_output_tokens=2048,
            ),
        ),
        timeout=settings.explain_timeout_s,
    )

    parsed: Explanations | None = resp.parsed
    if parsed is None:                       # schema parse failed
        parsed = Explanations.model_validate_json(resp.text)

    # Drop hallucinated ids and blanks
    return {i.movie_id: i.explanation.strip()
            for i in parsed.items
            if i.movie_id in valid_ids and i.explanation.strip()}