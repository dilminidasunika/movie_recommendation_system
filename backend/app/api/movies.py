from fastapi import APIRouter, Depends, Query
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.db.pool import get_pool

router = APIRouter(prefix="/movies", tags=["movies"])


class MovieHit(BaseModel):
    id: int
    title: str
    year: int | None = None


MOVIE_SEARCH_SQL = """
SELECT id, title, EXTRACT(YEAR FROM release_date)::int AS year
FROM movies
WHERE lower(title) LIKE %(contains)s
ORDER BY (lower(title) = %(q)s) DESC,                 -- exact title match
         (lower(title) LIKE %(prefix)s) DESC,         -- starts with query
         (lower(title) LIKE %(word_prefix)s) DESC,    -- a word starts with query
         length(title),                               -- shorter = closer match
         title,
         id                                           -- stable tie-break
LIMIT %(limit)s
"""


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.get("/search", response_model=list[MovieHit])
def search_movies(
    q: str = Query(..., max_length=100),
    limit: int = Query(10, ge=1, le=25),
    pool: ConnectionPool = Depends(get_pool),
):
    q = q.strip().lower()
    if len(q) < 3:
        return []
    safe = _escape_like(q)
    params = {
        "q": q,
        "prefix": safe + "%",
        "word_prefix": "% " + safe + "%",   # matches "the godfather" for q="god"
        "contains": "%" + safe + "%",
        "limit": limit,
    }
    with pool.connection() as conn:
        return conn.execute(MOVIE_SEARCH_SQL, params).fetchall()