from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.db.pool import get_pool

router = APIRouter(prefix="/tags", tags=["tags"])

VALID_TYPES = {
    "genre", "original_language", "cast", "director",
    "writer", "producer", "composer", "keyword",
    "company", "country", "spoken_language",
}
# Small enumerations: fetched once and filtered client-side, no type-ahead needed.
STATIC_TYPES = {"genre", "original_language", "spoken_language", "country"}


class TagOut(BaseModel):  # move to schemas.py if you prefer
    tag_type: str
    name: str
    movie_count: int  # display only, never used for ranking


def _parse_types(types: str) -> list[str]:
    parsed = [t.strip() for t in types.split(",") if t.strip()]
    bad = [t for t in parsed if t not in VALID_TYPES]
    if not parsed or bad:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tag type(s): {bad or 'none given'}. Allowed: {sorted(VALID_TYPES)}",
        )
    return parsed


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Note: psycopg needs a literal % doubled, hence %% for the trigram operator.
SEARCH_SQL = """
WITH m AS (
    SELECT tag_type, name, movie_count,
           (lower(name) LIKE lower(%(prefix)s)) AS is_prefix,
           similarity(lower(name), lower(%(q)s)) AS sim
    FROM tag_lookup
    WHERE tag_type = ANY(%(types)s::text[])
      AND (lower(name) LIKE lower(%(contains)s) OR lower(name) %% lower(%(q)s))
), ranked AS (
    SELECT tag_type, name, movie_count,
           ROW_NUMBER() OVER (
               PARTITION BY tag_type
               ORDER BY is_prefix DESC, sim DESC, name
           ) AS rn
    FROM m
)
SELECT tag_type, name, movie_count
FROM ranked
WHERE rn <= %(per_type)s
ORDER BY tag_type, rn
"""


@router.get("/search", response_model=list[TagOut])
def search_tags(
    q: str = Query(..., max_length=100),
    types: str = Query(..., description="comma-separated, e.g. 'cast' or 'writer,producer,keyword'"),
    per_type: int = Query(4, ge=1, le=20),
    pool: ConnectionPool = Depends(get_pool),
):
    type_list = _parse_types(types)
    q = q.strip()
    if len(q) < 3:  # trigram index is ineffective below 3 chars
        return []

    safe = _escape_like(q)
    params = {
        "q": q,
        "prefix": safe + "%",
        "contains": "%" + safe + "%",
        "types": type_list,
        "per_type": per_type,
    }
    with pool.connection() as conn:
        return conn.execute(SEARCH_SQL, params).fetchall()


_static_cache: dict[str, list[dict]] = {}


@router.get("/all", response_model=list[TagOut])
def all_tags(
    tag_type: str = Query(...),
    pool: ConnectionPool = Depends(get_pool),
):
    if tag_type not in STATIC_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"'{tag_type}' is too large to list. Static types: {sorted(STATIC_TYPES)}",
        )
    if tag_type not in _static_cache:
        with pool.connection() as conn:
            _static_cache[tag_type] = conn.execute(
                "SELECT tag_type, name, movie_count FROM tag_lookup "
                "WHERE tag_type = %s ORDER BY name",
                (tag_type,),
            ).fetchall()
    return _static_cache[tag_type]