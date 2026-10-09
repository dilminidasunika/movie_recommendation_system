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
STATIC_TYPES = {"genre", "original_language"}


class TagOut(BaseModel):
    tag_type: str
    name: str
    movie_count: int  # display only


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


SEARCH_SQL_FAST = """
SELECT t.tag_type, t.name, t.movie_count
FROM unnest(%(types)s::text[]) AS ty(tag_type)
CROSS JOIN LATERAL (
    SELECT tag_type, name, movie_count
    FROM tag_lookup
    WHERE tag_type = ty.tag_type
      AND lower(name) LIKE %(contains)s
    ORDER BY (lower(name) LIKE %(prefix)s) DESC, length(name), name
    LIMIT %(per_type)s
) t
"""

SEARCH_SQL_FUZZY = """
SELECT t.tag_type, t.name, t.movie_count
FROM unnest(%(types)s::text[]) AS ty(tag_type)
CROSS JOIN LATERAL (
    SELECT tag_type, name, movie_count
    FROM tag_lookup
    WHERE tag_type = ty.tag_type
      AND (lower(name) LIKE %(contains)s OR lower(name) %% %(q)s::text)
    ORDER BY (lower(name) LIKE %(prefix)s) DESC,
             similarity(lower(name), %(q)s::text) DESC,
             name
    LIMIT %(per_type)s
) t
"""


@router.get("/search", response_model=list[TagOut])
def search_tags(
    q: str = Query(..., max_length=100),
    types: str = Query(..., description="comma-separated tag types"),
    per_type: int = Query(4, ge=1, le=20),
    pool: ConnectionPool = Depends(get_pool),
):
    type_list = _parse_types(types)
    q = q.strip().lower()
    if len(q) < 3 and "original_language" not in type_list:
        return []
    safe = _escape_like(q)
    params = {
        "q": q,
        "prefix": safe + "%",
        "contains": "%" + safe + "%",
        "types": type_list,
        "per_type": per_type,
    }
    sql = SEARCH_SQL_FUZZY if len(q) >= 5 else SEARCH_SQL_FAST
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchall()


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