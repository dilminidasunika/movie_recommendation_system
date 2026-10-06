from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.db.pool import get_pool

POST_FILTER_BATCH = 5000


@dataclass(frozen=True)
class Filters:
    languages: tuple[str, ...] = field(default_factory=tuple)
    genres: tuple[str, ...] = field(default_factory=tuple)
    date_from: date | None = None
    date_to: date | None = None

    @property
    def active(self) -> bool:
        return bool(self.languages or self.genres or self.date_from or self.date_to)

    @classmethod
    def from_request(cls, req) -> "Filters":
        # request validation already lowercased these
        return cls(
            languages=tuple(req.original_language or ()),
            genres=tuple(req.genres or ()),
            date_from=req.date_from,
            date_to=req.date_to,
        )


def _where(f: Filters) -> tuple[list[str], list]:
    """THE single definition of the hard filters. Used by the gate and the post-filter."""
    clauses, params = [], []
    if f.languages:
        clauses.append("original_language = ANY(%s)")
        params.append(list(f.languages))
    if f.genres:
        clauses.append("genres && %s::text[]")        # any-of, served by the GIN index
        params.append(list(f.genres))
    if f.date_from:
        clauses.append("release_date >= %s")          # null release_date fails, by design
        params.append(f.date_from)
    if f.date_to:
        clauses.append("release_date <= %s")
        params.append(f.date_to)
    return clauses, params


def gate_ids(f: Filters, limit: int) -> list[int]:
    """Filtered ids, unordered. Caller passes EXACT_LIMIT + 1."""
    clauses, params = _where(f)
    sql = "SELECT id FROM movies"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " LIMIT %s"
    with get_pool().connection() as conn:
        rows = conn.execute(sql, [*params, limit]).fetchall()
    return [int(r[0]) for r in rows]


def filter_ids(ids: list[int], f: Filters) -> set[int]:
    """Subset of ids that satisfy the filters (same _where as the gate)."""
    if not ids:
        return set()
    clauses, params = _where(f)
    where = " AND ".join(["id = ANY(%s)", *clauses])
    keep: set[int] = set()
    with get_pool().connection() as conn:
        for i in range(0, len(ids), POST_FILTER_BATCH):
            chunk = ids[i:i + POST_FILTER_BATCH]
            rows = conn.execute(f"SELECT id FROM movies WHERE {where}",
                                [chunk, *params]).fetchall()
            keep.update(int(r[0]) for r in rows)
    return keep