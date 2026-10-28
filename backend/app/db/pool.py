from fastapi import Request
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None


def open_pool() -> ConnectionPool:
    global _pool
    _pool = ConnectionPool(
        settings.database_url,
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    _pool.open(wait=True)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool(request: Request) -> ConnectionPool:
    """FastAPI dependency: the pool created in lifespan."""
    return request.app.state.pool


def current_pool() -> ConnectionPool:                                   # NEW
    """Request-free accessor for services running in worker threads."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised (open_pool() not called)")
    return _pool