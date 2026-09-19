from fastapi import Request
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings


def open_pool() -> ConnectionPool:
    pool = ConnectionPool(
        settings.database_url,
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True)
    return pool


def get_pool(request: Request) -> ConnectionPool:
    """FastAPI dependency: the pool created in lifespan."""
    return request.app.state.pool