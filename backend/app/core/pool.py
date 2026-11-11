from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rec_worker")


def get_pool() -> ThreadPoolExecutor:
    """Return the shared module-level ThreadPoolExecutor."""
    return POOL
