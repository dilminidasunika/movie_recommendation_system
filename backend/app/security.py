import asyncio
from contextlib import asynccontextmanager

from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

PIPELINE_SEM = asyncio.Semaphore(4)   # concurrent retrieve_and_rank runs
LLM_SEM = asyncio.Semaphore(2)        # concurrent explanation calls


@asynccontextmanager
async def slot(sem: asyncio.Semaphore, wait_s: float = 5.0):
    try:
        await asyncio.wait_for(sem.acquire(), wait_s)
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy, try again shortly.",
                            headers={"Retry-After": "5"})
    try:
        yield
    finally:
        sem.release()
