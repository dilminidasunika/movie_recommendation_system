import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from psycopg_pool import ConnectionPool
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import settings
from app.db.pool import get_pool, open_pool, close_pool
from app.schemas import RecommendRequest, TagIn
from app.security import limiter
from app.services import recommender
from app.stores import artifacts, chroma, text_embedder, meta_index
from app.api import movies, tags, recommend

log = logging.getLogger(__name__)


def _warmup() -> None:
    """Warm up recommendation pipeline on the REAL path with fixed synthetic requests."""
    rng = np.random.default_rng(42)
    all_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()

    # Check that seeds 238, 240, 278 exist in DB/tag matrix; otherwise use real ids from tag matrix
    fixed_seeds = [238, 240, 278]
    if not all(s in id_to_row for s in fixed_seeds):
        fixed_seeds = all_ids[:3].tolist()
        log.warning("Default warmup seeds missing from tag matrix, using fallback: %s", fixed_seeds)

    warmup_requests = [
        RecommendRequest(movie_ids=fixed_seeds, tag_weight=0.5),
        RecommendRequest(movie_ids=rng.choice(all_ids, size=3, replace=False).tolist(), tag_weight=0.5),
        RecommendRequest(movie_ids=rng.choice(all_ids, size=3, replace=False).tolist(), tag_weight=0.5),
        RecommendRequest(movie_ids=fixed_seeds, plot_text="A dramatic story of crime, family loyalty, and redemption.", tag_weight=0.5),
        RecommendRequest(tags=[TagIn(tag_type="genres", name="Drama")], tag_weight=1.0),
    ]

    successes = 0
    for req in warmup_requests:
        try:
            recommender.retrieve_and_rank(req)
            successes += 1
        except Exception:
            log.exception("Warmup request failed for request: %s", req)

    log.info("Warmup complete: %d / %d requests succeeded", successes, len(warmup_requests))

    if settings.app_env == "prod" and successes == 0:
        raise RuntimeError("No warmup requests succeeded at boot in production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    if not settings.use_matvec_tag_candidates:
        chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()
    _warmup()
    yield
    close_pool()


prod = settings.app_env == "prod"
app = FastAPI(title="Movie Recommender", lifespan=lifespan,
              docs_url=None if prod else "/docs",
              redoc_url=None,
              openapi_url=None if prod else "/openapi.json")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,      # explicit list, never "*"
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)

if prod:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

app.include_router(tags.router, prefix="/api")
app.include_router(movies.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")


@app.get("/health")
def health(pool: ConnectionPool = Depends(get_pool)):
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}
