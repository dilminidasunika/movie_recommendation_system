from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import settings
from app.db.pool import get_pool, open_pool, close_pool
from app.security import limiter
from app.stores import artifacts, chroma, text_embedder
from app.api import movies, tags, recommend

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = open_pool()
    artifacts.load_all()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()
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
