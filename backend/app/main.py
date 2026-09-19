from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool

from app.api import tags  # , movies, recommend
from app.config import settings
from app.db.pool import get_pool, open_pool

# Uncomment once these exist:
# from app.stores.artifacts import load_artifacts
# from app.stores.chroma import open_chroma


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything once at startup, never per request.
    app.state.pool = open_pool()
    # app.state.artifacts = load_artifacts(settings.artifacts_dir, settings.artifact_version)
    # app.state.chroma = open_chroma(settings.chroma_dir)
    yield
    app.state.pool.close()


app = FastAPI(title="Movie Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(tags.router, prefix="/api")
# app.include_router(movies.router, prefix="/api")
# app.include_router(recommend.router, prefix="/api")


@app.get("/health")
def health(pool: ConnectionPool = Depends(get_pool)):
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}