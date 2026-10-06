from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool

from app.config import settings
from app.db.pool import get_pool, open_pool, close_pool
from app.stores import artifacts, chroma
from app.api import movies, tags, recommend

@asynccontextmanager
async def lifespan(app: FastAPI):
    open_pool()
    artifacts.load_all()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    yield
    close_pool()


app = FastAPI(title="Movie Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(tags.router, prefix="/api")
app.include_router(movies.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")


@app.get("/health")
def health(pool: ConnectionPool = Depends(get_pool)):
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}