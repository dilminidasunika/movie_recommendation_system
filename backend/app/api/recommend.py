import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.db import queries
from app.schemas import RecommendRequest, RecommendResponse
from app.services import explain, recommender
from app.services.query_builder import EmptyQueryError

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        out = await asyncio.to_thread(recommender.retrieve_and_rank, req)
    except EmptyQueryError as e:
        raise HTTPException(422, str(e))

    ranked = out["ranked"]
    if not ranked:
        return {"results": [], "warnings": out["warnings"], "pool_size": 0,
                "timings": out["timings"]}

    ids = [r["movie_id"] for r in ranked]
    movies = await asyncio.to_thread(queries.get_movies, ids)    # dict[int, row]

    # Ids missing in Postgres mean the DB and the artifacts/Chroma are out of sync
    missing = [i for i in ids if i not in movies]
    if missing:
        log.error("ids in ranking but not in Postgres: %s", missing)
        ranked = [r for r in ranked if r["movie_id"] in movies]

    try:
        explanations = await explain.explain(ranked, movies, out["query"], req)
    except Exception:
        log.exception("explain failed")      # never swallow silently
        explanations = {}

    results = [
        {**r, "movie": movies[r["movie_id"]],
         "explanation": explanations.get(r["movie_id"])}
        for r in ranked
    ]
    return {"results": results, "warnings": out["warnings"],
            "pool_size": out["pool_size"], "timings": out["timings"]}