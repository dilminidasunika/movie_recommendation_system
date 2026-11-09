import asyncio
import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query, Response

from app.db import queries
from app.schemas import RecommendRequest, RecommendResponse
from app.services import explain, paging, recommender
from app.services.query_builder import EmptyQueryError

log = logging.getLogger(__name__)
router = APIRouter()


def _require(rid: str) -> dict:
    entry = paging.get(rid)
    if entry is None:
        raise HTTPException(
            410,
            "Recommendations expired. Please search again.",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return entry


async def _page(rid: str, entry: dict, offset: int, limit: int) -> dict:
    ranked = entry["ranked"]
    total = len(ranked)
    page = ranked[offset: offset + limit]
    base = {"warnings": entry["warnings"], "pool_size": entry["pool_size"],
            "timings": entry["timings"], "request_id": rid, "total": total}
    if not page:
        return {**base, "results": [], "next_offset": None}

    ids = [r["movie_id"] for r in page]
    movies = await asyncio.to_thread(queries.get_movies, ids)
    missing = [i for i in ids if i not in movies]
    if missing:
        log.error("ids in ranking but not in Postgres: %s", missing)
        page = [r for r in page if r["movie_id"] in movies]

    expl = entry["explanations"]          # include only if already generated; NEVER call Gemini here
    results = [{**r, "movie": movies[r["movie_id"]],
                "explanation": expl.get(r["movie_id"])} for r in page]
    nxt = offset + limit
    return {**base, "results": results, "next_offset": nxt if nxt < total else None}


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, response: Response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    rid = paging.request_id(req)
    entry = paging.get(rid)
    if entry is None:
        try:
            out = await asyncio.to_thread(recommender.retrieve_and_rank, req)
        except EmptyQueryError as e:
            raise HTTPException(422, str(e))
        entry = {"req": req, "ranked": out["ranked"], "query": out["query"],
                 "warnings": out["warnings"], "pool_size": out["pool_size"],
                 "timings": out["timings"], "explanations": {},
                 "locks": defaultdict(asyncio.Lock)}
        paging.put(rid, entry)
    return await _page(rid, entry, 0, req.top_n)


@router.get("/recommend/{request_id}", response_model=RecommendResponse)
async def recommend_page(request_id: str,
                         response: Response,
                         offset: int = Query(0, ge=0),
                         limit: int = Query(20, ge=1, le=50)):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return await _page(request_id, _require(request_id), offset, limit)


@router.get("/recommend/{request_id}/explanations")
async def recommend_explanations(request_id: str,
                                 response: Response,
                                 offset: int = Query(0, ge=0),
                                 limit: int = Query(20, ge=1, le=50)):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    entry = _require(request_id)
    page = entry["ranked"][offset: offset + limit]
    ids = [r["movie_id"] for r in page]
    expl = entry["explanations"]

    async with entry["locks"][offset]:      # concurrent calls for the same page share one Gemini call
        todo = [r for r in page if r["movie_id"] not in expl]
        if todo:
            movies = await asyncio.to_thread(queries.get_movies, [r["movie_id"] for r in todo])
            todo = [r for r in todo if r["movie_id"] in movies]
            try:
                expl.update(await explain.explain(todo, movies, entry["query"], entry["req"]))
            except Exception:
                log.exception("explain failed")   # failures are NOT cached, so a revisit retries

    return {"explanations": {i: expl[i] for i in ids if i in expl}}
