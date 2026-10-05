import asyncio
from fastapi import APIRouter
from app.schemas import RecommendRequest
from app.services.recommender import retrieve_and_rank
# from app.services.explain import explain_all   # add later

router = APIRouter()

@router.post("/recommend")
async def recommend(req: RecommendRequest):
    result = await asyncio.to_thread(retrieve_and_rank, req)  # concurrent requests overlap (step 54)
    # explanations = await explain_all(req, result)           # one async LLM call for all top N
    return {"results": result["ranked"]}