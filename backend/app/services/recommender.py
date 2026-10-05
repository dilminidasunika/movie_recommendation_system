from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion

def retrieve_and_rank(req: RecommendRequest) -> dict:
    """Sync and CPU/IO-bound. Called via to_thread from the endpoint."""
    q = query_builder.build(req)              # tag_vec (sparse), plot_centroid | None
    pool_ids = candidates.generate(req, q)    # list[int]; gate + small/large branch
    tag_s, plot_s, ids = scoring.score(pool_ids, q)  # aligned arrays, no NaNs
    ranked = fusion.fuse(ids, tag_s, plot_s, req.tag_weight,
                         exclude=set(req.movie_ids), top_n=req.top_n)
    return {"ranked": ranked, "query": q}