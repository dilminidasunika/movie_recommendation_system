import time
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion


def retrieve_and_rank(req: RecommendRequest) -> dict:
    """Sync, CPU/IO-bound. Called via to_thread from the endpoint."""
    timings, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(now - t, 4)
        t = now

    q = query_builder.build(req)                 # may raise EmptyQueryError
    lap("query_build")

    pool_ids = candidates.generate(req, q)       # already excludes selected movies
    lap("candidates")

    warnings = []
    if q.missing_tag_ids:
        warnings.append(f"No tag data for movies {q.missing_tag_ids}; ignored for tag similarity.")
    if q.missing_plot_ids:
        warnings.append(f"No plot data for movies {q.missing_plot_ids}; ignored for plot similarity.")

    if not pool_ids:
        return {"ranked": [], "pool_size": 0, "warnings": warnings,
                "timings": timings, "query": q}

    s = scoring.score(pool_ids, q)
    lap("scoring")

    ranked = fusion.fuse(s, req.tag_weight, top_n=req.top_n)
    lap("fusion")

    return {"ranked": ranked, "pool_size": len(pool_ids), "warnings": warnings,
            "timings": timings, "query": q}