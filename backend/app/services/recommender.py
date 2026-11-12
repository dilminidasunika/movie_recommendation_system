import time
from app.config import settings
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion


def retrieve_and_rank(req: RecommendRequest) -> dict:
    """Sync, CPU/IO-bound. Called via to_thread from the endpoint."""
    t_start = time.perf_counter()
    timings, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(now - t, 4)
        t = now

    q = query_builder.build(req, timings=timings)                 # may raise EmptyQueryError
    lap("query_build")

    pool_ids = candidates.generate(req, q, timings=timings)       # already excludes selected movies
    lap("candidates")

    warnings = []
    if q.missing_tag_ids:
        warnings.append(f"No tag data for movies {q.missing_tag_ids}; ignored for tag similarity.")
    if q.missing_plot_ids:
        warnings.append(f"No plot data for movies {q.missing_plot_ids}; ignored for plot similarity.")

    if not pool_ids:
        timings["total"] = round(time.perf_counter() - t_start, 4)
        return {"ranked": [], "pool_size": 0, "warnings": warnings,
                "timings": timings, "query": q}

    s = scoring.score(pool_ids, q, timings=timings)
    lap("scoring")

    ranked = fusion.fuse(s, req.tag_weight, top_n=settings.max_results)
    lap("fusion")

    timings["total"] = round(time.perf_counter() - t_start, 4)

    # Release large numpy arrays that were only needed for scoring.
    # The cached Query (via paging) only needs q.chips for explanations;
    # tag_full_sims is float32[1M] ≈ 4.3 MB and would leak ~850 MB in
    # a TTLCache(maxsize=200).
    q.tag_full_sims = None
    q.tag_sparse = None
    q.tag_ann = None
    q.plot_centroid = None

    return {"ranked": ranked, "pool_size": len(pool_ids), "warnings": warnings,
            "timings": timings, "query": q}
