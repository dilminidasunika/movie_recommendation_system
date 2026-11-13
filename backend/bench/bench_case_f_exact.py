import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.pool import open_pool
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder, scoring, fusion
from app.stores import artifacts, meta_index

def main(n_distinct=20, rng_seed=42):
    print("=" * 65, flush=True)
    print("ITEM 3: CASE F EXACT SCORING BENCHMARK (295,100 ROWS)", flush=True)
    print("=" * 65, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    idx = meta_index.get_index()
    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()

    # Filter for Case f: genres=["Drama"]
    f = Filters(genres=("Drama",))
    t0 = time.perf_counter()
    mask = meta_index.mask_for(f)
    matching_ids = idx.ids[mask]
    gate_ms = (time.perf_counter() - t0) * 1000.0
    n_matches = len(matching_ids)
    print(f"Case f filter matches: {n_matches:,} movies (gate time: {gate_ms:.2f} ms)", flush=True)

    common_ids = np.array(list(set(id_to_row.keys()).intersection(set(plot_id_to_r.keys()))))
    rng = np.random.default_rng(rng_seed)

    # 1. Warm-up run
    seeds0 = rng.choice(common_ids, size=3, replace=False).tolist()
    req0 = RecommendRequest(movie_ids=seeds0, genres=["Drama"], tag_weight=0.5)
    q0 = query_builder.build(req0)
    pool0 = [int(mid) for mid in matching_ids if mid not in seeds0]

    t0 = time.perf_counter()
    s0 = scoring.score(pool0, q0)
    t_score_warm = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    r0 = fusion.fuse(s0, req0.tag_weight, top_n=200)
    t_fuse_warm = (time.perf_counter() - t0) * 1000.0
    total_warm = t_score_warm + t_fuse_warm
    print(f"\nWarm-up Exact Scoring over {len(pool0):,} rows:", flush=True)
    print(f"  Score: {t_score_warm:.2f} ms, Fuse: {t_fuse_warm:.2f} ms, Total: {total_warm:.2f} ms ({total_warm/1000:.3f} s)", flush=True)

    # 2. Distinct queries benchmark
    print(f"\nRunning {n_distinct} distinct requests with exact scoring over {n_matches:,} rows...", flush=True)
    score_times = []
    fuse_times = []
    total_times = []

    for trial in range(n_distinct):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, genres=["Drama"], tag_weight=0.5)
        q = query_builder.build(req)
        pool = [int(mid) for mid in matching_ids if mid not in seeds]

        t0 = time.perf_counter()
        s = scoring.score(pool, q)
        t_sc = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        r = fusion.fuse(s, req.tag_weight, top_n=200)
        t_fu = (time.perf_counter() - t0) * 1000.0

        score_times.append(t_sc)
        fuse_times.append(t_fu)
        total_times.append(t_sc + t_fu)

    print("\nCase f Exact Scoring Latency Results (N=20 distinct requests):", flush=True)
    print(f"  Scoring Time (score_tag + score_plot):")
    print(f"    Mean:   {np.mean(score_times):.2f} ms ({np.mean(score_times)/1000:.3f} s)")
    print(f"    Median: {np.median(score_times):.2f} ms ({np.median(score_times)/1000:.3f} s)")
    print(f"    p95:    {np.percentile(score_times, 95):.2f} ms ({np.percentile(score_times, 95)/1000:.3f} s)")
    print(f"  Fusion Time:")
    print(f"    Mean:   {np.mean(fuse_times):.2f} ms")
    print(f"  Total Retrieval & Rank (excluding detail fetch):")
    print(f"    Mean:   {np.mean(total_times):.2f} ms ({np.mean(total_times)/1000:.3f} s)")
    print(f"    Median: {np.median(total_times):.2f} ms ({np.median(total_times)/1000:.3f} s)")
    print(f"    p95:    {np.percentile(total_times, 95):.2f} ms ({np.percentile(total_times, 95)/1000:.3f} s)")
    print(f"    Min:    {np.min(total_times):.2f} ms")
    print(f"    Max:    {np.max(total_times):.2f} ms")
    print(f"\nRecall@200 against exact ground truth: 100.00% (By definition, scores all {n_matches:,} rows)")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main(n_distinct=20, rng_seed=42)
