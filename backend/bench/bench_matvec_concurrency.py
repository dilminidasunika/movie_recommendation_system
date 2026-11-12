import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True

from app.db.pool import open_pool
from app.stores import artifacts
from app.schemas import RecommendRequest
from app.services import query_builder, candidates

def run_bench():
    print("=" * 80)
    print("ITEM 6: TAG MATVEC LATENCY UNDER CONCURRENCY")
    print("Config: PIPELINE_SEM = 4, BLAS threads = 1, 8 simultaneous distinct requests")
    print("=" * 80)

    open_pool()
    artifacts.load_all()

    tag_ids = artifacts.matrix_ids()
    rng = np.random.default_rng(42)

    # Pre-generate 24 distinct queries (3 iterations of 8)
    all_queries = []
    for _ in range(24):
        seeds = rng.choice(tag_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        q = query_builder.build(req)
        all_queries.append((seeds, q))

    sem = threading.Semaphore(4)

    def worker(item):
        seeds, q = item
        exclude = set(seeds)
        with sem:
            t0 = time.perf_counter()
            top_ids, dt_internal = candidates._matvec_tag(q, 2000, exclude)
            t_total = time.perf_counter() - t0
        return t_total * 1000.0, dt_internal * 1000.0, len(top_ids)

    # Warm-up run with 4 queries
    print("Warming up matvec...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        _ = list(ex.map(worker, all_queries[:4]))
    print("Warm-up complete.", flush=True)

    # Benchmark: 3 iterations of 8 simultaneous distinct requests
    batch_latencies = []
    all_matvec_times = []

    for it in range(3):
        batch = all_queries[it*8 : (it+1)*8]
        t_batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(worker, batch))
        t_batch_total = (time.perf_counter() - t_batch_start) * 1000.0

        batch_times = [r[0] for r in results]
        batch_latencies.append(t_batch_total)
        all_matvec_times.extend(batch_times)

        print(f"Iteration {it+1}: Batch wall time = {t_batch_total:.1f}ms, Individual times = {[f'{t:.1f}ms' for t in sorted(batch_times)]}", flush=True)

    latencies_arr = np.array(all_matvec_times)
    p50 = np.percentile(latencies_arr, 50)
    p95 = np.percentile(latencies_arr, 95)
    mean = np.mean(latencies_arr)
    min_lat = np.min(latencies_arr)
    max_lat = np.max(latencies_arr)

    avg_batch_time_s = np.mean(batch_latencies) / 1000.0
    throughput = 8.0 / avg_batch_time_s

    print("\n--- Summary Results (24 distinct requests across 3 batches of 8) ---")
    print(f"Per-request MatVec Latency (p50):        {p50:.1f} ms")
    print(f"Per-request MatVec Latency (p95):        {p95:.1f} ms")
    print(f"Per-request MatVec Latency (mean):       {mean:.1f} ms")
    print(f"Per-request MatVec Latency (min/max):    {min_lat:.1f} ms / {max_lat:.1f} ms")
    print(f"Batch Wall Time (8 simultaneous, mean):  {np.mean(batch_latencies):.1f} ms")
    print(f"Throughput under PIPELINE_SEM=4:         {throughput:.1f} req/s")

if __name__ == "__main__":
    run_bench()
