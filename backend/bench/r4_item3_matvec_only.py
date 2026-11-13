"""Matvec-only contention test (supplement to Item 3)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True

from app.stores import artifacts
from app.services import query_builder
from app.schemas import RecommendRequest
import numpy as np

def run():
    print("=" * 70, flush=True)
    print("MATVEC CONTENTION TEST (1, 2, 4 concurrent callers)", flush=True)
    print("=" * 70, flush=True)

    artifacts.load_all()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()

    rng = np.random.default_rng(42)
    tag_ids = artifacts.matrix_ids()
    common_ids = np.array(sorted(set(int(i) for i in tag_ids)))

    # Build 8 queries
    queries = []
    for i in range(8):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        q = query_builder.build(req)
        queries.append(q)

    def do_matvec(q):
        t0 = time.perf_counter()
        dots = (tag_m @ q.tag_sparse.T).toarray().ravel()
        valid = tag_norms > 0
        sims = np.zeros(len(tag_norms), dtype=np.float32)
        sims[valid] = dots[valid] / tag_norms[valid]
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed

    # Warmup
    for q in queries[:3]:
        do_matvec(q)

    for n_workers in [1, 2, 4]:
        lats = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(do_matvec, queries[i % len(queries)]) for i in range(8)]
            for f in as_completed(futures):
                lats.append(f.result())
        lats = np.array(lats)
        print(f"  {n_workers} concurrent: p50={np.percentile(lats,50):.1f}ms p95={np.percentile(lats,95):.1f}ms mean={np.mean(lats):.1f}ms", flush=True)

if __name__ == "__main__":
    run()
