"""
Benchmark for Item 1:
1. Matvec p50/p95 at 1, 2, 4 concurrent callers BEFORE vs AFTER
2. score_tag for 55k and 295k exact cases BEFORE vs AFTER
"""
import sys, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder

def old_matvec(tag_sparse):
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    dots = (tag_m @ tag_sparse.T).toarray().ravel()
    valid = tag_norms > 0
    sims = np.zeros(len(tag_norms), dtype=np.float32)
    sims[valid] = dots[valid] / tag_norms[valid]
    return sims

def new_matvec(tag_sparse):
    tag_m = artifacts.tag_matrix()
    inv = artifacts.inv_norms()
    V = tag_m.shape[1]
    qd = np.zeros(V, dtype=np.float64)
    qd[tag_sparse.indices] = tag_sparse.data
    dots = tag_m @ qd
    sims = (dots * inv).astype(np.float32)
    return sims

def old_score_tag(pool_ids, q):
    i2r = artifacts.id_to_row()
    pos = [k for k, i in enumerate(pool_ids) if i in i2r]
    rows = [i2r[pool_ids[k]] for k in pos]
    t0 = time.perf_counter()
    norms = artifacts.row_norms()[rows]
    dots = (artifacts.tag_matrix()[rows] @ q.tag_sparse.T).toarray().ravel()
    valid = norms > 0
    sims = np.zeros(len(rows))
    sims[valid] = dots[valid] / norms[valid]
    dt = time.perf_counter() - t0
    return dt

def new_score_tag(pool_ids, q):
    i2r = artifacts.id_to_row()
    pos = [k for k, i in enumerate(pool_ids) if i in i2r]
    rows = [i2r[pool_ids[k]] for k in pos]
    t0 = time.perf_counter()
    tag_m = artifacts.tag_matrix()
    inv = artifacts.inv_norms()
    V = tag_m.shape[1]
    qd = np.zeros(V, dtype=np.float64)
    qd[q.tag_sparse.indices] = q.tag_sparse.data
    full_sims = ((tag_m @ qd) * inv).astype(np.float32)
    sims = full_sims[rows]
    dt = time.perf_counter() - t0
    return dt

def main():
    print("Opening DB pool and loading artifacts and meta_index...", flush=True)
    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    idx = meta_index.get_index()

    # Create 30 realistic queries
    rng = np.random.default_rng(123)
    all_ids = idx.ids[idx.ids > 0]
    sample_seeds = [rng.choice(all_ids, size=3, replace=False).tolist() for _ in range(50)]
    queries = []
    for s in sample_seeds:
        req = RecommendRequest(movie_ids=s, tag_weight=0.5)
        q = query_builder.build(req)
        if q.tag_sparse is not None:
            queries.append(q)
    queries = queries[:30]
    print(f"Prepared {len(queries)} sample queries with tag_sparse", flush=True)

    # 1. Matvec p50/p95 at 1, 2, 4 concurrent callers BEFORE vs AFTER
    print("\n--- 1. Matvec Concurrency (30 queries) ---", flush=True)
    for name, fn in [("BEFORE (old)", old_matvec), ("AFTER (new)", new_matvec)]:
        print(f"\n{name}:")
        for conc in [1, 2, 4]:
            def _task(q):
                t0 = time.perf_counter()
                fn(q.tag_sparse)
                return time.perf_counter() - t0

            # warm-up
            with ThreadPoolExecutor(max_workers=conc) as pool:
                list(pool.map(_task, queries[:conc]))
                t_wall0 = time.perf_counter()
                durations = list(pool.map(_task, queries))
                t_wall = time.perf_counter() - t_wall0

            p50 = np.percentile(durations, 50) * 1000
            p95 = np.percentile(durations, 95) * 1000
            print(f"  {conc} concurrent callers: p50={p50:5.1f}ms  p95={p95:5.1f}ms  wall={t_wall*1000:6.0f}ms", flush=True)

    # 2. score_tag for 55k and 295k exact cases BEFORE vs AFTER
    print("\n--- 2. score_tag for 55k and 295k Exact Cases ---", flush=True)
    # 55k filter
    langs = idx.languages
    lang_codes = idx.lang_to_code
    filter_55k_lang = "ja"
    f_55k = Filters(languages=frozenset([filter_55k_lang]))
    mask_55k = meta_index.mask_for(f_55k)
    pool_55k = idx.ids[mask_55k].tolist()
    print(f"55k pool size: {len(pool_55k):,}", flush=True)

    # 295k filter
    f_295k = Filters(genres=frozenset(["Drama"]))
    mask_295k = meta_index.mask_for(f_295k)
    pool_295k = idx.ids[mask_295k].tolist()
    print(f"295k pool size: {len(pool_295k):,}", flush=True)

    # Test queries (10 queries for scoring benchmark)
    test_queries = queries[:10]

    for case_name, pool in [("55k filter", pool_55k), ("295k filter", pool_295k)]:
        print(f"\nCase: {case_name} ({len(pool):,} items)")
        # Old score_tag
        old_durations = [old_score_tag(pool, q) for q in test_queries]
        old_p50 = np.percentile(old_durations, 50) * 1000
        old_p95 = np.percentile(old_durations, 95) * 1000
        print(f"  BEFORE (old CSR slice): p50={old_p50:6.1f}ms  p95={old_p95:6.1f}ms", flush=True)

        # New score_tag
        new_durations = [new_score_tag(pool, q) for q in test_queries]
        new_p50 = np.percentile(new_durations, 50) * 1000
        new_p95 = np.percentile(new_durations, 95) * 1000
        print(f"  AFTER  (new full matvec): p50={new_p50:6.1f}ms  p95={new_p95:6.1f}ms", flush=True)
        print(f"  Speedup: {old_p50 / new_p50:.2f}x", flush=True)

if __name__ == "__main__":
    main()
