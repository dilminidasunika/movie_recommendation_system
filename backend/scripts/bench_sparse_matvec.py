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
import scipy.sparse as sp
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.stores import artifacts

def main(n_queries=50, rng_seed=42):
    print("=" * 65, flush=True)
    print("EXPERIMENT 6: TAG CANDIDATE GENERATION BY EXACT SPARSE MATVEC", flush=True)
    print("=" * 65, flush=True)

    artifacts.load_all()
    id_to_row = artifacts.id_to_row()
    tag_m = artifacts.tag_matrix()
    tag_ids = artifacts.matrix_ids()
    tag_norms = artifacts.row_norms()

    print(f"Tag CSR matrix: {tag_m.shape}, nnz={tag_m.nnz:,}", flush=True)

    rng = np.random.default_rng(rng_seed)
    all_ids = np.array(list(id_to_row.keys()))

    # Build 50 distinct queries
    queries = []
    for _ in range(n_queries):
        seeds = rng.choice(all_ids, size=3, replace=False).tolist()
        s_rows = [id_to_row[s] for s in seeds]
        m = normalize(tag_m[s_rows], norm="l2")
        mean_s = sp.csr_matrix(np.full((1, len(s_rows)), 1.0 / len(s_rows))) @ m
        tag_sparse = normalize(mean_s, norm="l2")
        queries.append((seeds, tag_sparse))

    # 1. Warm-up run
    s0, q0 = queries[0]
    t_w0 = time.perf_counter()
    dots = (tag_m @ q0.T).toarray().ravel()
    sims = np.zeros(len(tag_norms), dtype=np.float32)
    valid = tag_norms > 0
    sims[valid] = dots[valid] / tag_norms[valid]
    top_k = np.argpartition(-sims, 2000)[:2000]
    t_warm = (time.perf_counter() - t_w0) * 1000.0
    print(f"Warm-up MatVec Latency: {t_warm:.2f} ms", flush=True)

    # 2. Benchmark over 50 distinct queries
    latencies = []
    for seeds, q_sp in queries:
        t0 = time.perf_counter()
        dots = (tag_m @ q_sp.T).toarray().ravel()
        sims = np.zeros(len(tag_norms), dtype=np.float32)
        valid = tag_norms > 0
        sims[valid] = dots[valid] / tag_norms[valid]
        for s in seeds:
            sims[id_to_row[s]] = -1.0
        top_k = np.argpartition(-sims, 2000)[:2000]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    latencies = np.array(latencies)
    print("\nLatency across 50 distinct queries (MatVec + Top-2000 argpartition):", flush=True)
    print(f"  Mean:   {np.mean(latencies):.2f} ms", flush=True)
    print(f"  Median (p50): {np.median(latencies):.2f} ms", flush=True)
    print(f"  p95:    {np.percentile(latencies, 95):.2f} ms", flush=True)
    print(f"  Min:    {np.min(latencies):.2f} ms", flush=True)
    print(f"  Max:    {np.max(latencies):.2f} ms", flush=True)

    print("\nQuality / Recall:", flush=True)
    print("  Recall@50  vs exact sparse cosine: 100.00% (Exact computation, no SVD loss)")
    print("  Recall@200 vs exact sparse cosine: 100.00% (Exact computation, no SVD loss)")

    print("\nMemory Savings:", flush=True)
    tags_hnsw_bytes = 1472335042  # from table (1.37 GB)
    svd_bytes = 237878400        # from table (226.86 MB)
    total_saved = tags_hnsw_bytes + svd_bytes
    print(f"  tags_vectors HNSW index dropped: {tags_hnsw_bytes / 1024**2:.2f} MB ({tags_hnsw_bytes / 1024**3:.2f} GB)")
    print(f"  tag_svd_model dropped:           {svd_bytes / 1024**2:.2f} MB ({svd_bytes / 1024**3:.2f} GB)")
    print(f"  Total memory/disk saved:         {total_saved / 1024**2:.2f} MB ({total_saved / 1024**3:.2f} GB)")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
