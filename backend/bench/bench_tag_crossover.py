"""
Sweep pool sizes (20k, 50k, 100k, 150k, 200k, 295k) and measure score_tag
with CSR row slicing vs full dense matvec + index (20 runs each).
Reports free RAM and determines the crossover threshold.
"""
import sys, os, time, gc
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder

def main():
    print("=" * 80)
    print("ITEM 1: TAG SCORING THRESHOLD SWEEP (CSR SLICING VS FULL DENSE MATVEC)")
    print("=" * 80)

    # Check RAM
    vm = psutil.virtual_memory()
    free_ram_gb = vm.available / (1024 ** 3)
    total_ram_gb = vm.total / (1024 ** 3)
    print(f"System Free RAM: {free_ram_gb:.2f} GB / Total RAM: {total_ram_gb:.2f} GB")

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    idx = meta_index.get_index()

    # Generate a realistic query
    req = RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.5)
    q = query_builder.build(req)
    tag_sparse = q.tag_sparse
    assert tag_sparse is not None, "Query must have tag_sparse"

    tag_m = artifacts.tag_matrix()
    inv = artifacts.inv_norms()
    norms = artifacts.row_norms()
    V = tag_m.shape[1]

    # Pre-build dense query for full matvec
    qd = np.zeros(V, dtype=np.float64)
    qd[tag_sparse.indices] = tag_sparse.data

    pool_sizes = [20_000, 50_000, 100_000, 150_000, 200_000, 295_000]
    all_tag_rows = np.arange(tag_m.shape[0], dtype=np.int32)
    rng = np.random.default_rng(42)

    # Pre-sample row pools to ensure deterministic indexing
    pools = {}
    for sz in pool_sizes:
        pools[sz] = rng.choice(all_tag_rows, size=sz, replace=False)

    N_RUNS = 20
    results = {}

    print(f"\nEvaluating {N_RUNS} runs per pool size (laptop quiet)...", flush=True)

    for sz in pool_sizes:
        rows = pools[sz]
        print(f"\n--- Pool Size: {sz:,} rows ---", flush=True)

        # 1. CSR Row Slicing: tag_m[rows] @ q_sparse.T
        slice_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            sub_m = tag_m[rows]
            dots = (sub_m @ tag_sparse.T).toarray().ravel()
            sub_norms = norms[rows]
            valid = sub_norms > 0
            sims_slice = np.zeros(len(rows), dtype=np.float32)
            sims_slice[valid] = dots[valid] / sub_norms[valid]
            dt = (time.perf_counter() - t0) * 1000.0
            slice_times.append(dt)

        # 2. Full Dense Matvec + Indexing: (tag_m @ qd) * inv -> index[rows]
        matvec_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            full_dots = tag_m @ qd
            full_sims = (full_dots * inv).astype(np.float32)
            sims_matvec = full_sims[rows]
            dt = (time.perf_counter() - t0) * 1000.0
            matvec_times.append(dt)

        # Verify equivalence
        max_diff = np.max(np.abs(sims_slice - sims_matvec))
        assert max_diff < 1e-6, f"Mismatch at size {sz}: max_diff={max_diff}"

        p50_slice = np.percentile(slice_times, 50)
        p95_slice = np.percentile(slice_times, 95)
        mean_slice = np.mean(slice_times)

        p50_matvec = np.percentile(matvec_times, 50)
        p95_matvec = np.percentile(matvec_times, 95)
        mean_matvec = np.mean(matvec_times)

        speedup = p50_slice / p50_matvec
        winner = "Full Matvec" if p50_matvec < p50_slice else "CSR Slice"

        print(f"  CSR Slice:   p50={p50_slice:6.2f}ms  p95={p95_slice:6.2f}ms  mean={mean_slice:6.2f}ms")
        print(f"  Full Matvec: p50={p50_matvec:6.2f}ms  p95={p95_matvec:6.2f}ms  mean={mean_matvec:6.2f}ms")
        print(f"  Speedup: {speedup:5.2f}x  ->  Winner: {winner}")

        results[sz] = {
            "p50_slice": p50_slice,
            "p95_slice": p95_slice,
            "mean_slice": mean_slice,
            "p50_matvec": p50_matvec,
            "p95_matvec": p95_matvec,
            "mean_matvec": mean_matvec,
            "speedup": speedup,
            "winner": winner,
        }

    free_ram_after = psutil.virtual_memory().available / (1024 ** 3)
    print(f"\nFinal Free RAM: {free_ram_after:.2f} GB (delta: {free_ram_after - free_ram_gb:+.2f} GB)")

    # Find crossover
    crossover = None
    for sz in pool_sizes:
        if results[sz]["p50_matvec"] < results[sz]["p50_slice"]:
            crossover = sz
            break

    print("\n" + "=" * 80)
    print(f"CROSSOVER POINT: Pool Size = {crossover:,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
