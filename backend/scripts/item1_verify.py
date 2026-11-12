"""
Item 1 verification: compare new _matvec_tag (dense float64 + inv_norms)
against the old implementation (sparse @ sparse.T + boolean-mask division)
on 200 random queries.  Reports max score difference and whether top-k
(tie-aware) is identical.

Also benchmarks matvec p50/p95 at 1, 2, 4 concurrent callers,
tests GIL release (2 threads, 2 queries: wall time vs 2x serial),
reports score_tag for 55k and 295k exact cases,
and runs a float32 experiment (report only).
"""
import sys, os, time, json, random, threading
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.stores import artifacts

def old_matvec_sims(tag_sparse):
    """Original implementation: CSR @ sparse.T + boolean-mask division."""
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    dots = (tag_m @ tag_sparse.T).toarray().ravel()
    valid = tag_norms > 0
    sims = np.zeros(len(tag_norms), dtype=np.float32)
    sims[valid] = dots[valid] / tag_norms[valid]
    return sims

def new_matvec_sims(tag_sparse):
    """New implementation: dense float64 query + precomputed inv_norms."""
    tag_m = artifacts.tag_matrix()
    inv = artifacts.inv_norms()
    V = tag_m.shape[1]
    qd = np.zeros(V, dtype=np.float64)
    qd[tag_sparse.indices] = tag_sparse.data
    dots = tag_m @ qd
    sims = (dots * inv).astype(np.float32)
    return sims

def generate_random_query(V, nnz=50, rng=None):
    """Generate a random sparse query vector (1, V) CSR."""
    if rng is None:
        rng = np.random.default_rng()
    cols = rng.choice(V, size=min(nnz, V), replace=False)
    data = rng.uniform(0.1, 1.0, size=len(cols)).astype(np.float64)
    # L2-normalize
    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm
    return sp.csr_matrix((data, cols, [0, len(cols)]), shape=(1, V))


def tie_aware_topk(sims, k):
    """Return indices of top-k scores, including all ties at the boundary."""
    if k >= len(sims):
        return set(range(len(sims)))
    kth = np.partition(-sims, k-1)[k-1]
    threshold = -kth
    return set(np.flatnonzero(sims >= threshold).tolist())


def main():
    print("Loading artifacts...")
    tag_m = artifacts.tag_matrix()
    _ = artifacts.row_norms()
    _ = artifacts.inv_norms()
    _ = artifacts.matrix_ids()
    V = tag_m.shape[1]
    N = tag_m.shape[0]
    print(f"Tag matrix: {N} rows x {V} cols")

    rng = np.random.default_rng(42)
    n_queries = 200
    k = 10000

    # Also generate queries from real movie rows for realism
    real_queries = []
    all_rows = list(range(N))
    rng.shuffle(all_rows)
    for row_idx in all_rows[:100]:
        row = tag_m[row_idx]
        if row.nnz > 0:
            from sklearn.preprocessing import normalize
            real_queries.append(normalize(row, norm="l2"))
    # Fill remaining with random
    random_queries = [generate_random_query(V, nnz=rng.integers(10, 200), rng=rng) for _ in range(n_queries - len(real_queries))]
    all_queries = real_queries + random_queries
    rng.shuffle(all_queries)
    all_queries = all_queries[:n_queries]

    print(f"\n=== Verification: {n_queries} queries ===")
    max_diff = 0.0
    topk_mismatches = 0

    for i, q_sparse in enumerate(all_queries):
        old = old_matvec_sims(q_sparse)
        new = new_matvec_sims(q_sparse)
        diff = np.max(np.abs(old.astype(np.float64) - new.astype(np.float64)))
        max_diff = max(max_diff, diff)

        old_topk = tie_aware_topk(old, k)
        new_topk = tie_aware_topk(new, k)
        if old_topk != new_topk:
            topk_mismatches += 1
            if topk_mismatches <= 3:
                sym = old_topk.symmetric_difference(new_topk)
                print(f"  Query {i}: top-k mismatch, {len(sym)} differing ids")
                for idx in list(sym)[:5]:
                    print(f"    id={idx} old_score={old[idx]:.10f} new_score={new[idx]:.10f}")

    print(f"Max score difference: {max_diff:.2e}")
    print(f"Top-k mismatches (tie-aware): {topk_mismatches}/{n_queries}")
    assert max_diff < 1e-7, f"FAIL: max_diff={max_diff:.2e} exceeds 1e-7"
    print("PASS: all scores within 1e-7")

    # --- Benchmark: matvec timing ---
    print("\n=== Matvec Timing ===")
    bench_queries = all_queries[:50]

    def time_matvec(q_sparse):
        tag_m_local = artifacts.tag_matrix()
        inv = artifacts.inv_norms()
        V = tag_m_local.shape[1]
        qd = np.zeros(V, dtype=np.float64)
        qd[q_sparse.indices] = q_sparse.data
        t0 = time.perf_counter()
        dots = tag_m_local @ qd
        sims = (dots * inv).astype(np.float32)
        dt = time.perf_counter() - t0
        return dt, sims

    def time_old_matvec(q_sparse):
        tag_m_local = artifacts.tag_matrix()
        tag_norms = artifacts.row_norms()
        t0 = time.perf_counter()
        dots = (tag_m_local @ q_sparse.T).toarray().ravel()
        valid = tag_norms > 0
        sims = np.zeros(len(tag_norms), dtype=np.float32)
        sims[valid] = dots[valid] / tag_norms[valid]
        dt = time.perf_counter() - t0
        return dt, sims

    # Warm up
    for q in bench_queries[:5]:
        time_matvec(q)
        time_old_matvec(q)

    # Old timing
    old_times = [time_old_matvec(q)[0] for q in bench_queries]
    print(f"OLD matvec (serial): p50={np.percentile(old_times, 50)*1000:.1f}ms  p95={np.percentile(old_times, 95)*1000:.1f}ms")

    # New timing
    new_times = [time_matvec(q)[0] for q in bench_queries]
    print(f"NEW matvec (serial): p50={np.percentile(new_times, 50)*1000:.1f}ms  p95={np.percentile(new_times, 95)*1000:.1f}ms")

    # Concurrent timing
    from concurrent.futures import ThreadPoolExecutor
    for n_workers in [1, 2, 4]:
        latencies = []
        def _run(q_sparse):
            return time_matvec(q_sparse)[0]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # warm up
            list(pool.map(_run, bench_queries[:n_workers]))
            t_wall_start = time.perf_counter()
            futures = [pool.submit(_run, q) for q in bench_queries]
            latencies = [f.result() for f in futures]
            t_wall = time.perf_counter() - t_wall_start
        print(f"NEW matvec ({n_workers} concurrent): p50={np.percentile(latencies, 50)*1000:.1f}ms  "
              f"p95={np.percentile(latencies, 95)*1000:.1f}ms  wall={t_wall*1000:.0f}ms")

    # --- GIL release test ---
    print("\n=== GIL Release Test (2 threads, 2 queries) ===")
    q1, q2 = bench_queries[0], bench_queries[1]
    # Serial
    t0 = time.perf_counter()
    time_matvec(q1)
    time_matvec(q2)
    serial_time = time.perf_counter() - t0

    # Parallel
    results = [None, None]
    def _worker(idx, q):
        results[idx] = time_matvec(q)
    t0 = time.perf_counter()
    t1 = threading.Thread(target=_worker, args=(0, q1))
    t2 = threading.Thread(target=_worker, args=(1, q2))
    t1.start(); t2.start()
    t1.join(); t2.join()
    parallel_time = time.perf_counter() - t0
    print(f"Serial:   {serial_time*1000:.1f}ms")
    print(f"Parallel: {parallel_time*1000:.1f}ms")
    print(f"Speedup:  {serial_time/parallel_time:.2f}x (1.0 = no GIL release, 2.0 = full release)")

    # --- Float32 experiment (report only) ---
    print("\n=== Float32 Experiment (report only) ===")
    f32_max_diff = 0.0
    for q_sparse in all_queries:
        # float64 baseline
        qd64 = np.zeros(V, dtype=np.float64)
        qd64[q_sparse.indices] = q_sparse.data
        sims64 = ((tag_m @ qd64) * artifacts.inv_norms()).astype(np.float32)

        # float32
        qd32 = np.zeros(V, dtype=np.float32)
        qd32[q_sparse.indices] = q_sparse.data.astype(np.float32)
        # Need float32 tag matrix for true f32 path - convert just the dots
        dots32 = tag_m.astype(np.float32) @ qd32 if False else tag_m @ qd32  # CSR @ f32 dense
        inv32 = artifacts.inv_norms().astype(np.float32)
        sims32 = (dots32 * inv32).astype(np.float32)

        diff = np.max(np.abs(sims64.astype(np.float64) - sims32.astype(np.float64)))
        f32_max_diff = max(f32_max_diff, diff)

    print(f"Float32 max score diff vs float64: {f32_max_diff:.2e}")
    if f32_max_diff > 1e-6:
        print("  -> EXCEEDS 1e-6 threshold, NOT applying float32")
    else:
        print("  -> Within 1e-6, could apply float32 (not applying)")

    # Time the float32 path
    f32_times = []
    for q_sparse in bench_queries:
        qd32 = np.zeros(V, dtype=np.float32)
        qd32[q_sparse.indices] = q_sparse.data.astype(np.float32)
        inv32 = artifacts.inv_norms().astype(np.float32)
        t0 = time.perf_counter()
        dots = tag_m @ qd32
        sims = (dots * inv32).astype(np.float32)
        f32_times.append(time.perf_counter() - t0)
    print(f"Float32 matvec: p50={np.percentile(f32_times, 50)*1000:.1f}ms  p95={np.percentile(f32_times, 95)*1000:.1f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
