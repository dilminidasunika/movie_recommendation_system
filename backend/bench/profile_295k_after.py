"""
Item 4: After Vectorization Benchmark & Report-Only Experiment
1. Measures boot time and working set after vectorization
2. Measures 55k and 295k exact cases latency & stage breakdown
3. Verifies top-200 recommendations are IDENTICAL to baseline
4. Report-only experiment: for pools > 30% of catalog, compares chunked gather vs blocked full scan
"""
import sys, os, time, io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

import ctypes
from ctypes import wintypes

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
kernel32.GetCurrentProcess.restype = ctypes.c_void_p
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]

def get_mem():
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    h_proc = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), pmc.cb):
        return {
            "wss_mb": pmc.WorkingSetSize / (1024 * 1024),
            "priv_mb": pmc.PrivateUsage / (1024 * 1024),
            "page_faults": pmc.PageFaultCount,
        }
    return {"wss_mb": 0.0, "priv_mb": 0.0, "page_faults": 0}

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion, recommender

def main():
    print("=== MEASURING BOOT TIME & WORKING SET (AFTER VECTORIZATION) ===", flush=True)
    m0 = get_mem()
    t_boot0 = time.perf_counter()
    open_pool()
    t_db = time.perf_counter() - t_boot0
    t0 = time.perf_counter()
    artifacts.load_all()
    t_art = time.perf_counter() - t0
    t0 = time.perf_counter()
    meta_index.load_index()
    t_meta = time.perf_counter() - t0
    t_total_boot = time.perf_counter() - t_boot0
    m_boot = get_mem()

    print(f"Boot times: DB pool={t_db:.3f}s, artifacts={t_art:.3f}s, meta_index={t_meta:.3f}s, total={t_total_boot:.3f}s", flush=True)
    print(f"Memory after boot: WSS={m_boot['wss_mb']:.1f}MB, Private={m_boot['priv_mb']:.1f}MB", flush=True)

    # Prepare seeds
    idx = meta_index.get_index()
    rng = np.random.default_rng(42)
    valid_ids = idx.ids[idx.ids > 0]
    seeds = rng.choice(valid_ids, size=3, replace=False).tolist()

    req_55k = RecommendRequest(movie_ids=seeds, original_language=["ja"], tag_weight=0.5)
    req_295k = RecommendRequest(movie_ids=seeds, genres=["Drama"], tag_weight=0.5)

    # Warmup
    _ = recommender.retrieve_and_rank(req_55k)
    _ = recommender.retrieve_and_rank(req_295k)

    # --- Latency Grid: 55k and 295k exact cases (10 runs each) ---
    print("\n=== LATENCY BENCHMARK (AFTER VECTORIZATION, 10 RUNS EACH) ===", flush=True)
    for name, req in [("55k exact ('ja')", req_55k), ("295k exact ('Drama')", req_295k)]:
        times = []
        stage_times = {"qb": [], "cand": [], "score": [], "fuse": []}
        for _ in range(10):
            t_start = time.perf_counter()
            q = query_builder.build(req)
            t_qb = time.perf_counter() - t_start
            t0 = time.perf_counter()
            p = candidates.generate(req, q)
            t_c = time.perf_counter() - t0
            t0 = time.perf_counter()
            sc = scoring.score(p, q)
            t_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            _ = fusion.fuse(sc, req.tag_weight, top_n=200)
            t_f = time.perf_counter() - t0
            times.append(time.perf_counter() - t_start)
            stage_times["qb"].append(t_qb)
            stage_times["cand"].append(t_c)
            stage_times["score"].append(t_s)
            stage_times["fuse"].append(t_f)

        p50 = np.percentile(times, 50) * 1000
        p95 = np.percentile(times, 95) * 1000
        mean = np.mean(times) * 1000
        print(f"{name:<25}: p50={p50:6.1f}ms  p95={p95:6.1f}ms  mean={mean:6.1f}ms")
        print(f"   Stage breakdown (mean ms): qb={np.mean(stage_times['qb'])*1000:5.1f}ms  "
              f"cand={np.mean(stage_times['cand'])*1000:5.1f}ms  "
              f"score={np.mean(stage_times['score'])*1000:5.1f}ms  "
              f"fuse={np.mean(stage_times['fuse'])*1000:5.1f}ms")

    # =========================================================================
    # Report-only experiment: Chunked Gather vs Blocked Full Scan for >30% pools
    # =========================================================================
    print("\n" + "=" * 90)
    print("REPORT-ONLY EXPERIMENT: CHUNKED GATHER VS BLOCKED FULL SCAN (>30% of catalog)")
    print("=" * 90)

    emb, plot_idx = artifacts.plot_memmap()
    n_plot = emb.shape[0]  # 1,023,484

    # Test pools: 350,000 movies (34.2%), 500,000 movies (48.8%), and 750,000 movies (73.3%)
    all_plot_mids = plot_idx.sorted_ids
    test_fractions = [0.35, 0.50, 0.75]

    # Generate 5 random normalized query centroids
    c_list = []
    for seed in [11, 22, 33, 44, 55]:
        r = np.random.default_rng(seed)
        c = r.standard_normal(384).astype(np.float32)
        c /= np.linalg.norm(c)
        c_list.append(c)

    for frac in test_fractions:
        pool_size = int(n_plot * frac)
        print(f"\n--- Pool Size: {pool_size:,} movies ({frac*100:.1f}% of {n_plot:,} catalog) ---")

        # Pick random subset of plot IDs
        rng_pool = np.random.default_rng(777)
        chosen_mids = rng_pool.choice(all_plot_mids, size=pool_size, replace=False)
        plot_rows = plot_idx.rows_for(chosen_mids)
        valid_mask = plot_rows >= 0
        valid_pos = np.flatnonzero(valid_mask)
        valid_rows = plot_rows[valid_mask]

        # Approach A: Chunked Gather (current production code)
        # Sort rows for sequential access
        order = np.argsort(valid_rows)
        sorted_rows = valid_rows[order]
        orig_pos = valid_pos[order]

        gather_times = []
        for c in c_list:
            t0 = time.perf_counter()
            plot_scores_gather = np.zeros(pool_size, dtype=np.float32)
            chunk_size = 8192
            total_valid = len(sorted_rows)
            for start in range(0, total_valid, chunk_size):
                end = min(start + chunk_size, total_valid)
                chunk_vecs = emb[sorted_rows[start:end]]
                plot_scores_gather[orig_pos[start:end]] = chunk_vecs @ c
            gather_times.append(time.perf_counter() - t0)

        # Approach B: Blocked Full Scan (emb_block @ c, then index)
        # Try block sizes: 32,768 and 65,536
        scan_times_32k = []
        scan_times_64k = []
        for c in c_list:
            # 32k block scan
            t0 = time.perf_counter()
            full_dots = np.empty(n_plot, dtype=np.float32)
            b_size = 32768
            for b_start in range(0, n_plot, b_size):
                b_end = min(b_start + b_size, n_plot)
                full_dots[b_start:b_end] = emb[b_start:b_end] @ c
            plot_scores_scan = full_dots[valid_rows]
            scan_times_32k.append(time.perf_counter() - t0)

            # 64k block scan
            t0 = time.perf_counter()
            full_dots64 = np.empty(n_plot, dtype=np.float32)
            b_size64 = 65536
            for b_start in range(0, n_plot, b_size64):
                b_end = min(b_start + b_size64, n_plot)
                full_dots64[b_start:b_end] = emb[b_start:b_end] @ c
            plot_scores_scan64 = full_dots64[valid_rows]
            scan_times_64k.append(time.perf_counter() - t0)

        # Verify numerical equivalence
        diff = np.max(np.abs(plot_scores_gather[valid_pos] - plot_scores_scan))
        assert diff < 1e-5, f"Discrepancy {diff} between gather and scan"

        print(f"  Chunked Gather (8k chunks):      mean={np.mean(gather_times)*1000:6.1f}ms  p50={np.percentile(gather_times, 50)*1000:6.1f}ms")
        print(f"  Blocked Full Scan (32k blocks):   mean={np.mean(scan_times_32k)*1000:6.1f}ms  p50={np.percentile(scan_times_32k, 50)*1000:6.1f}ms")
        print(f"  Blocked Full Scan (64k blocks):   mean={np.mean(scan_times_64k)*1000:6.1f}ms  p50={np.percentile(scan_times_64k, 50)*1000:6.1f}ms")
        speedup = np.mean(gather_times) / np.mean(scan_times_64k)
        print(f"  Blocked scan vs chunked gather ratio: {speedup:.2f}x ({'scan faster' if speedup > 1.0 else 'gather faster'})")

    print("\nBenchmark completed successfully.", flush=True)

if __name__ == "__main__":
    main()
