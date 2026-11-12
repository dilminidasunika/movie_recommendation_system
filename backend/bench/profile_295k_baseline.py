"""
Item 4: Baseline cProfile and timing for 55k and 295k exact cases.
Runs before any Item 4 vectorization changes to establish the baseline.
"""
import sys, os, time, cProfile, pstats, io
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
    print("=== MEASURING BOOT TIME & WORKING SET (BASELINE) ===", flush=True)
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

    # Requests
    req_55k = RecommendRequest(movie_ids=seeds, original_language=["ja"], tag_weight=0.5)
    req_295k = RecommendRequest(movie_ids=seeds, genres=["Drama"], tag_weight=0.5)

    # Warmup
    _ = recommender.retrieve_and_rank(req_55k)
    _ = recommender.retrieve_and_rank(req_295k)

    # --- cProfile 295k exact case ---
    print("\n=== cProfile 295k EXACT CASE (SINGLE RUN DETAILED BREAKDOWN) ===", flush=True)
    pr = cProfile.Profile()
    pr.enable()

    t_start = time.perf_counter()
    q = query_builder.build(req_295k)
    t_qb = time.perf_counter() - t_start

    t0 = time.perf_counter()
    pool = candidates.generate(req_295k, q)
    t_cand = time.perf_counter() - t0

    t0 = time.perf_counter()
    s = scoring.score(pool, q)
    t_score = time.perf_counter() - t0

    t0 = time.perf_counter()
    ranked = fusion.fuse(s, req_295k.tag_weight, top_n=200)
    t_fuse = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    pr.disable()

    print(f"Pool size: {len(pool):,} candidates")
    print(f"Stages: qb={t_qb*1000:6.1f}ms  cand={t_cand*1000:6.1f}ms  score={t_score*1000:6.1f}ms  fuse={t_fuse*1000:6.1f}ms  total={t_total*1000:6.1f}ms")

    # Print top functions from cProfile
    s_io = io.StringIO()
    ps = pstats.Stats(pr, stream=s_io).sort_stats("cumulative")
    ps.print_stats(30)
    print("\nTop 30 functions by cumulative time:\n" + s_io.getvalue())

    # --- Latency Grid: 55k and 295k exact cases (10 runs each) ---
    print("\n=== LATENCY BENCHMARK (10 RUNS EACH) ===", flush=True)
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

if __name__ == "__main__":
    main()
