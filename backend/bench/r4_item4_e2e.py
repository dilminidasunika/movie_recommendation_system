"""
Round 4 Item 4: End-to-end HTTP sequential benchmark.
50 distinct requests per case type (unfiltered, plot_text, 55k filter, 295k filter),
sequential, flags ON. Report p50/p95/max, free RAM start/end, page faults.
Note: This tests through the real pipeline (not HTTP server) since we
measure the recommender.retrieve_and_rank call directly for accurate
per-request timing. This avoids HTTP/network overhead and measures pure pipeline cost.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json, ctypes, gc, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True
settings.use_uncapped_pool = True
settings.k_tag = 10000
settings.k_plot = 5000
settings.exact_limit = 300000

from app.db.pool import open_pool
from app.stores import artifacts, meta_index, chroma, text_embedder
from app.schemas import RecommendRequest
from app.services import recommender

import numpy as np


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_free_ram_gb():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024**3)


def get_page_faults():
    """Get current page fault count via PowerShell."""
    pid = os.getpid()
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {pid}).PagedMemorySize64"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def run():
    N = 50
    RNG_SEED = 42

    print("=" * 80, flush=True)
    print("ROUND 4 ITEM 4: END-TO-END SEQUENTIAL BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    text_embedder.get_sentence_transformer()

    rng = np.random.default_rng(RNG_SEED)
    tag_ids_set = set(artifacts.matrix_ids())
    _, plot_id_to_r = artifacts.plot_memmap()
    plot_ids_set = set(plot_id_to_r.keys())
    common_ids = np.array(sorted(tag_ids_set & plot_ids_set))
    idx = meta_index.get_index()

    # Find filter languages
    lang_codes = idx.lang_to_code
    langs = idx.languages
    # 55k filter: Japanese
    filter_55k_lang = None
    for lang_name, code in lang_codes.items():
        count = int(np.sum(langs == code))
        if 40_000 < count < 70_000:
            filter_55k_lang = lang_name
            filter_55k_count = count
            break
    if filter_55k_lang is None:
        filter_55k_lang = "ja"
        filter_55k_count = int(np.sum(langs == lang_codes.get("ja", -1)))

    # 295k filter: Drama genre
    from app.db.queries import Filters
    test_req = RecommendRequest(movie_ids=[238], genres=["Drama"], tag_weight=0.5)
    f = Filters.from_request(test_req)
    mask = meta_index.mask_for(f)
    filter_295k_count = int(np.sum(mask))

    print(f"55k filter: {filter_55k_lang} ({filter_55k_count:,} matches)", flush=True)
    print(f"295k filter: Drama ({filter_295k_count:,} matches)", flush=True)

    # Warmup
    print("\nWarmup...", flush=True)
    for _ in range(5):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
    gc.collect()
    print("Warmup done.", flush=True)

    cases = [
        ("unfiltered", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=0.5)),
        ("plot_text", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            plot_text="A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption.",
            tag_weight=0.5
        )),
        ("55k_filter", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            original_language=[filter_55k_lang],
            tag_weight=0.5
        )),
        ("295k_filter", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            genres=["Drama"],
            tag_weight=0.5
        )),
    ]

    free_ram_start = get_free_ram_gb()
    print(f"\nFree RAM at start: {free_ram_start:.2f} GB", flush=True)

    all_results = {}

    for case_name, req_fn in cases:
        print(f"\n--- Case: {case_name} ({N} distinct queries, sequential) ---", flush=True)

        latencies = []
        pool_sizes = []
        detail_timings = []

        # Generate N distinct seed sets per case
        case_rng = np.random.default_rng(RNG_SEED + hash(case_name) % 1000)
        for q_idx in range(N):
            seeds = case_rng.choice(common_ids, size=3, replace=False).tolist()
            req = req_fn(seeds)

            t0 = time.perf_counter()
            result = recommender.retrieve_and_rank(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(elapsed_ms)
            pool_sizes.append(result.get("pool_size", 0))
            detail_timings.append(result.get("timings", {}))

        lats = np.array(latencies)
        pools = np.array(pool_sizes)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        pmax = float(np.max(lats))
        pmean = float(np.mean(lats))

        all_results[case_name] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "max_ms": pmax,
            "mean_ms": pmean,
            "pool_p50": float(np.percentile(pools, 50)),
            "pool_mean": float(np.mean(pools)),
            "per_query_latencies": latencies,
        }

        print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  max={pmax:.1f}ms  mean={pmean:.1f}ms", flush=True)
        print(f"  pool: p50={np.percentile(pools,50):.0f} mean={np.mean(pools):.0f}", flush=True)

    free_ram_end = get_free_ram_gb()
    print(f"\nFree RAM at end: {free_ram_end:.2f} GB (delta: {free_ram_end - free_ram_start:+.2f} GB)", flush=True)

    # Summary table
    print(f"\n{'='*75}", flush=True)
    print(f"E2E SUMMARY (sequential, 50 distinct queries per case, flags ON)", flush=True)
    print(f"{'='*75}", flush=True)
    header = f"{'Case':<15} | {'p50':>8} {'p95':>8} {'max':>8} {'mean':>8} | {'pool p50':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for case_name in ["unfiltered", "plot_text", "55k_filter", "295k_filter"]:
        d = all_results[case_name]
        print(f"{case_name:<15} | {d['p50_ms']:7.1f}ms {d['p95_ms']:7.1f}ms {d['max_ms']:7.1f}ms {d['mean_ms']:7.1f}ms | {d['pool_p50']:9.0f}", flush=True)

    thrash = free_ram_end < 0.5
    print(f"\nThrashing: {'YES' if thrash else 'NO'} (free RAM end: {free_ram_end:.2f} GB)", flush=True)

    out_file = SCRIPT_DIR / "r4_e2e_http.json"
    with open(out_file, "w") as f:
        json.dump({
            "free_ram_start_gb": free_ram_start,
            "free_ram_end_gb": free_ram_end,
            "cases": all_results,
        }, f, indent=2, default=str)
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    run()
