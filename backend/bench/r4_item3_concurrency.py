"""
Round 4 Item 3: Full-pipeline concurrency.
8 simultaneous DISTINCT requests through the real pipeline at PIPELINE_SEM=4 / BLAS=1.
3 iterations + sequential baseline.
Also report matvec alone under 1, 2, 4 concurrent callers.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json, gc, ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from app.services import recommender, query_builder, candidates

import numpy as np
import scipy.sparse as sp


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


def run():
    print("=" * 80, flush=True)
    print("ROUND 4 ITEM 3: FULL-PIPELINE CONCURRENCY", flush=True)
    free_ram = get_free_ram_gb()
    print(f"Free RAM at start: {free_ram:.2f} GB", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    text_embedder.get_sentence_transformer()

    rng = np.random.default_rng(42)
    tag_ids_set = set(artifacts.matrix_ids())
    _, plot_id_to_r = artifacts.plot_memmap()
    plot_ids_set = set(plot_id_to_r.keys())
    common_ids = np.array(sorted(tag_ids_set & plot_ids_set))

    # Warmup
    print("Warmup...", flush=True)
    for _ in range(3):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
    gc.collect()
    print("Warmup done.", flush=True)

    # Generate 24 distinct seed sets (8 per iteration × 3 iterations)
    all_seed_sets = [rng.choice(common_ids, size=3, replace=False).tolist() for _ in range(24)]

    # === Part A: Sequential baseline ===
    print(f"\n--- Sequential baseline (8 distinct requests) ---", flush=True)
    seq_lats = []
    for i in range(8):
        seeds = all_seed_sets[i]
        t0 = time.perf_counter()
        result = recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
        elapsed = (time.perf_counter() - t0) * 1000.0
        seq_lats.append(elapsed)
    seq_arr = np.array(seq_lats)
    seq_wall = sum(seq_lats)
    print(f"  Sequential: wall={seq_wall:.0f}ms, per-req p50={np.percentile(seq_arr,50):.1f}ms p95={np.percentile(seq_arr,95):.1f}ms", flush=True)

    # === Part B: 8 concurrent, 3 iterations ===
    print(f"\n--- Concurrent (8 threads, 3 iterations) ---", flush=True)
    conc_results = []
    for it in range(3):
        seed_batch = all_seed_sets[it*8:(it+1)*8]
        per_req_lats = []
        t_wall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for idx, seeds in enumerate(seed_batch):
                def worker(s=seeds):
                    t0 = time.perf_counter()
                    result = recommender.retrieve_and_rank(RecommendRequest(movie_ids=s, tag_weight=0.5))
                    return (time.perf_counter() - t0) * 1000.0, result.get("pool_size", 0)
                futures[pool.submit(worker)] = idx

            for f in as_completed(futures):
                lat, ps = f.result()
                per_req_lats.append(lat)

        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
        lats = np.array(per_req_lats)
        throughput = 8000.0 / wall_ms  # requests per second
        conc_results.append({
            "wall_ms": wall_ms,
            "per_req_p50": float(np.percentile(lats, 50)),
            "per_req_p95": float(np.percentile(lats, 95)),
            "per_req_mean": float(np.mean(lats)),
            "throughput_rps": throughput,
        })
        print(f"  Iter {it+1}: wall={wall_ms:.0f}ms, per-req p50={np.percentile(lats,50):.1f}ms p95={np.percentile(lats,95):.1f}ms throughput={throughput:.1f} req/s", flush=True)

    # === Part C: Matvec alone under 1, 2, 4 concurrent callers ===
    print(f"\n--- Matvec alone under concurrency ---", flush=True)
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()

    # Build 8 queries
    matvec_queries = []
    for i in range(8):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        q = query_builder.build(req)
        matvec_queries.append(q)

    def do_matvec(q):
        t0 = time.perf_counter()
        dots = (tag_m @ q.tag_sparse.T).toarray().ravel()
        valid = tag_norms > 0
        sims = np.zeros(len(tag_norms), dtype=np.float32)
        sims[valid] = dots[valid] / tag_norms[valid]
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed

    for n_workers in [1, 2, 4]:
        lats = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(do_matvec, matvec_queries[i % len(matvec_queries)]) for i in range(8)]
            for f in as_completed(futures):
                lats.append(f.result())
        lats = np.array(lats)
        print(f"  {n_workers} concurrent: p50={np.percentile(lats,50):.1f}ms p95={np.percentile(lats,95):.1f}ms mean={np.mean(lats):.1f}ms", flush=True)

    free_ram_end = get_free_ram_gb()
    print(f"\nFree RAM at end: {free_ram_end:.2f} GB", flush=True)

    # Save
    out_file = SCRIPT_DIR / "r4_concurrency.json"
    with open(out_file, "w") as f:
        json.dump({
            "sequential": {
                "wall_ms": seq_wall,
                "per_req_p50": float(np.percentile(seq_arr, 50)),
                "per_req_p95": float(np.percentile(seq_arr, 95)),
            },
            "concurrent": conc_results,
            "free_ram_start": free_ram,
            "free_ram_end": free_ram_end,
        }, f, indent=2)
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    run()
