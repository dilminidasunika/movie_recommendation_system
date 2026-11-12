"""
Round 4 Item 2: Per-query latency grid with randomized config order.
50 distinct queries per (k_tag, k_plot) config, with untimed warm-up per config.
Reports p50/p95 from the per-query values.
Run with: python scripts/r4_item2_latency_grid.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json, ctypes, gc
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True
settings.use_uncapped_pool = True

from app.db.pool import open_pool
from app.stores import artifacts, meta_index, chroma, text_embedder
from app.schemas import RecommendRequest, TagIn
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


def run():
    N_QUERIES = 50
    RNG_SEED = 42

    print("=" * 80, flush=True)
    print("ROUND 4 ITEM 2: PER-QUERY LATENCY GRID (MEASURED, NOT ESTIMATED)", flush=True)
    free_ram = get_free_ram_gb()
    print(f"Free RAM at start: {free_ram:.2f} GB", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    text_embedder.get_sentence_transformer()

    rng = np.random.default_rng(RNG_SEED)
    all_ids = list(artifacts.id_to_row().keys())
    # We only use common_ids (have both tag and plot)
    tag_ids_set = set(artifacts.matrix_ids())
    _, plot_id_to_r = artifacts.plot_memmap()
    plot_ids_set = set(plot_id_to_r.keys())
    common_ids = np.array(sorted(tag_ids_set & plot_ids_set))

    # Pre-generate 50 distinct seed sets
    seed_sets = [rng.choice(common_ids, size=3, replace=False).tolist() for _ in range(N_QUERIES)]

    # (k_tag, k_plot) configs
    configs = [(kt, kp) for kt in [2000, 5000, 10000] for kp in [2000, 5000, 10000]]

    # Randomize config order
    config_order = list(range(len(configs)))
    rng2 = np.random.default_rng(123)
    rng2.shuffle(config_order)
    print(f"Config evaluation order (randomized): {[configs[i] for i in config_order]}", flush=True)

    # Global warmup (untimed)
    print("Global warmup (3 queries)...", flush=True)
    for i in range(3):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        try:
            recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
        except Exception:
            pass
    gc.collect()
    print("Global warmup done.", flush=True)

    results = {}

    for ci in config_order:
        kt, kp = configs[ci]
        settings.k_tag = kt
        settings.k_plot = kp

        config_key = f"kt_{kt}_kp_{kp}"
        print(f"\n--- Config ({kt}, {kp}) ---", flush=True)

        # Per-config warm-up (untimed, 3 queries)
        for i in range(3):
            seeds = rng.choice(common_ids, size=3, replace=False).tolist()
            try:
                recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
            except Exception:
                pass
        gc.collect()

        # Timed runs: 50 distinct queries
        latencies = []
        timing_details = []
        for q_idx, seeds in enumerate(seed_sets):
            req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
            t0 = time.perf_counter()
            result = recommender.retrieve_and_rank(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            t = result.get("timings", {})
            timing_details.append({
                "query_idx": q_idx,
                "total_ms": elapsed_ms,
                "pool_size": result.get("pool_size", 0),
                "matvec_tag": t.get("matvec_tag"),
                "ann_plot": t.get("ann_plot"),
                "score_tag": t.get("score_tag"),
                "score_plot": t.get("score_plot"),
            })

        lats = np.array(latencies)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        mean = float(np.mean(lats))
        pmin = float(np.min(lats))
        pmax = float(np.max(lats))

        results[config_key] = {
            "kt": kt,
            "kp": kp,
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": mean,
            "min_ms": pmin,
            "max_ms": pmax,
            "per_query": timing_details,
        }

        print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  mean={mean:.1f}ms  min={pmin:.1f}ms  max={pmax:.1f}ms", flush=True)

        # Print sub-timing medians
        mt_vals = [d["matvec_tag"] for d in timing_details if d["matvec_tag"] is not None]
        ap_vals = [d["ann_plot"] for d in timing_details if d["ann_plot"] is not None]
        st_vals = [d["score_tag"] for d in timing_details if d["score_tag"] is not None]
        sp_vals = [d["score_plot"] for d in timing_details if d["score_plot"] is not None]
        pool_vals = [d["pool_size"] for d in timing_details]

        if mt_vals:
            print(f"    matvec_tag  p50={np.percentile(mt_vals, 50)*1000:.1f}ms", flush=True)
        if ap_vals:
            print(f"    ann_plot    p50={np.percentile(ap_vals, 50)*1000:.1f}ms", flush=True)
        if st_vals:
            print(f"    score_tag   p50={np.percentile(st_vals, 50)*1000:.1f}ms", flush=True)
        if sp_vals:
            print(f"    score_plot  p50={np.percentile(sp_vals, 50)*1000:.1f}ms", flush=True)
        print(f"    pool_size   p50={np.percentile(pool_vals, 50):.0f}", flush=True)

    # Summary table
    free_ram_end = get_free_ram_gb()
    print(f"\n{'='*70}", flush=True)
    print(f"LATENCY GRID SUMMARY (measured, flags ON, free RAM: start {free_ram:.2f} -> end {free_ram_end:.2f} GB)", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'kt':>6} {'kp':>6} | {'p50':>8} {'p95':>8} {'mean':>8} {'min':>8} {'max':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for kt in [2000, 5000, 10000]:
        for kp in [2000, 5000, 10000]:
            key = f"kt_{kt}_kp_{kp}"
            d = results[key]
            print(f"{kt:6d} {kp:6d} | {d['p50_ms']:7.1f}ms {d['p95_ms']:7.1f}ms {d['mean_ms']:7.1f}ms {d['min_ms']:7.1f}ms {d['max_ms']:7.1f}ms", flush=True)

    # Save
    out_file = SCRIPT_DIR / "r4_latency_grid.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    run()
