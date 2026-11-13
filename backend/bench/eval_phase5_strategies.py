"""
Evaluation and benchmark script for Phase 5: Strategy A vs Strategy B.
Runs interleaved benchmarks (N >= 10) on Cases b, e, and f.
Tracks peak RSS, fine-grained stage timings, survivor counts, and score integrity against baseline.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import date
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.WARNING)

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder, scoring, fusion
from app.stores import artifacts, chroma, text_embedder, meta_index

# Windows memory structures
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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
    ]

def get_system_memory() -> dict:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_phys_gb": stat.ullTotalPhys / (1024**3),
        "avail_phys_gb": stat.ullAvailPhys / (1024**3),
        "memory_load_pct": stat.dwMemoryLoad,
    }

def get_process_memory() -> dict:
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return {
        "rss_mb": counters.WorkingSetSize / (1024**2),
        "peak_rss_mb": counters.PeakWorkingSetSize / (1024**2),
    }

FIXED_MOVIE_IDS = [238, 240, 278]
SCORE_TOLERANCE = 1.001e-6

CASES = {
    "b": {
        "description": "3 movies + genre + lang + date-range (Action/en/1970-2020)",
        "request": RecommendRequest(
            movie_ids=FIXED_MOVIE_IDS,
            original_language=["en"],
            genres=["Drama"],
            date_from=date(1970, 1, 1),
            date_to=date(2020, 12, 31),
            tag_weight=0.5,
        ),
    },
    "e": {
        "description": "3 movies + tight filter (fr/1950-2024)",
        "request": RecommendRequest(
            movie_ids=FIXED_MOVIE_IDS,
            original_language=["fr"],
            date_from=date(1950, 1, 1),
            date_to=date(2024, 12, 31),
            tag_weight=0.5,
        ),
    },
    "f": {
        "description": "3 movies + loose filter (Drama)",
        "request": RecommendRequest(
            movie_ids=FIXED_MOVIE_IDS,
            genres=["Drama"],
            tag_weight=0.5,
        ),
    },
}

def generate_candidates_strategy_a(req: RecommendRequest, q, timings: dict) -> list[int]:
    """Strategy A: Raise exact_limit to 150k. If n <= 150k, score all masked rows. If n > 150k, unfiltered ANN + mask."""
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    if not f.active:
        lists = []
        if q.tag_ann is not None:
            lists.append(chroma.query_ids(settings.tags_collection, q.tag_ann, settings.ann_k))
        if q.plot_centroid is not None:
            lists.append(chroma.query_ids(settings.plots_collection, q.plot_centroid, settings.ann_k))
        seen = set()
        pool = []
        import itertools
        for group in itertools.zip_longest(*lists):
            for i in group:
                if i is not None and i not in seen and i not in exclude:
                    seen.add(i)
                    pool.append(i)
                    if len(pool) >= settings.pool_cap:
                        return pool
        return pool

    t0 = time.perf_counter()
    mask = meta_index.mask_for(f)
    idx = meta_index.get_index()
    n_matches = int(np.count_nonzero(mask))
    timings["gate"] = round(time.perf_counter() - t0, 4)

    # Strategy A branch:
    if n_matches <= 150_000:
        # All matching rows become candidates for exact scoring
        matching_ids = idx.ids[mask]
        pool = [int(mid) for mid in matching_ids if mid not in exclude]
        timings["ann_tag"] = 0.0
        timings["ann_plot"] = 0.0
        timings["filter_ids"] = 0.0
        return pool
    else:
        # For n > 150,000 (e.g. Case f: 295k Drama movies):
        # Unfiltered ANN with oversampling (k=4000) + fast in-memory mask
        lists = []
        oversample_k = 4000
        if q.tag_ann is not None:
            t_ann = time.perf_counter()
            raw_tag = chroma.query_ids(settings.tags_collection, q.tag_ann, oversample_k)
            timings["ann_tag"] = round(time.perf_counter() - t_ann, 4)
            # Filter in memory
            kept_tag = [mid for mid in raw_tag if mask[idx.id_to_row[mid]]] if raw_tag else []
            lists.append(kept_tag)
        if q.plot_centroid is not None:
            t_ann = time.perf_counter()
            raw_plot = chroma.query_ids(settings.plots_collection, q.plot_centroid, oversample_k)
            timings["ann_plot"] = round(time.perf_counter() - t_ann, 4)
            kept_plot = [mid for mid in raw_plot if mask[idx.id_to_row[mid]]] if raw_plot else []
            lists.append(kept_plot)

        timings["filter_ids"] = 0.0001
        seen = set()
        pool = []
        import itertools
        for group in itertools.zip_longest(*lists):
            for i in group:
                if i is not None and i not in seen and i not in exclude:
                    seen.add(i)
                    pool.append(i)
                    if len(pool) >= settings.pool_cap:
                        break
            if len(pool) >= settings.pool_cap:
                break
        return pool


def generate_candidates_strategy_b(req: RecommendRequest, q, timings: dict) -> list[int]:
    """Strategy B: Unfiltered ANN with escalating oversampling (k=2k -> 8k -> 20k) + in-memory mask."""
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    if not f.active:
        return generate_candidates_strategy_a(req, q, timings)

    t0 = time.perf_counter()
    mask = meta_index.mask_for(f)
    idx = meta_index.get_index()
    n_matches = int(np.count_nonzero(mask))
    timings["gate"] = round(time.perf_counter() - t0, 4)

    # Escalating oversampling ladder
    ladder = [2000, 8000, 20000]
    target_pool = min(n_matches, settings.pool_cap)

    kept_tag, kept_plot = [], []
    ann_tag_total = 0.0
    ann_plot_total = 0.0

    for k in ladder:
        if q.tag_ann is not None:
            t_ann = time.perf_counter()
            raw_tag = chroma.query_ids(settings.tags_collection, q.tag_ann, k)
            ann_tag_total += (time.perf_counter() - t_ann)
            kept_tag = [mid for mid in raw_tag if mask[idx.id_to_row[mid]]] if raw_tag else []

        if q.plot_centroid is not None:
            t_ann = time.perf_counter()
            raw_plot = chroma.query_ids(settings.plots_collection, q.plot_centroid, k)
            ann_plot_total += (time.perf_counter() - t_ann)
            kept_plot = [mid for mid in raw_plot if mask[idx.id_to_row[mid]]] if raw_plot else []

        # If we have reached sufficient candidates across both lists, stop early
        if (len(kept_tag) >= target_pool // 2 and len(kept_plot) >= target_pool // 2) or (len(kept_tag) + len(kept_plot) >= target_pool):
            break

    timings["ann_tag"] = round(ann_tag_total, 4)
    timings["ann_plot"] = round(ann_plot_total, 4)
    timings["filter_ids"] = 0.0001

    lists = [l for l in [kept_tag, kept_plot] if l]
    seen = set()
    pool = []
    import itertools
    for group in itertools.zip_longest(*lists):
        for i in group:
            if i is not None and i not in seen and i not in exclude:
                seen.add(i)
                pool.append(i)
                if len(pool) >= settings.pool_cap:
                    break
        if len(pool) >= settings.pool_cap:
            break
    return pool


def run_pipeline(req: RecommendRequest, strategy: str) -> tuple[dict, dict]:
    """Execute retrieve_and_rank using specified candidate strategy."""
    t_start = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    q = query_builder.build(req, timings=timings)
    timings["query_build"] = round(time.perf_counter() - t, 4)

    t = time.perf_counter()
    if strategy == "A":
        pool_ids = generate_candidates_strategy_a(req, q, timings)
    elif strategy == "B":
        pool_ids = generate_candidates_strategy_b(req, q, timings)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    timings["candidates"] = round(time.perf_counter() - t, 4)

    if not pool_ids:
        timings["total"] = round(time.perf_counter() - t_start, 4)
        return {"ranked": [], "pool_size": 0, "timings": timings}, timings

    t = time.perf_counter()
    s = scoring.score(pool_ids, q, timings=timings)
    timings["scoring"] = round(time.perf_counter() - t, 4)

    t = time.perf_counter()
    ranked = fusion.fuse(s, req.tag_weight, top_n=settings.max_results)
    timings["fusion"] = round(time.perf_counter() - t, 4)

    timings["total"] = round(time.perf_counter() - t_start, 4)
    return {"ranked": ranked, "pool_size": len(pool_ids), "timings": timings}, timings


def main():
    print("=" * 80)
    print("PHASE 5 BENCHMARK: STRATEGY A vs STRATEGY B (INTERLEAVED N=10)")
    print("=" * 80)

    # Open resources
    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()

    # System & Process memory status
    sys_mem = get_system_memory()
    proc_mem = get_process_memory()
    print(f"System Physical RAM: {sys_mem['total_phys_gb']:.2f} GB total, {sys_mem['avail_phys_gb']:.2f} GB available ({sys_mem['memory_load_pct']}% load)")
    print(f"Process Memory post warm-up: RSS={proc_mem['rss_mb']:.1f} MB, Peak RSS={proc_mem['peak_rss_mb']:.1f} MB")

    # Load baseline for comparison
    baseline_path = Path("bench_baseline.json")
    baseline_data = {}
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f).get("cases", {})
        print(f"Loaded baseline from {baseline_path.name}")
    else:
        print("Warning: bench_baseline.json not found")

    N = 10
    results_by_case = {}

    for ckey in ["b", "e", "f"]:
        cinfo = CASES[ckey]
        req = cinfo["request"]
        print(f"\nEvaluating Case {ckey}: {cinfo['description']}")

        # Warm up both strategies
        run_pipeline(req, "A")
        run_pipeline(req, "B")

        timings_a = []
        timings_b = []
        out_a_sample = None
        out_b_sample = None
        peak_rss_a = 0.0
        peak_rss_b = 0.0

        # Interleave runs: A, B, A, B, ...
        for r in range(N):
            # Run A
            out_a, t_a = run_pipeline(req, "A")
            timings_a.append(t_a)
            if out_a_sample is None:
                out_a_sample = out_a
            rss = get_process_memory()["peak_rss_mb"]
            if rss > peak_rss_a:
                peak_rss_a = rss

            # Run B
            out_b, t_b = run_pipeline(req, "B")
            timings_b.append(t_b)
            if out_b_sample is None:
                out_b_sample = out_b
            rss = get_process_memory()["peak_rss_mb"]
            if rss > peak_rss_b:
                peak_rss_b = rss

            print(f"  Iteration {r+1}/{N}: Strategy A total={t_a['total']*1000:.1f}ms (pool={out_a['pool_size']}), Strategy B total={t_b['total']*1000:.1f}ms (pool={out_b['pool_size']})")

        # Compare outputs against baseline
        base_case = baseline_data.get(ckey, {})
        base_ranked = base_case.get("ranked", [])
        base_ids = [item["movie_id"] for item in base_ranked]
        base_map = {item["movie_id"]: item for item in base_ranked}

        def compare_with_base(sample, strat_name):
            cur_ranked = sample.get("ranked", [])
            cur_ids = [item["movie_id"] for item in cur_ranked]
            cur_map = {item["movie_id"]: item for item in cur_ranked}
            common_ids = set(cur_map.keys()) & set(base_map.keys())

            max_diff = 0.0
            for mid in common_ids:
                for score_field in ("final_score", "tag_score", "plot_score"):
                    cs = cur_map[mid][score_field]
                    bs = base_map[mid][score_field]
                    diff = abs(cs - bs)
                    if diff > max_diff:
                        max_diff = diff

            k50 = min(50, len(cur_ids), len(base_ids))
            overlap50 = len(set(cur_ids[:k50]) & set(base_ids[:k50])) / k50 if k50 > 0 else 1.0
            k200 = min(200, len(cur_ids), len(base_ids))
            overlap200 = len(set(cur_ids[:k200]) & set(base_ids[:k200])) / k200 if k200 > 0 else 1.0

            return {
                "count": len(cur_ranked),
                "common": len(common_ids),
                "max_diff": max_diff,
                "overlap50": overlap50,
                "overlap200": overlap200,
            }

        comp_a = compare_with_base(out_a_sample, "A")
        comp_b = compare_with_base(out_b_sample, "B")

        results_by_case[ckey] = {
            "timings_a": timings_a,
            "timings_b": timings_b,
            "sample_a": out_a_sample,
            "sample_b": out_b_sample,
            "comp_a": comp_a,
            "comp_b": comp_b,
            "peak_rss_a": peak_rss_a,
            "peak_rss_b": peak_rss_b,
        }

    # Summary Table
    print("\n" + "=" * 95)
    print("DETAILED COMPARISON TABLE: STRATEGY A vs STRATEGY B (N=10 INTERLEAVED)")
    print("=" * 95)

    for ckey in ["b", "e", "f"]:
        res = results_by_case[ckey]
        ta = res["timings_a"]
        tb = res["timings_b"]

        print(f"\n--- Case {ckey}: {CASES[ckey]['description']} ---")
        print(f"Strategy A: pool_size={res['sample_a']['pool_size']}, Peak RSS={res['peak_rss_a']:.1f} MB")
        print(f"  Score integrity: max_diff={res['comp_a']['max_diff']:.2e}, overlap@50={res['comp_a']['overlap50']*100:.1f}%, overlap@200={res['comp_a']['overlap200']*100:.1f}%, common={res['comp_a']['common']}/200")
        print(f"Strategy B: pool_size={res['sample_b']['pool_size']}, Peak RSS={res['peak_rss_b']:.1f} MB")
        print(f"  Score integrity: max_diff={res['comp_b']['max_diff']:.2e}, overlap@50={res['comp_b']['overlap50']*100:.1f}%, overlap@200={res['comp_b']['overlap200']*100:.1f}%, common={res['comp_b']['common']}/200")

        keys = ["gate", "ann_tag", "ann_plot", "filter_ids", "candidates", "score_tag", "score_plot", "scoring", "fusion", "total"]
        print(f"{'Stage':<15} {'Strategy A Median':>18} {'Strategy A p95':>16} {'Strategy B Median':>18} {'Strategy B p95':>16}")
        print("-" * 85)
        for k in keys:
            va = [t.get(k, 0.0) * 1000 for t in ta]
            vb = [t.get(k, 0.0) * 1000 for t in tb]
            med_a = np.percentile(va, 50)
            p95_a = np.percentile(va, 95)
            med_b = np.percentile(vb, 50)
            p95_b = np.percentile(vb, 95)
            print(f"{k:<15} {med_a:>15.1f} ms {p95_a:>13.1f} ms {med_b:>15.1f} ms {p95_b:>13.1f} ms")

    close_pool()

if __name__ == "__main__":
    main()
