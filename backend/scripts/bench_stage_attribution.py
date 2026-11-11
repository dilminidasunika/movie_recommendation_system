import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import ctypes
from ctypes import wintypes
import time
import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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

psapi = ctypes.windll.psapi
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

kernel32 = ctypes.windll.kernel32

def get_sys_mem():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / 1024**2  # MB

def get_proc_mem():
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 1024**2, pmc.PageFaultCount

from app.db.pool import open_pool, close_pool
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion
from app.stores import artifacts, meta_index

def run_attribution(n_distinct=20, rng_seed=42):
    print("=" * 70, flush=True)
    print("ITEM 4: PER-STAGE PAGE-FAULT & WORKING-SET ATTRIBUTION", flush=True)
    print("=" * 70, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    common_ids = np.array(list(set(id_to_row.keys()).intersection(set(plot_id_to_r.keys()))))
    common_ids.sort()

    rng = np.random.default_rng(rng_seed)

    from datetime import date
    cases = {
        "case_e_rare_lang": {
            "original_language": ["fr"],
            "genres": [],
            "date_from": date(1950, 1, 1),
            "date_to": date(2024, 12, 31),
        },
        "case_b_filter_55k": {
            "original_language": ["en"],
            "genres": ["Drama"],
            "date_from": date(1970, 1, 1),
            "date_to": date(2020, 12, 31),
        }
    }

    # Run for both cases
    for case_name, filter_params in cases.items():
        print(f"\n--- Profiling Case: {case_name} (20 distinct requests) ---", flush=True)
        stage_times = {"build_query": [], "generate_candidates": [], "score_tag": [], "score_plot": [], "fuse": []}
        stage_pfs = {"build_query": [], "generate_candidates": [], "score_tag": [], "score_plot": [], "fuse": []}
        stage_ws = {"start": [], "after_query": [], "after_cands": [], "after_score_tag": [], "after_score_plot": [], "after_fuse": []}
        match_counts = []

        for req_idx in range(n_distinct):
            seeds = rng.choice(common_ids, size=3, replace=False).tolist()
            req = RecommendRequest(
                movie_ids=seeds,
                original_language=filter_params["original_language"],
                genres=filter_params["genres"],
                date_from=filter_params["date_from"],
                date_to=filter_params["date_to"],
                tag_weight=0.5,
            )

            ws0, pf0 = get_proc_mem()
            stage_ws["start"].append(ws0)

            # Stage 1: build_query
            t0 = time.perf_counter()
            q = query_builder.build(req)
            t_query = (time.perf_counter() - t0) * 1000.0
            ws1, pf1 = get_proc_mem()
            stage_times["build_query"].append(t_query)
            stage_pfs["build_query"].append(pf1 - pf0)
            stage_ws["after_query"].append(ws1)

            # Stage 2: generate candidates
            t0 = time.perf_counter()
            pool_ids = candidates.generate(req, q)
            t_cands = (time.perf_counter() - t0) * 1000.0
            ws2, pf2 = get_proc_mem()
            stage_times["generate_candidates"].append(t_cands)
            stage_pfs["generate_candidates"].append(pf2 - pf1)
            stage_ws["after_cands"].append(ws2)
            match_counts.append(len(pool_ids))

            # Stage 3: score (instrumenting score_tag and score_plot separately)
            # To measure tag vs plot scoring separately:
            t0 = time.perf_counter()
            # Run score_tag
            pos = [k for k, mid in enumerate(pool_ids) if mid in id_to_row]
            rows = [id_to_row[pool_ids[k]] for k in pos]
            dots = (artifacts.tag_matrix()[rows] @ q.tag_sparse.T).toarray().ravel()
            norms = artifacts.row_norms()[rows]
            valid = norms > 0
            tag_scores = np.zeros(len(rows))
            tag_scores[valid] = dots[valid] / norms[valid]
            t_score_tag = (time.perf_counter() - t0) * 1000.0
            ws3, pf3 = get_proc_mem()
            stage_times["score_tag"].append(t_score_tag)
            stage_pfs["score_tag"].append(pf3 - pf2)
            stage_ws["after_score_tag"].append(ws3)

            # Run score_plot
            t0 = time.perf_counter()
            c = np.asarray(q.plot_centroid, dtype=np.float32)
            c = c / np.linalg.norm(c)
            valid_items = [(plot_id_to_r[mid], k) for k, mid in enumerate(pool_ids) if mid in plot_id_to_r]
            valid_items.sort(key=lambda x: x[0])
            p_rows = np.array([x[0] for x in valid_items], dtype=np.int64)
            orig_pos = np.array([x[1] for x in valid_items], dtype=np.int64)
            chunk_size = 8192
            plot_scores = np.zeros(len(valid_items), dtype=np.float32)
            for ch_start in range(0, len(p_rows), chunk_size):
                ch_end = min(ch_start + chunk_size, len(p_rows))
                ch_rows = p_rows[ch_start:ch_end]
                plot_scores[ch_start:ch_end] = plot_mem[ch_rows] @ c
            t_score_plot = (time.perf_counter() - t0) * 1000.0
            ws4, pf4 = get_proc_mem()
            stage_times["score_plot"].append(t_score_plot)
            stage_pfs["score_plot"].append(pf4 - pf3)
            stage_ws["after_score_plot"].append(ws4)

            # Stage 4: fuse
            t0 = time.perf_counter()
            s = scoring.Scores(
                ids=pool_ids,
                tag=np.zeros(len(pool_ids)),
                plot=np.zeros(len(pool_ids)),
                tag_ok=np.ones(len(pool_ids), bool),
                plot_ok=np.ones(len(pool_ids), bool),
            )
            ranked = fusion.fuse(s, req.tag_weight, top_n=200)
            t_fuse = (time.perf_counter() - t0) * 1000.0
            ws5, pf5 = get_proc_mem()
            stage_times["fuse"].append(t_fuse)
            stage_pfs["fuse"].append(pf5 - pf4)
            stage_ws["after_fuse"].append(ws5)

        print(f"Results for {case_name} (Avg pool size: {np.mean(match_counts):,.0f} rows):", flush=True)
        print(f"{'Stage':<25} {'Time (ms) Mean':<16} {'Time p95':<12} {'PageFaults Mean':<16} {'WorkingSet After (MB)':<22}")
        print("-" * 90)
        print(f"{'1. build_query':<25} {np.mean(stage_times['build_query']):<16.2f} {np.percentile(stage_times['build_query'], 95):<12.2f} {np.mean(stage_pfs['build_query']):<16.1f} {np.mean(stage_ws['after_query']):<22.2f}")
        print(f"{'2. generate (candidates)':<25} {np.mean(stage_times['generate_candidates']):<16.2f} {np.percentile(stage_times['generate_candidates'], 95):<12.2f} {np.mean(stage_pfs['generate_candidates']):<16.1f} {np.mean(stage_ws['after_cands']):<22.2f}")
        print(f"{'3. score_tag (CSR)':<25} {np.mean(stage_times['score_tag']):<16.2f} {np.percentile(stage_times['score_tag'], 95):<12.2f} {np.mean(stage_pfs['score_tag']):<16.1f} {np.mean(stage_ws['after_score_tag']):<22.2f}")
        print(f"{'4. score_plot (memmap)':<25} {np.mean(stage_times['score_plot']):<16.2f} {np.percentile(stage_times['score_plot'], 95):<12.2f} {np.mean(stage_pfs['score_plot']):<16.1f} {np.mean(stage_ws['after_score_plot']):<22.2f}")
        print(f"{'5. fuse':<25} {np.mean(stage_times['fuse']):<16.2f} {np.percentile(stage_times['fuse'], 95):<12.2f} {np.mean(stage_pfs['fuse']):<16.1f} {np.mean(stage_ws['after_fuse']):<22.2f}")
        total_time_mean = sum(np.mean(stage_times[k]) for k in stage_times)
        total_pf_mean = sum(np.mean(stage_pfs[k]) for k in stage_pfs)
        print("-" * 90)
        print(f"{'TOTAL PIPELINE':<25} {total_time_mean:<16.2f} {'-':<12} {total_pf_mean:<16.1f} {np.mean(stage_ws['after_fuse']):<22.2f}")

    print("\nSystem Available Physical Memory at end: {:.2f} MB".format(get_sys_mem()))
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_attribution(n_distinct=20, rng_seed=42)
