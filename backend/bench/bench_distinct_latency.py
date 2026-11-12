#!/usr/bin/env python3
"""
Distinct-Request Benchmark for Movie Recommendation System.
Measures realistic latency across 6 case types x N requests with distinct random seed triples.
Disables paging cache, runs sequentially at PIPELINE_SEM=1.
Tracks Win32 page faults, disk read volume, and memory counters.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import asyncio
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import time

import httpx
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
import app.security as security
from app.services import paging
from app.stores import artifacts

# ── Win32 structures ──────────────────────────────────────────────────────────

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

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
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

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def _p(*args, **kwargs):
    """Print with flush."""
    print(*args, **kwargs, flush=True)

def get_system_counters():
    handle = kernel32.GetCurrentProcess()
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)

    ioc = IO_COUNTERS()
    kernel32.GetProcessIoCounters(handle, ctypes.byref(ioc))

    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))

    return {
        "page_faults": pmc.PageFaultCount,
        "rss_mb": pmc.WorkingSetSize / (1024 * 1024),
        "peak_rss_mb": pmc.PeakWorkingSetSize / (1024 * 1024),
        "read_bytes_mb": ioc.ReadTransferCount / (1024 * 1024),
        "read_ops": ioc.ReadOperationCount,
        "free_ram_mb": mem.ullAvailPhys / (1024 * 1024),
    }

def prefault_all_pages():
    t0 = time.perf_counter()
    emb, _ = artifacts.plot_memmap()
    m = artifacts.tag_matrix()
    chunk = 65536
    s = 0.0
    for i in range(0, emb.shape[0], chunk):
        s += float(np.sum(emb[i:i+chunk, :]))
    s += float(m.data.sum()) + float(m.indices.sum()) + float(m.indptr.sum())
    elapsed = time.perf_counter() - t0
    _p(f"[Prefault] Touched all memmap & tag matrix pages in {elapsed*1000:.1f}ms")
    return elapsed

def generate_request_suite(n_per_case: int = 20):
    rng = np.random.default_rng(42)
    all_ids = np.array(list(artifacts.id_to_row().keys()))

    cases = {}

    # 1. Unfiltered
    cases["unfiltered"] = []
    for _ in range(n_per_case):
        mids = rng.choice(all_ids, size=3, replace=False).tolist()
        cases["unfiltered"].append({"movie_ids": mids, "tag_weight": 0.5})

    # 2. Plot text
    cases["plot_text"] = []
    for _ in range(n_per_case):
        mids = rng.choice(all_ids, size=3, replace=False).tolist()
        cases["plot_text"].append({
            "movie_ids": mids,
            "plot_text": "A dramatic story of crime, family loyalty, and redemption.",
            "tag_weight": 0.5,
        })

    # 3. Rare-language filter (French + date range, Case e: ~57k matches, exact path)
    cases["rare_lang"] = []
    for _ in range(n_per_case):
        mids = rng.choice(all_ids, size=3, replace=False).tolist()
        cases["rare_lang"].append({
            "movie_ids": mids,
            "original_language": ["fr"],
            "date_from": "1950-01-01",
            "date_to": "2024-12-31",
            "tag_weight": 0.5,
        })

    # 4. 55k-match filter (English + Drama + date range, Case b: ~55k matches, exact path)
    cases["filter_55k"] = []
    for _ in range(n_per_case):
        mids = rng.choice(all_ids, size=3, replace=False).tolist()
        cases["filter_55k"].append({
            "movie_ids": mids,
            "original_language": ["en"],
            "genres": ["Drama"],
            "date_from": "1970-01-01",
            "date_to": "2020-12-31",
            "tag_weight": 0.5,
        })

    # 5. 295k-match filter (Drama, Case f: 295k matches, ANN ladder path)
    cases["filter_295k"] = []
    for _ in range(n_per_case):
        mids = rng.choice(all_ids, size=3, replace=False).tolist()
        cases["filter_295k"].append({
            "movie_ids": mids,
            "genres": ["Drama"],
            "tag_weight": 0.5,
        })

    # 6. Filters-only (Animation + tag chip, Case c: small branch)
    cases["filters_only"] = []
    for _ in range(n_per_case):
        cases["filters_only"].append({
            "tags": [{"tag_type": "genres", "name": "Animation"}],
            "original_language": ["ja"],
            "genres": ["Animation"],
            "tag_weight": 1.0,
        })

    return cases

async def run_benchmark(mode_name: str, do_prefault: bool = False, n_per_case: int = 20):
    _p("=" * 100)
    _p(f"DISTINCT-REQUEST BENCHMARK: {mode_name}")
    _p(f"Sequential @ PIPELINE_SEM=1 | Paging cache cleared | {n_per_case} distinct requests / case")
    _p("=" * 100)

    if do_prefault:
        prefault_all_pages()

    suite = generate_request_suite(n_per_case)

    app.state.limiter.enabled = False
    security.PIPELINE_SEM = asyncio.Semaphore(1)

    initial = get_system_counters()
    _p(f"Initial: Free RAM={initial['free_ram_mb']:.0f}MB  RSS={initial['rss_mb']:.0f}MB  "
       f"PageFaults={initial['page_faults']}  DiskRead={initial['read_bytes_mb']:.0f}MB")
    _p("-" * 100)

    results = {}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=120.0) as client:
        for case_name, reqs in suite.items():
            case_t0 = time.perf_counter()
            c0 = get_system_counters()
            lats = {"http": [], "pipe": [], "qb": [], "cand": [], "score": [], "fuse": [], "pool": []}
            errors = 0

            _p(f"  {case_name:<14} [{n_per_case} reqs]: ", end="")

            for i, payload in enumerate(reqs):
                with paging._lock:
                    paging._cache.clear()

                t0 = time.perf_counter()
                res = await client.post("/api/recommend", json=payload)
                dt = (time.perf_counter() - t0) * 1000.0

                if res.status_code != 200:
                    errors += 1
                    _p("X", end="")
                    continue

                _p(".", end="")
                data = res.json()
                t = data.get("timings", {})
                lats["http"].append(dt)
                lats["pipe"].append(t.get("total", 0) * 1000)
                lats["qb"].append(t.get("query_build", 0) * 1000)
                lats["cand"].append(t.get("candidates", 0) * 1000)
                lats["score"].append(t.get("scoring", 0) * 1000)
                lats["fuse"].append(t.get("fusion", 0) * 1000)
                lats["pool"].append(data.get("pool_size", 0))

            c1 = get_system_counters()
            case_wall = time.perf_counter() - case_t0
            n = len(lats["http"])

            if n < 2:
                _p(f" SKIP (only {n} OK, {errors} errors)")
                continue

            s = {
                "n": n, "errors": errors,
                "http_p50": np.percentile(lats["http"], 50),
                "http_p95": np.percentile(lats["http"], 95),
                "http_max": np.max(lats["http"]),
                "pipe_p50": np.percentile(lats["pipe"], 50),
                "pipe_p95": np.percentile(lats["pipe"], 95),
                "cand_p50": np.percentile(lats["cand"], 50),
                "cand_p95": np.percentile(lats["cand"], 95),
                "score_p50": np.percentile(lats["score"], 50),
                "score_p95": np.percentile(lats["score"], 95),
                "fuse_p50": np.percentile(lats["fuse"], 50),
                "pool_p50": int(np.median(lats["pool"])),
                "d_faults": c1["page_faults"] - c0["page_faults"],
                "d_read_mb": c1["read_bytes_mb"] - c0["read_bytes_mb"],
                "rss_mb": c1["rss_mb"],
                "free_mb": c1["free_ram_mb"],
                "wall_s": case_wall,
            }
            results[case_name] = s

            _p(f"\n    HTTP p50={s['http_p50']:.0f}ms p95={s['http_p95']:.0f}ms max={s['http_max']:.0f}ms | "
               f"Pipe p50={s['pipe_p50']:.0f}ms p95={s['pipe_p95']:.0f}ms | "
               f"Cand p50={s['cand_p50']:.0f}ms Score p50={s['score_p50']:.0f}ms | "
               f"Faults={s['d_faults']:+d} Disk={s['d_read_mb']:.0f}MB RSS={s['rss_mb']:.0f}MB Free={s['free_mb']:.0f}MB "
               f"Wall={s['wall_s']:.1f}s")

    final = get_system_counters()
    _p("-" * 100)
    _p(f"TOTAL: Faults={final['page_faults'] - initial['page_faults']:+d}  "
       f"DiskRead={final['read_bytes_mb'] - initial['read_bytes_mb']:.0f}MB  "
       f"FreeRAM={final['free_ram_mb']:.0f}MB  PeakRSS={final['peak_rss_mb']:.0f}MB")
    _p("=" * 100)
    return results

async def main():
    async with app.router.lifespan_context(app):
        _p("\n" + "#" * 100)
        _p("RUN (a): AS-IS (normal boot warm-up only)")
        _p("#" * 100)
        res_a = await run_benchmark("Run (a) - As-Is", do_prefault=False, n_per_case=20)

        _p("\n" + "#" * 100)
        _p("RUN (b): IMMEDIATELY AFTER PRE-FAULTING MEMMAP + TAG MATRIX")
        _p("#" * 100)
        res_b = await run_benchmark("Run (b) - Pre-faulted", do_prefault=True, n_per_case=20)

if __name__ == "__main__":
    asyncio.run(main())
