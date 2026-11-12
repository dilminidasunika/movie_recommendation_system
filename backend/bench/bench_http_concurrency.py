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
from app.config import settings

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

psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def get_peak_rss_mb() -> float:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    handle = kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / (1024**2)

REQUESTS = {
    "a": {"movie_ids": [238, 240, 278], "tag_weight": 0.5},
    "b": {
        "movie_ids": [238, 240, 278],
        "original_language": ["en"],
        "genres": ["Drama"],
        "date_from": "1970-01-01",
        "date_to": "2020-12-31",
        "tag_weight": 0.5,
    },
    "e": {
        "movie_ids": [238, 240, 278],
        "original_language": ["fr"],
        "date_from": "1950-01-01",
        "date_to": "2024-12-31",
        "tag_weight": 0.5,
    },
    "f": {
        "movie_ids": [238, 240, 278],
        "genres": ["Drama"],
        "tag_weight": 0.5,
    },
}

async def send_req(client: httpx.AsyncClient, case_name: str, payload: dict) -> dict:
    t0 = time.perf_counter()
    resp = await client.post("/api/recommend", json=payload)
    elapsed = time.perf_counter() - t0
    return {
        "case": case_name,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed * 1000.0,
        "success": resp.status_code == 200,
        "body_len": len(resp.content),
    }

async def run_http_concurrency_test():
    print("=" * 75)
    print("HTTP CONCURRENCY TEST: 20 CONCURRENT REQUESTS THROUGH PIPELINE_SEM")
    print("=" * 75)

    async with app.router.lifespan_context(app):
        # Disable rate limiting for the load test so requests test PIPELINE_SEM capacity
        app.state.limiter.enabled = False

        # 20 requests: 5 of each case (a, b, e, f) interleaved
        tasks_to_run = []
        cases_order = ["a", "b", "e", "f"] * 5

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
            print("Sending warm-up HTTP request...")
            warm_resp = await client.post("/api/recommend", json=REQUESTS["a"])
            print(f"Warm-up finished with status {warm_resp.status_code}")

            rss_before = get_peak_rss_mb()
            t_batch_start = time.perf_counter()

            tasks = [send_req(client, c, REQUESTS[c]) for c in cases_order]
            results = await asyncio.gather(*tasks)

            t_batch_total = time.perf_counter() - t_batch_start
            peak_rss = get_peak_rss_mb()

    # Analyze results
    latencies = [r["elapsed_ms"] for r in results]
    errors = [r for r in results if not r["success"]]

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    max_lat = np.max(latencies)

    print(f"\nCompleted {len(results)} concurrent requests in {t_batch_total:.2f}s total wall-clock time.")
    print(f"Total Errors / Non-200: {len(errors)}")
    print(f"Latency p50: {p50:.1f} ms")
    print(f"Latency p95: {p95:.1f} ms")
    print(f"Latency Max: {max_lat:.1f} ms")
    print(f"Peak RSS during run: {peak_rss:.1f} MB (delta: {peak_rss - rss_before:+.1f} MB)")

    print("\nPer-Case Breakdown:")
    for c in ["a", "b", "e", "f"]:
        c_lats = [r["elapsed_ms"] for r in results if r["case"] == c]
        print(f"  Case {c} (N=5): p50={np.percentile(c_lats, 50):.1f} ms, p95={np.percentile(c_lats, 95):.1f} ms, max={np.max(c_lats):.1f} ms")

if __name__ == "__main__":
    asyncio.run(run_http_concurrency_test())
