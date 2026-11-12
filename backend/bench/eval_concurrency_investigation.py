import asyncio
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import time

# Set single-threaded BLAS to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx
import numpy as np

from app.main import app
import app.security as security

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

def get_current_rss_mb() -> float:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    handle = kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / (1024**2)

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

class MemorySampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.samples = []
        self._stop = False
        self._task = None

    async def _sample_loop(self):
        while not self._stop:
            self.samples.append(get_current_rss_mb())
            await asyncio.sleep(self.interval_s)

    def start(self):
        self.samples = [get_current_rss_mb()]
        self._stop = False
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self):
        self._stop = True
        if self._task:
            await self._task
            self.samples.append(get_current_rss_mb())

    @property
    def peak_mb(self) -> float:
        return max(self.samples) if self.samples else get_current_rss_mb()

    @property
    def start_mb(self) -> float:
        return self.samples[0] if self.samples else get_current_rss_mb()


async def send_req(client: httpx.AsyncClient, case_name: str, payload: dict) -> dict:
    t0 = time.perf_counter()
    resp = await client.post("/api/recommend", json=payload)
    elapsed = time.perf_counter() - t0
    return {
        "case": case_name,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed * 1000.0,
        "success": resp.status_code == 200,
    }

async def run_burst(client: httpx.AsyncClient, cases_list: list[str]) -> tuple[dict, float, float]:
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()

    tasks = [send_req(client, c, REQUESTS[c]) for c in cases_list]
    results = await asyncio.gather(*tasks)

    t_wall = time.perf_counter() - t0
    await sampler.stop()

    latencies = [r["elapsed_ms"] for r in results]
    success_lats = [r["elapsed_ms"] for r in results if r["success"]]
    codes = [r["status_code"] for r in results]
    n_200 = codes.count(200)
    n_503 = codes.count(503)

    stats = {
        "total": len(results),
        "n_200": n_200,
        "n_503": n_503,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "max": np.max(latencies),
        "success_p50": np.percentile(success_lats, 50) if success_lats else 0.0,
        "wall_s": t_wall,
        "throughput": n_200 / t_wall,
        "start_rss_mb": sampler.start_mb,
        "peak_rss_mb": sampler.peak_mb,
        "delta_rss_mb": sampler.peak_mb - sampler.start_mb,
    }
    return stats


async def run_steady_load(client: httpx.AsyncClient, sem_val: int, duration_s: float = 60.0):
    print(f"\n--- Running steady load test (4 at a time for {duration_s:.0f}s) with PIPELINE_SEM={sem_val} ---")
    sampler = MemorySampler(interval_s=0.2)
    sampler.start()

    t_end = time.perf_counter() + duration_s
    t_start = time.perf_counter()
    all_results = []
    cases_cycle = ["a", "b", "e", "f"]
    c_idx = 0

    while time.perf_counter() < t_end:
        batch = [cases_cycle[(c_idx + i) % 4] for i in range(4)]
        c_idx += 4
        tasks = [send_req(client, c, REQUESTS[c]) for c in batch]
        res = await asyncio.gather(*tasks)
        all_results.extend(res)

    t_wall = time.perf_counter() - t_start
    await sampler.stop()

    codes = [r["status_code"] for r in all_results]
    lats = [r["elapsed_ms"] for r in all_results]
    n_200 = codes.count(200)
    n_503 = codes.count(503)

    print(f"Completed {len(all_results)} requests in {t_wall:.1f}s: 200 OK = {n_200}, 503 Busy = {n_503}")
    print(f"Throughput: {n_200 / t_wall:.2f} req/s")
    print(f"Latency: p50={np.percentile(lats, 50):.1f}ms, p95={np.percentile(lats, 95):.1f}ms, max={np.max(lats):.1f}ms")
    print(f"Memory: Start RSS={sampler.start_mb:.1f}MB, Peak RSS={sampler.peak_mb:.1f}MB (delta: {sampler.peak_mb - sampler.start_mb:+.1f}MB)")


async def main():
    print("=" * 80)
    print("HTTP CONCURRENCY & PIPELINE_SEM INVESTIGATION")
    print("=" * 80)
    print(f"CPU count: {os.cpu_count()} logical cores")
    print(f"Environment: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")

    async with app.router.lifespan_context(app):
        app.state.limiter.enabled = False

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
            # Warm-up
            _ = await client.post("/api/recommend", json=REQUESTS["a"])
            _ = await client.post("/api/recommend", json=REQUESTS["b"])

            burst_8 = ["a", "b", "e", "f"] * 2
            burst_20 = ["a", "b", "e", "f"] * 5

            print("\n" + "=" * 80)
            print("EXPERIMENT 1: BURST OF 8 MIXED REQUESTS (ZERO 503 TARGET)")
            print("=" * 80)
            print(f"{'SEM':<6} {'200 OK':>8} {'503':>6} {'p50 (ms)':>10} {'p95 (ms)':>10} {'Max (ms)':>10} {'Wall (s)':>10} {'Throughput':>12} {'Peak RSS':>10}")
            print("-" * 88)

            for sem_val in [1, 2, 4]:
                security.PIPELINE_SEM = asyncio.Semaphore(sem_val)
                stats = await run_burst(client, burst_8)
                print(f"{sem_val:<6} {stats['n_200']:>8} {stats['n_503']:>6} {stats['p50']:>10.1f} {stats['p95']:>10.1f} {stats['max']:>10.1f} {stats['wall_s']:>10.2f} {stats['throughput']:>10.2f}/s {stats['peak_rss_mb']:>9.1f}MB")

            print("\n" + "=" * 80)
            print("EXPERIMENT 2: BURST OF 20 MIXED REQUESTS")
            print("=" * 80)
            print(f"{'SEM':<6} {'200 OK':>8} {'503':>6} {'p50 (ms)':>10} {'p95 (ms)':>10} {'Max (ms)':>10} {'Wall (s)':>10} {'Throughput':>12} {'Peak RSS':>10}")
            print("-" * 88)

            for sem_val in [1, 2, 4]:
                security.PIPELINE_SEM = asyncio.Semaphore(sem_val)
                stats = await run_burst(client, burst_20)
                print(f"{sem_val:<6} {stats['n_200']:>8} {stats['n_503']:>6} {stats['p50']:>10.1f} {stats['p95']:>10.1f} {stats['max']:>10.1f} {stats['wall_s']:>10.2f} {stats['throughput']:>10.2f}/s {stats['peak_rss_mb']:>9.1f}MB")

            # Steady load test with the best semaphore setting
            security.PIPELINE_SEM = asyncio.Semaphore(2)
            await run_steady_load(client, sem_val=2, duration_s=30.0)

if __name__ == "__main__":
    asyncio.run(main())
