import asyncio
import os
from pathlib import Path
import sys
import time
import httpx
import numpy as np

# Set single-threaded BLAS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
import app.security as security
from scripts.eval_concurrency_investigation import MemorySampler

async def test_distinct():
    print("=" * 80)
    print("BURST OF 8 DISTINCT (CACHE-MISSING) REQUESTS ACROSS PIPELINE_SEM 1, 2, 4")
    print("=" * 80)

    async with app.router.lifespan_context(app):
        app.state.limiter.enabled = False

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
            for sem_val in [1, 2, 4]:
                security.PIPELINE_SEM = asyncio.Semaphore(sem_val)

                # 8 distinct requests with unique tag_weights so paging cache is bypassed
                distinct_8 = [
                    {"movie_ids": [238, 240, 278], "tag_weight": 0.11 + sem_val*0.01 + i*0.02} for i in range(2)
                ] + [
                    {"movie_ids": [238, 240, 278], "original_language": ["en"], "genres": ["Drama"], "tag_weight": 0.21 + sem_val*0.01 + i*0.02} for i in range(2)
                ] + [
                    {"movie_ids": [238, 240, 278], "original_language": ["fr"], "tag_weight": 0.31 + sem_val*0.01 + i*0.02} for i in range(2)
                ] + [
                    {"movie_ids": [238, 240, 278], "genres": ["Drama"], "tag_weight": 0.41 + sem_val*0.01 + i*0.02} for i in range(2)
                ]

                sampler = MemorySampler()
                sampler.start()
                t0 = time.perf_counter()

                tasks = [client.post("/api/recommend", json=payload) for payload in distinct_8]
                res = await asyncio.gather(*tasks)

                t_wall = time.perf_counter() - t0
                await sampler.stop()

                codes = [r.status_code for r in res]
                lats = [r.elapsed.total_seconds() * 1000.0 for r in res]
                n_200 = codes.count(200)
                n_503 = codes.count(503)

                print(f"SEM={sem_val}: 200 OK={n_200}/8, 503={n_503}/8, Wall={t_wall:.2f}s, Throughput={n_200/t_wall:.2f}/s, p50={np.percentile(lats, 50):.1f}ms, p95={np.percentile(lats, 95):.1f}ms, Max={np.max(lats):.1f}ms, Peak RSS={sampler.peak_mb:.1f}MB (delta: {sampler.peak_mb - sampler.start_mb:+.1f}MB)")

if __name__ == "__main__":
    asyncio.run(test_distinct())
