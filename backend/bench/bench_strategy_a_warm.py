import os
from pathlib import Path
import sys
import time
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import recommender
from app.stores import artifacts, meta_index, chroma, text_embedder
from datetime import date

def run_30_warm_runs():
    print("=" * 80)
    print("STRATEGY A: 30 WARM RUNS (PLOT MEMMAP RESIDENT IN RAM)")
    print("=" * 80)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()

    # Pre-fault memmap explicitly
    emb, _ = artifacts.plot_memmap()
    t_fault = time.perf_counter()
    _ = np.sum(emb[:, 0])
    print(f"Plot memmap pre-faulted into OS page cache in {time.perf_counter() - t_fault:.2f}s")

    cases = {
        "b": RecommendRequest(
            movie_ids=[238, 240, 278],
            original_language=["en"],
            genres=["Drama"],
            date_from=date(1970, 1, 1),
            date_to=date(2020, 12, 31),
            tag_weight=0.5,
        ),
        "e": RecommendRequest(
            movie_ids=[238, 240, 278],
            original_language=["fr"],
            date_from=date(1950, 1, 1),
            date_to=date(2024, 12, 31),
            tag_weight=0.5,
        ),
    }

    N = 30
    results = {}

    for ckey, req in cases.items():
        print(f"\nBenchmarking Case {ckey} (30 warm runs)...")
        # Discard 2 warm-up runs
        recommender.retrieve_and_rank(req)
        recommender.retrieve_and_rank(req)

        run_timings = []
        pool_size = 0

        for r in range(N):
            out = recommender.retrieve_and_rank(req)
            if pool_size == 0:
                pool_size = out["pool_size"]
            run_timings.append(out["timings"])
            print(".", end="", flush=True)
        print(" done.")

        keys = ["tag_part", "plot_centroid", "query_build", "gate", "candidates", "score_tag", "score_plot", "scoring", "fusion", "total"]
        stats = {}
        for k in keys:
            vals = [t.get(k, 0.0) * 1000.0 for t in run_timings]
            stats[k] = {
                "p50": np.percentile(vals, 50),
                "p95": np.percentile(vals, 95),
                "max": np.max(vals),
            }
        results[ckey] = {"pool_size": pool_size, "stats": stats}

    print("\n" + "=" * 80)
    print("RESULTS: STRATEGY A WARM LATENCY (N=30 RUNS)")
    print("=" * 80)

    for ckey in ["b", "e"]:
        res = results[ckey]
        st = res["stats"]
        print(f"\nCase {ckey} (Pool size = {res['pool_size']}):")
        print(f"{'Stage':<18} {'p50 (ms)':>12} {'p95 (ms)':>12} {'Max (ms)':>12}")
        print("-" * 58)
        for k in ["gate", "candidates", "score_tag", "score_plot", "scoring", "fusion", "total"]:
            print(f"{k:<18} {st[k]['p50']:>12.1f} {st[k]['p95']:>12.1f} {st[k]['max']:>12.1f}")

    close_pool()

if __name__ == "__main__":
    run_30_warm_runs()
