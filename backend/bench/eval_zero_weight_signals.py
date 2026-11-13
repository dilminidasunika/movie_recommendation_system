import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion
from app.stores import artifacts, chroma, meta_index


def run_eval(n_queries: int = 50):
    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)

    common_ids = list(set(artifacts.matrix_ids()) & set(artifacts.plot_ids()))
    rng = np.random.default_rng(42)
    sample_queries = [rng.choice(common_ids, size=3, replace=False).tolist() for _ in range(n_queries)]

    print(f"Evaluating {n_queries} queries at tag_weight = 0.0 and tag_weight = 1.0...")

    results = {}
    for tw in [0.0, 1.0]:
        recalls = []
        latencies_before = []
        latencies_after = []

        for seeds in sample_queries:
            req = RecommendRequest(movie_ids=seeds, tag_weight=tw)
            q = query_builder.build(req)

            # Ground Truth (BEFORE): Both signals retrieved
            t0 = time.perf_counter()
            raw_tag_gt, _ = candidates._matvec_tag(q, k=2000, exclude=set(seeds))
            raw_plot_gt = np.asarray(chroma.query_ids(settings.plots_collection, q.plot_centroid, 2000), dtype=np.int64)
            pool_gt = candidates._unique_ordered(raw_tag_gt, raw_plot_gt)
            s_gt = scoring.score(pool_gt, q)
            ranked_gt = fusion.fuse(s_gt, tw, top_n=50)
            dt_before = (time.perf_counter() - t0) * 1000
            latencies_before.append(dt_before)

            # Optimized (AFTER): Zero-weight signal skipped during retrieval
            t0 = time.perf_counter()
            pool_opt = candidates.generate(req, q)
            s_opt = scoring.score(pool_opt, q)
            ranked_opt = fusion.fuse(s_opt, tw, top_n=50)
            dt_after = (time.perf_counter() - t0) * 1000
            latencies_after.append(dt_after)

            gt_ids = [r["movie_id"] for r in ranked_gt]
            opt_ids = [r["movie_id"] for r in ranked_opt]

            if len(gt_ids) == 0:
                rec = 1.0
            else:
                rec = len(set(opt_ids) & set(gt_ids)) / len(gt_ids)
            recalls.append(rec)

        mean_rec = np.mean(recalls) * 100
        p50_before = np.median(latencies_before)
        p50_after = np.median(latencies_after)
        speedup = p50_before / max(0.001, p50_after)
        results[tw] = {
            "recall": mean_rec,
            "latency_p50_before_ms": round(p50_before, 2),
            "latency_p50_after_ms": round(p50_after, 2),
            "speedup": round(speedup, 2),
        }
        print(f"tag_weight = {tw:.1f} | Recall vs GT: {mean_rec:.2f}% | Latency p50: Before={p50_before:.1f}ms, After={p50_after:.1f}ms ({speedup:.2f}x faster)")

    close_pool()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    q_count = 2 if args.smoke else args.queries
    run_eval(q_count)
