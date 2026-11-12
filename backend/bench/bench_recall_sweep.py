#!/usr/bin/env python3
"""
Recall vs Candidate Pool Sweep (Unconfounded).
Candidate pool = deduplicated UNION of tag and plot ANN results (no pool_cap truncation).
Sweeps k in {2000, 5000, 10000, 20000} x tag_weight in {0.0, 0.3, 0.5, 0.8, 1.0}.
Includes old behavior (k=20000 with pool_cap=2000) to confirm cap limited recall.
Reports recall@50/@200 vs exact ground truth and median/p95 latency (distinct requests).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import logging
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("posthog").setLevel(logging.WARNING)

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion, recommender
from app.stores import artifacts, chroma, text_embedder, meta_index

def _p(*a, **kw):
    print(*a, **kw, flush=True)

# ── Cases ──────────────────────────────────────────────────────────────────────

def get_sweep_cases():
    from datetime import date
    return {
        "a": RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.5),
        "d": RecommendRequest(
            movie_ids=[238, 240, 278],
            plot_text="A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption across generations.",
            tag_weight=0.5,
        ),
        "f": RecommendRequest(movie_ids=[238, 240, 278], genres=["Drama"], tag_weight=0.5),
    }


# ── Ground truth ──────────────────────────────────────────────────────────────

def compute_ground_truth(case_key, req, q):
    """Brute-force exact scoring over the full universe / masked set."""
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    if f.active and meta_index.is_loaded():
        mask = meta_index.mask_for(f)
        idx = meta_index.get_index()
        full_pool = [int(mid) for mid in idx.ids[mask] if mid not in exclude]
    else:
        # Unfiltered: score all movies in tag matrix
        all_ids = list(artifacts.id_to_row().keys())
        full_pool = [mid for mid in all_ids if mid not in exclude]

    t0 = time.perf_counter()
    s = scoring.score(full_pool, q)
    ranked = fusion.fuse(s, req.tag_weight, top_n=200)
    t_gt = time.perf_counter() - t0

    gt50 = set(r["movie_id"] for r in ranked[:50])
    gt200 = set(r["movie_id"] for r in ranked[:200])
    _p(f"  GT for case {case_key}: scored {len(full_pool)} movies in {t_gt:.2f}s")
    return gt50, gt200, len(full_pool), t_gt


# ── Sweep logic ──────────────────────────────────────────────────────────────

def sweep_one(case_key, req, q, k, tag_weight, gt50, gt200, use_pool_cap=False, n_distinct=10):
    """Run ANN at k, build pool as dedup union (or capped), score, rank, compute recall.
    
    For distinct-request latency, we re-run n_distinct times with slightly varied tag_weight
    to bypass any paging cache (the query changes the cache key).
    """
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    # Pool construction
    if f.active and meta_index.is_loaded():
        mask = meta_index.mask_for(f)
        idx = meta_index.get_index()
        i2r = idx.id_to_row
    else:
        mask = None

    latencies = []
    pool_sizes = []
    recall_50 = None
    recall_200 = None

    for trial in range(n_distinct):
        # tiny variation to bypass paging cache, clamped to [0, 1]
        tw = min(1.0, max(0.0, tag_weight + trial * 1e-7 * (1 if tag_weight < 0.5 else -1)))
        t0 = time.perf_counter()

        # Run ANN
        raw_lists = candidates._ann(q, where=None, k=k)

        # Build pool: dedup union vs capped interleave
        if use_pool_cap:
            if mask is not None:
                filtered_lists = [[mid for mid in lst if mask[i2r[mid]]] for lst in raw_lists]
            else:
                filtered_lists = raw_lists
            pool = candidates._interleave(filtered_lists, settings.pool_cap, exclude)
        else:
            # Deduplicated union, no cap
            seen = set()
            pool = []
            for lst in raw_lists:
                for mid in lst:
                    if mid not in seen and mid not in exclude:
                        if mask is not None:
                            if mask[i2r[mid]]:
                                seen.add(mid)
                                pool.append(mid)
                        else:
                            seen.add(mid)
                            pool.append(mid)

        # Score and rank
        s = scoring.score(pool, q)
        ranked = fusion.fuse(s, tw, top_n=200)
        t_total = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_total)
        pool_sizes.append(len(pool))

        # Compute recall only on first trial (tag_weight exact match)
        if trial == 0:
            ids_50 = set(r["movie_id"] for r in ranked[:50])
            ids_200 = set(r["movie_id"] for r in ranked[:200])
            n50 = min(50, len(ids_50), len(gt50))
            n200 = min(200, len(ids_200), len(gt200))
            recall_50 = len(ids_50 & gt50) / n50 if n50 > 0 else 1.0
            recall_200 = len(ids_200 & gt200) / n200 if n200 > 0 else 1.0

    return {
        "k": k,
        "tag_weight": tag_weight,
        "pool_cap": use_pool_cap,
        "pool_size": int(np.median(pool_sizes)),
        "recall_50": recall_50,
        "recall_200": recall_200,
        "lat_p50": float(np.percentile(latencies, 50)),
        "lat_p95": float(np.percentile(latencies, 95)),
        "n_trials": n_distinct,
    }


def main():
    open_pool()
    artifacts.load_all()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()
    meta_index.load_index()

    # Warm-up
    _p("Warm-up...")
    cases = get_sweep_cases()
    for ck, req in cases.items():
        recommender.retrieve_and_rank(req)
    _p("Warm-up done.\n")

    K_VALUES = [2000, 5000, 10000, 20000]
    TW_VALUES = [0.0, 0.3, 0.5, 0.8, 1.0]
    N_DISTINCT = 5  # trials per sweep cell for latency stats

    for case_key, req in cases.items():
        _p("=" * 110)
        _p(f"CASE {case_key}")
        _p("=" * 110)

        q = query_builder.build(req)
        gt50, gt200, gt_n, gt_time = compute_ground_truth(case_key, req, q)

        # Header
        _p(f"\n{'k':>6} {'tw':>5} {'cap':>5} {'pool':>8} {'R@50':>8} {'R@200':>8} {'lat_p50':>10} {'lat_p95':>10}")
        _p("-" * 75)

        # Old behavior: k=20000, pool_cap=2000
        for tw in TW_VALUES:
            r = sweep_one(case_key, req, q, 20000, tw, gt50, gt200, use_pool_cap=True, n_distinct=N_DISTINCT)
            _p(f"{r['k']:>6} {tw:>5.1f} {'YES':>5} {r['pool_size']:>8} {r['recall_50']*100:>7.1f}% {r['recall_200']*100:>7.1f}% {r['lat_p50']:>9.0f}ms {r['lat_p95']:>9.0f}ms")

        _p("-" * 75)

        # Sweep: union pool (no cap)
        for k_val in K_VALUES:
            for tw in TW_VALUES:
                r = sweep_one(case_key, req, q, k_val, tw, gt50, gt200, use_pool_cap=False, n_distinct=N_DISTINCT)
                _p(f"{r['k']:>6} {tw:>5.1f} {'NO':>5} {r['pool_size']:>8} {r['recall_50']*100:>7.1f}% {r['recall_200']*100:>7.1f}% {r['lat_p50']:>9.0f}ms {r['lat_p95']:>9.0f}ms")

        # Exact scoring (for case f comparison)
        f = Filters.from_request(req)
        if f.active:
            _p("-" * 75)
            _p(f"{'EXACT':>6} {'0.5':>5} {'N/A':>5} {gt_n:>8} {'100.0':>7}% {'100.0':>7}% {gt_time*1000:>9.0f}ms {gt_time*1000:>9.0f}ms")

        _p()

    close_pool()


if __name__ == "__main__":
    main()
