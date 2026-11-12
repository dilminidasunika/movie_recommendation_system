"""
Round 4 Item 8: Product note.
Count how many of top-50 results for cases a,d at tw=0.5 are plot-only
or tag-only movies vs their ~22% catalog share.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True
settings.use_uncapped_pool = True
settings.k_tag = 10000
settings.k_plot = 5000
settings.exact_limit = 300000

from app.db.pool import open_pool
from app.stores import artifacts, meta_index, chroma, text_embedder
from app.schemas import RecommendRequest
from app.services import recommender

import numpy as np


def run():
    N = 50
    RNG_SEED = 42

    print("=" * 80, flush=True)
    print("ROUND 4 ITEM 8: PRODUCT NOTE — PLOT-ONLY & TAG-ONLY IN TOP-50", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    text_embedder.get_sentence_transformer()

    tag_ids = artifacts.matrix_ids()
    plot_ids = np.load(settings.artifact_path / "plot_embeddings_movie_ids.npy")

    set_tag = set(int(i) for i in tag_ids)
    set_plot = set(int(i) for i in plot_ids)
    both = set_tag & set_plot
    plot_only = set_plot - set_tag
    tag_only = set_tag - set_plot
    unified = set_tag | set_plot

    print(f"Catalog stats:", flush=True)
    print(f"  Both tag+plot: {len(both):,} ({100*len(both)/len(unified):.1f}%)", flush=True)
    print(f"  Plot-only:     {len(plot_only):,} ({100*len(plot_only)/len(unified):.1f}%)", flush=True)
    print(f"  Tag-only:      {len(tag_only):,} ({100*len(tag_only)/len(unified):.1f}%)", flush=True)
    print(f"  Unified:       {len(unified):,}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    common_ids = np.array(sorted(both))

    cases = [
        ("a", "Unfiltered", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=0.5)),
        ("d", "Plot Text", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            plot_text="A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption across generations.",
            tag_weight=0.5
        )),
    ]

    for case_code, case_desc, req_fn in cases:
        print(f"\n--- Case {case_code.upper()}: {case_desc} (tw=0.5) ---", flush=True)

        plot_only_counts = []
        tag_only_counts = []
        both_counts = []
        total_ranked = []

        for q_idx in range(N):
            seeds = rng.choice(common_ids, size=3, replace=False).tolist()
            req = req_fn(seeds)
            result = recommender.retrieve_and_rank(req)
            ranked = result["ranked"]
            top50_ids = [r["movie_id"] for r in ranked[:50]]

            n_both = sum(1 for mid in top50_ids if mid in both)
            n_plot_only = sum(1 for mid in top50_ids if mid in plot_only)
            n_tag_only = sum(1 for mid in top50_ids if mid in tag_only)
            n_neither = len(top50_ids) - n_both - n_plot_only - n_tag_only

            plot_only_counts.append(n_plot_only)
            tag_only_counts.append(n_tag_only)
            both_counts.append(n_both)
            total_ranked.append(len(top50_ids))

            if q_idx < 3 or q_idx % 10 == 0:
                print(f"  Q{q_idx}: top50={len(top50_ids)} both={n_both} plot_only={n_plot_only} tag_only={n_tag_only} neither={n_neither}", flush=True)

        po = np.array(plot_only_counts)
        to = np.array(tag_only_counts)
        bo = np.array(both_counts)
        total = np.array(total_ranked)

        print(f"\n  Summary for Case {case_code.upper()} ({N} queries):", flush=True)
        print(f"    Both tag+plot in top-50:  mean={np.mean(bo):.1f} (median={np.median(bo):.0f})", flush=True)
        print(f"    Plot-only in top-50:      mean={np.mean(po):.1f} (median={np.median(po):.0f})", flush=True)
        print(f"    Tag-only in top-50:       mean={np.mean(to):.1f} (median={np.median(to):.0f})", flush=True)
        print(f"    Total ranked:             mean={np.mean(total):.1f}", flush=True)
        print(f"    Plot-only fraction:       {100*np.mean(po)/50:.1f}% (catalog: {100*len(plot_only)/len(unified):.1f}%)", flush=True)
        print(f"    Tag-only fraction:        {100*np.mean(to)/50:.1f}% (catalog: {100*len(tag_only)/len(unified):.1f}%)", flush=True)
        print(f"    Single-signal fraction:   {100*(np.mean(po)+np.mean(to))/50:.1f}% (catalog: {100*(len(plot_only)+len(tag_only))/len(unified):.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("INTERPRETATION", flush=True)
    print(f"{'='*70}", flush=True)
    print("At tw=0.5, both tag and plot scores contribute equally.", flush=True)
    print("Plot-only movies get tag_score=0 and tag_ok=False, so their final", flush=True)
    print("score is 0.5*plot_score + 0 = 0.5*plot_score (halved).", flush=True)
    print("Similarly, tag-only movies get final_score = 0.5*tag_score.", flush=True)
    print("Both-signal movies get the full weighted combination.", flush=True)
    print("This means single-signal movies are penalized at mixed tag_weights.", flush=True)
    print("Their presence in top-50 depends on whether their single score", flush=True)
    print("is strong enough to overcome the halving.", flush=True)


if __name__ == "__main__":
    run()
