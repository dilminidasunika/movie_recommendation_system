"""
Round 4 Item 5: Tail recall investigation.
At tag_weight=0.0 the p5 recall@200 is 57.8%. Identify the 3 worst queries.
For each: how many seeds have plot embeddings, ANN recall at k=5000 and 10000,
how many GT top-200 ids are missing from Chroma result, and whether those ids exist.
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

from app.db.pool import open_pool
from app.stores import artifacts, meta_index, chroma
from app.schemas import RecommendRequest
from app.services import query_builder

import numpy as np
from sklearn.preprocessing import normalize


def run():
    N = 50
    RNG_SEED = 42

    print("=" * 80, flush=True)
    print("ROUND 4 ITEM 5: TAIL RECALL INVESTIGATION (tag_weight=0.0)", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)

    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_ids = np.load(settings.artifact_path / "plot_embeddings_movie_ids.npy")

    set_tag = set(tag_ids)
    set_plot = set(int(pid) for pid in plot_ids)
    unified_ids = np.array(sorted(set_tag | set_plot))

    u_has_plot = np.array([mid in plot_id_to_r for mid in unified_ids], dtype=bool)
    u_plot_rows = np.array([plot_id_to_r[mid] if mid in plot_id_to_r else 0 for mid in unified_ids], dtype=np.int32)
    u_id_to_pos = {int(mid): i for i, mid in enumerate(unified_ids)}

    rng = np.random.default_rng(RNG_SEED)
    common_ids = np.array(sorted(set_tag & set_plot))

    # Verify Chroma collection count
    col = chroma.get_collection(settings.plots_collection)
    if col:
        print(f"Chroma plots_embeddings count: {col.count()}", flush=True)
    print(f"Plot memmap rows: {len(plot_mem)}", flush=True)
    print(f"Unified catalog: {len(unified_ids)}", flush=True)

    CHUNK = 100_000
    n_plot_rows = len(plot_mem)

    # Generate queries
    queries = []
    for _ in range(N):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, tag_weight=0.0)
        q = query_builder.build(req)
        queries.append((seeds, req, q))

    # For each query, compute exact plot top-200 and ANN recall
    recalls = []
    query_details = []

    for q_idx, (seeds, req, q) in enumerate(queries):
        exclude = set(seeds)

        # Query vector (normalized)
        c = np.asarray(q.plot_centroid, dtype=np.float32)
        cn = np.linalg.norm(c)
        c_unit = (c / cn) if cn > 0 else c

        # EXACT plot cosine over FULL memmap
        all_sims = np.empty(n_plot_rows, dtype=np.float32)
        for ch_s in range(0, n_plot_rows, CHUNK):
            ch_e = min(ch_s + CHUNK, n_plot_rows)
            all_sims[ch_s:ch_e] = plot_mem[ch_s:ch_e] @ c_unit

        # Mark seeds
        for s in exclude:
            if s in plot_id_to_r:
                all_sims[plot_id_to_r[s]] = -999.0

        # Exact top-200 by plot
        top200_idx = np.argpartition(-all_sims, 200)[:200]
        top200_sorted = top200_idx[np.argsort(-all_sims[top200_idx])]
        gt_200 = set(int(plot_ids[r]) for r in top200_sorted)
        gt_200_scores = {int(plot_ids[r]): float(all_sims[r]) for r in top200_sorted}

        # Now compute: over UNIFIED universe with tw=0.0 (pure plot)
        u_p = np.zeros(len(unified_ids), dtype=np.float32)
        u_p[u_has_plot] = np.clip(all_sims[u_plot_rows[u_has_plot]], 0.0, 1.0)
        for s in exclude:
            if s in u_id_to_pos:
                u_p[u_id_to_pos[s]] = -1.0

        # Unified GT top-200 (tw=0.0 means pure plot on unified universe)
        top200_u_idx = np.argpartition(-u_p, 200)[:200]
        top200_u_sorted = top200_u_idx[np.argsort(-u_p[top200_u_idx])]
        gt_200_unified = set(int(unified_ids[i]) for i in top200_u_sorted)

        # Chroma ANN results at k=5000 and k=10000
        ann_5k = set(chroma.query_ids(settings.plots_collection, c_unit, 5000))
        ann_10k = set(chroma.query_ids(settings.plots_collection, c_unit, 10000))

        # Recall of ANN vs exact-plot GT
        r_5k = len(ann_5k & gt_200) / len(gt_200) if gt_200 else 1.0
        r_10k = len(ann_10k & gt_200) / len(gt_200) if gt_200 else 1.0

        # Recall of pipeline at k_plot=5000 (production setting)
        # Pipeline recall = fraction of unified GT top-200 that appear in the pool
        # Pool = tag_top_10k union plot_top_5k (filtered for exclude)
        pool_plot_5k = [mid for mid in chroma.query_ids(settings.plots_collection, c_unit, 5000) if mid not in exclude]
        # For tw=0.0, tag candidates don't matter for scoring, but they are in the pool
        # Let's measure just the plot-ANN recall against unified GT
        pool_set = set(pool_plot_5k)
        pipeline_recall = len(pool_set & gt_200_unified) / len(gt_200_unified) if gt_200_unified else 1.0

        # How many seeds have plot embeddings
        seeds_with_plot = sum(1 for s in seeds if s in plot_id_to_r)

        detail = {
            "q_idx": q_idx,
            "seeds": seeds,
            "seeds_with_plot": seeds_with_plot,
            "ann_recall_5k": r_5k,
            "ann_recall_10k": r_10k,
            "pipeline_recall_200": pipeline_recall,
            "gt_200_plot_count": len(gt_200),
            "gt_200_unified_count": len(gt_200_unified),
            "missing_from_ann_10k": len(gt_200 - ann_10k),
        }
        recalls.append(pipeline_recall)
        query_details.append(detail)

        if q_idx < 3 or q_idx % 10 == 0:
            print(f"  Q{q_idx}: seeds_plot={seeds_with_plot} ann_r@5k={r_5k:.3f} ann_r@10k={r_10k:.3f} pipeline_r@200={pipeline_recall:.3f}", flush=True)

    recalls = np.array(recalls)
    print(f"\nOverall: mean={np.mean(recalls):.4f} p5={np.percentile(recalls, 5):.4f} min={np.min(recalls):.4f}", flush=True)

    # Find 3 worst
    worst_idx = np.argsort(recalls)[:3]
    print(f"\n{'='*70}", flush=True)
    print("3 WORST QUERIES AT tag_weight=0.0", flush=True)
    print(f"{'='*70}", flush=True)

    for rank, wi in enumerate(worst_idx):
        d = query_details[wi]
        print(f"\n--- Worst #{rank+1}: Query {d['q_idx']} (pipeline recall@200 = {recalls[wi]:.4f}) ---", flush=True)
        print(f"  Seeds: {d['seeds']}", flush=True)
        print(f"  Seeds with plot embeddings: {d['seeds_with_plot']}/3", flush=True)
        print(f"  ANN recall@5k  (vs exact-plot top-200): {d['ann_recall_5k']:.4f}", flush=True)
        print(f"  ANN recall@10k (vs exact-plot top-200): {d['ann_recall_10k']:.4f}", flush=True)
        print(f"  GT top-200 in plot-only space: {d['gt_200_plot_count']}", flush=True)
        print(f"  GT top-200 on unified universe: {d['gt_200_unified_count']}", flush=True)
        print(f"  Missing from ANN 10k result: {d['missing_from_ann_10k']}", flush=True)

        # Deep dive: are the missing IDs in the Chroma collection?
        seeds = d["seeds"]
        exclude = set(seeds)
        q = queries[wi][2]
        c = np.asarray(q.plot_centroid, dtype=np.float32)
        cn = np.linalg.norm(c)
        c_unit = (c / cn) if cn > 0 else c

        # Exact plot sims
        all_sims = np.empty(n_plot_rows, dtype=np.float32)
        for ch_s in range(0, n_plot_rows, CHUNK):
            ch_e = min(ch_s + CHUNK, n_plot_rows)
            all_sims[ch_s:ch_e] = plot_mem[ch_s:ch_e] @ c_unit
        for s in exclude:
            if s in plot_id_to_r:
                all_sims[plot_id_to_r[s]] = -999.0

        top200_idx = np.argpartition(-all_sims, 200)[:200]
        top200_sorted = top200_idx[np.argsort(-all_sims[top200_idx])]
        gt_200 = [int(plot_ids[r]) for r in top200_sorted]

        ann_10k = set(chroma.query_ids(settings.plots_collection, c_unit, 10000))
        missing = [mid for mid in gt_200 if mid not in ann_10k]

        # Score at rank 200
        score_200 = float(all_sims[top200_sorted[199]]) if len(top200_sorted) >= 200 else None
        score_1 = float(all_sims[top200_sorted[0]]) if len(top200_sorted) >= 1 else None

        print(f"  Score at rank 1: {score_1:.6f}", flush=True)
        print(f"  Score at rank 200: {score_200:.6f}", flush=True)

        # How many ties at the boundary?
        if score_200 is not None:
            n_ties = int(np.sum(np.abs(all_sims - score_200) < 1e-6))
            print(f"  Ties at rank-200 score (±1e-6): {n_ties}", flush=True)

        if missing:
            print(f"  {len(missing)} missing IDs from ANN 10k:", flush=True)
            for mid in missing[:10]:
                in_collection = mid in plot_id_to_r  # if it's in memmap, it should be in Chroma
                exact_score = float(all_sims[plot_id_to_r[mid]]) if mid in plot_id_to_r else None
                print(f"    ID {mid}: in_plot_memmap={mid in plot_id_to_r} exact_score={exact_score}", flush=True)
            if len(missing) > 10:
                print(f"    ... and {len(missing)-10} more", flush=True)
        else:
            print(f"  All GT top-200 found in ANN 10k!", flush=True)

    out_file = SCRIPT_DIR / "r4_tail_recall.json"
    with open(out_file, "w") as f:
        json.dump({
            "overall_mean": float(np.mean(recalls)),
            "overall_p5": float(np.percentile(recalls, 5)),
            "overall_min": float(np.min(recalls)),
            "worst_3": [query_details[i] for i in worst_idx],
            "all_queries": query_details,
        }, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    run()
