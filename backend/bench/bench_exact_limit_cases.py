import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
from datetime import date
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.exact_limit = 300_000
settings.use_matvec_tag_candidates = True

from app.db.pool import open_pool
from app.db.queries import Filters
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import query_builder, recommender

def run():
    print("=" * 80)
    print("ITEM 4: EXACT_LIMIT = 300,000 ON CASES B, E, F (DISTINCT QUERIES)")
    print("=" * 80)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    idx = meta_index.get_index()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()

    common_ids = np.array(list(set(id_to_row.keys()).intersection(set(plot_id_to_r.keys()))))
    common_ids.sort()

    rng = np.random.default_rng(42)

    cases = [
        ("b", "Filter 55k (en, Drama, 1970-2020)", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            original_language=["en"],
            genres=["Drama"],
            date_from=date(1970, 1, 1),
            date_to=date(2020, 12, 31),
            tag_weight=0.5
        )),
        ("e", "Rare Lang (fr, 1950-2024, ~60k matches)", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            original_language=["fr"],
            date_from=date(1950, 1, 1),
            date_to=date(2024, 12, 31),
            tag_weight=0.5
        )),
        ("f", "Loose Filter (Drama, ~295k matches)", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            genres=["Drama"],
            tag_weight=0.5
        )),
    ]

    for case_code, case_name, req_fn in cases:
        print(f"\n--- Testing Case {case_code.upper()}: {case_name} ---")
        dummy_req = req_fn([238, 240, 278])
        f = Filters.from_request(dummy_req)
        mask = meta_index.mask_for(f)
        matching_ids = idx.ids[mask]
        n_matches = len(matching_ids)
        print(f"Mask matching movies: {n_matches:,} (exact_limit = {settings.exact_limit:,})")

        # Pick 20 sets of 3 distinct seeds from common_ids
        n_queries = 20
        all_seeds = [rng.choice(common_ids, size=3, replace=False).tolist() for _ in range(n_queries)]

        r50_list, r200_list = [], []
        lat_list = []
        timings_breakdown = {"query_build": [], "candidates": [], "scoring": [], "fusion": [], "total": []}

        for q_i, seeds in enumerate(all_seeds):
            req = req_fn(seeds)
            res = recommender.retrieve_and_rank(req)

            ranked_ids = [r["movie_id"] for r in res["ranked"]]
            ranked_top50 = set(ranked_ids[:50])
            ranked_top200 = set(ranked_ids[:200])

            # Ground truth: Exact full scoring over all matching_ids excluding seeds
            exclude = set(seeds)
            q = res["query"]

            # Vectorized exact ground truth over matching_ids
            # Tag score:
            dots_tag = (tag_m @ q.tag_sparse.T).toarray().ravel()
            valid_t = tag_norms > 0
            sims_tag = np.zeros(len(tag_norms), dtype=np.float32)
            sims_tag[valid_t] = dots_tag[valid_t] / tag_norms[valid_t]

            # Plot score:
            if q.plot_centroid is not None:
                c = np.asarray(q.plot_centroid, dtype=np.float32)
                cn = np.linalg.norm(c)
                c_unit = (c / cn) if cn > 0 else c
                sims_plot = plot_mem @ c_unit
            else:
                sims_plot = np.zeros(len(plot_mem), dtype=np.float32)

            valid_mids = [m for m in matching_ids if m not in exclude]
            t_scores = np.array([sims_tag[id_to_row[m]] if m in id_to_row else 0.0 for m in valid_mids], dtype=np.float32)
            p_scores = np.array([sims_plot[plot_id_to_r[m]] if m in plot_id_to_r else 0.0 for m in valid_mids], dtype=np.float32)

            tw = req.tag_weight
            t_scaled = np.clip(t_scores, 0.0, 1.0)
            p_scaled = np.clip(p_scores, 0.0, 1.0)
            t_ok = np.array([m in id_to_row and tag_norms[id_to_row[m]] > 0 for m in valid_mids], dtype=bool)
            p_ok = np.array([m in plot_id_to_r for m in valid_mids], dtype=bool)
            t_scaled[~t_ok] = 0.0
            p_scaled[~p_ok] = 0.0

            combined = tw * t_scaled + (1.0 - tw) * p_scaled
            combined[~(t_ok | p_ok)] = -1.0

            top200_idx = np.argpartition(-combined, min(200, len(combined)-1))[:200]
            top200_sorted = top200_idx[np.argsort(-combined[top200_idx])]

            gt_top50 = set(np.array(valid_mids)[top200_sorted[:50]])
            gt_top200 = set(np.array(valid_mids)[top200_sorted[:200]])

            r50 = len(ranked_top50 & gt_top50) / float(len(gt_top50)) if gt_top50 else 1.0
            r200 = len(ranked_top200 & gt_top200) / float(len(gt_top200)) if gt_top200 else 1.0

            r50_list.append(r50)
            r200_list.append(r200)
            lat_list.append(res["timings"]["total"] * 1000.0)
            for k in timings_breakdown:
                if k in res["timings"]:
                    timings_breakdown[k].append(res["timings"][k] * 1000.0)

        mean_r50 = np.mean(r50_list) * 100.0
        mean_r200 = np.mean(r200_list) * 100.0
        p50_lat = np.percentile(lat_list, 50)
        p95_lat = np.percentile(lat_list, 95)
        mean_cands = np.mean(timings_breakdown["candidates"])
        mean_score = np.mean(timings_breakdown["scoring"])
        mean_fusion = np.mean(timings_breakdown["fusion"])

        print(f"Results for Case {case_code.upper()} across 20 distinct queries:")
        print(f"  Recall@50:                     {mean_r50:.2f}% (min: {np.min(r50_list)*100:.1f}%)")
        print(f"  Recall@200:                    {mean_r200:.2f}% (min: {np.min(r200_list)*100:.1f}%)")
        print(f"  Latency p50 / p95:             {p50_lat:.1f} ms / {p95_lat:.1f} ms")
        print(f"  Mean Candidates Stage Time:    {mean_cands:.1f} ms")
        print(f"  Mean Scoring Stage Time:       {mean_score:.1f} ms")
        print(f"  Mean Fusion Stage Time:        {mean_fusion:.1f} ms")

if __name__ == "__main__":
    run()
