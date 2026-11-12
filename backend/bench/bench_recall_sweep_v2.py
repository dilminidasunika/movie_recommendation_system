import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.db.pool import open_pool
from app.db.queries import Filters
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion
from app.stores import artifacts, chroma, meta_index

def run_sweep(n_queries=50, rng_seed=42):
    print("=" * 80, flush=True)
    print("ITEM 2: PHASED HIGH-PERFORMANCE RECALL SWEEP (N=50 DISTINCT QUERIES PER CASE)", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    idx = meta_index.get_index()
    id_to_row = artifacts.id_to_row()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()

    plot_mem, plot_id_to_r = artifacts.plot_memmap()

    # Pre-warm Chroma
    print("Warming up Chroma collections...", flush=True)
    dummy_v = np.zeros(384, dtype=np.float32); dummy_v[0] = 1.0
    _ = chroma.query_ids(settings.plots_collection, dummy_v, 2000)
    dummy_t = np.zeros(300, dtype=np.float32); dummy_t[0] = 1.0
    _ = chroma.query_ids(settings.tags_collection, dummy_t, 2000)
    print("Chroma warmed up.", flush=True)

    common_ids = np.array(list(set(id_to_row.keys()).intersection(set(plot_id_to_r.keys()))))
    common_ids.sort()
    rng = np.random.default_rng(rng_seed)

    k_values = [2000, 5000, 10000, 20000]
    tw_values = [0.0, 0.2, 0.5, 0.8, 1.0]

    case_definitions = [
        ("a", "Unfiltered (3 seed movies)", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=0.5)),
        ("d", "Plot Text (3 seeds + query text)", lambda seeds: RecommendRequest(
            movie_ids=seeds,
            plot_text="A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption across generations.",
            tag_weight=0.5
        )),
        ("f", "Loose Filter (Drama, 295k matches)", lambda seeds: RecommendRequest(movie_ids=seeds, genres=["Drama"], tag_weight=0.5)),
    ]

    CHUNK = 100_000
    n_plot_rows = len(plot_mem)

    u_ids_all = common_ids
    u_rows_t_all = np.array([id_to_row[mid] for mid in u_ids_all], dtype=np.int32)
    u_rows_p_all = np.array([plot_id_to_r[mid] for mid in u_ids_all], dtype=np.int32)
    u_pos_map_all = {int(mid): i for i, mid in enumerate(u_ids_all)}

    results_all = {}

    for case_code, case_desc, req_fn in case_definitions:
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING CASE {case_code.upper()}: {case_desc}", flush=True)
        print(f"=======================================================", flush=True)

        t_case_start = time.perf_counter()

        # Step 1: Generate 50 queries
        queries = []
        for _ in range(n_queries):
            seeds = rng.choice(common_ids, size=3, replace=False).tolist()
            req = req_fn(seeds)
            q = query_builder.build(req)
            queries.append((seeds, req, q))

        f = Filters.from_request(queries[0][1])
        if f.active:
            mask = meta_index.mask_for(f)
            matching_ids = set(idx.ids[mask])
            u_ids_case = np.array([mid for mid in common_ids if mid in matching_ids])
            u_rows_t_case = np.array([id_to_row[mid] for mid in u_ids_case], dtype=np.int32)
            u_rows_p_case = np.array([plot_id_to_r[mid] for mid in u_ids_case], dtype=np.int32)
            u_pos_map_case = {int(mid): i for i, mid in enumerate(u_ids_case)}
            print(f"Universe for Case {case_code}: {len(u_ids_case):,} movies matching filter", flush=True)
        else:
            mask = None
            matching_ids = None
            u_ids_case = u_ids_all
            u_rows_t_case = u_rows_t_all
            u_rows_p_case = u_rows_p_all
            u_pos_map_case = u_pos_map_all
            print(f"Universe for Case {case_code}: Full catalog ({len(u_ids_case):,} movies)", flush=True)

        # Step 2: Phase 1 - Chroma ANN candidate retrieval (Chroma index stays resident in RAM)
        print(f"Phase 1: Retrieving Chroma ANN candidates for {n_queries} queries...", flush=True)
        t_p1 = time.perf_counter()
        query_candidates = []  # list of dicts: {k: (raw_tag, raw_plot, ann_lat)}
        for q_idx, (seeds, req, q) in enumerate(queries):
            cands_by_k = {}
            for k in k_values:
                t0 = time.perf_counter()
                raw_tag = chroma.query_ids(settings.tags_collection, q.tag_ann, k)
                raw_plot = chroma.query_ids(settings.plots_collection, q.plot_centroid, k)
                dt_ann = (time.perf_counter() - t0) * 1000.0
                cands_by_k[k] = (raw_tag, raw_plot, dt_ann)
            query_candidates.append(cands_by_k)
            if (q_idx + 1) % 10 == 0:
                print(f"  ANN done {q_idx+1}/{n_queries} queries (Elapsed: {time.perf_counter() - t_p1:.1f}s)", flush=True)
        print(f"Phase 1 completed in {time.perf_counter() - t_p1:.1f}s", flush=True)

        # Step 3: Phase 2 - Exact scoring and Ground Truth evaluation
        print(f"Phase 2: Computing Ground Truth and scoring candidate pools...", flush=True)
        t_p2 = time.perf_counter()
        case_results = {tw: {k: {"r50": [], "r200": [], "lat": []} for k in k_values} for tw in tw_values}
        old_behavior = {tw: {"r50": [], "r200": [], "lat": []} for tw in tw_values}

        for q_idx, (seeds, req, q) in enumerate(queries):
            exclude = set(seeds)
            cands_by_k = query_candidates[q_idx]

            # Vectorized Ground Truth
            dots_tag = (tag_m @ q.tag_sparse.T).toarray().ravel()
            valid_t = tag_norms > 0
            sims_tag = np.zeros(len(tag_norms), dtype=np.float32)
            sims_tag[valid_t] = dots_tag[valid_t] / tag_norms[valid_t]

            c = np.asarray(q.plot_centroid, dtype=np.float32)
            cn = np.linalg.norm(c)
            c_unit = (c / cn) if cn > 0 else c
            sims_plot = np.empty(n_plot_rows, dtype=np.float32)
            for ch_start in range(0, n_plot_rows, CHUNK):
                ch_end = min(ch_start + CHUNK, n_plot_rows)
                sims_plot[ch_start:ch_end] = plot_mem[ch_start:ch_end] @ c_unit

            t_vec = sims_tag[u_rows_t_case].copy()
            p_vec = sims_plot[u_rows_p_case].copy()

            for s in seeds:
                if s in u_pos_map_case:
                    t_vec[u_pos_map_case[s]] = -1.0
                    p_vec[u_pos_map_case[s]] = -1.0

            gt50 = {}
            gt200 = {}
            for tw in tw_values:
                combined = tw * t_vec + (1.0 - tw) * p_vec
                top200_idx = np.argpartition(-combined, 200)[:200]
                top200_sorted = top200_idx[np.argsort(-combined[top200_idx])]
                gt50[tw] = set(u_ids_case[top200_sorted[:50]])
                gt200[tw] = set(u_ids_case[top200_sorted[:200]])

            # Evaluate each k
            for k in k_values:
                raw_tag, raw_plot, dt_ann = cands_by_k[k]

                if mask is not None:
                    raw_tag = [mid for mid in raw_tag if mid in matching_ids]
                    raw_plot = [mid for mid in raw_plot if mid in matching_ids]

                pool = list(dict.fromkeys(raw_tag + raw_plot))
                pool = [mid for mid in pool if mid not in exclude]

                t_sc0 = time.perf_counter()
                s = scoring.score(pool, q)
                dt_score = (time.perf_counter() - t_sc0) * 1000.0

                for tw in tw_values:
                    t_f0 = time.perf_counter()
                    ranked = fusion.fuse(s, tw, top_n=200)
                    dt_total = dt_ann + dt_score + (time.perf_counter() - t_f0) * 1000.0

                    r_ids50 = set(r["movie_id"] for r in ranked[:50])
                    r_ids200 = set(r["movie_id"] for r in ranked[:200])

                    case_results[tw][k]["r50"].append(len(r_ids50.intersection(gt50[tw])) / 50.0)
                    case_results[tw][k]["r200"].append(len(r_ids200.intersection(gt200[tw])) / 200.0)
                    case_results[tw][k]["lat"].append(dt_total)

            # Evaluate old behavior (k=20000 with pool_cap=2000)
            raw_tag_20k, raw_plot_20k, dt_ann_20k = cands_by_k[20000]
            if mask is not None:
                raw_tag_20k = [mid for mid in raw_tag_20k if mid in matching_ids]
                raw_plot_20k = [mid for mid in raw_plot_20k if mid in matching_ids]
            pool_old = candidates._interleave([raw_tag_20k, raw_plot_20k], 2000, exclude)
            t_sc0 = time.perf_counter()
            s_old = scoring.score(pool_old, q)
            dt_score_old = (time.perf_counter() - t_sc0) * 1000.0

            for tw in tw_values:
                t_f0 = time.perf_counter()
                ranked_old = fusion.fuse(s_old, tw, top_n=200)
                dt_total_old = dt_ann_20k + dt_score_old + (time.perf_counter() - t_f0) * 1000.0
                r_old50 = set(r["movie_id"] for r in ranked_old[:50])
                r_old200 = set(r["movie_id"] for r in ranked_old[:200])
                old_behavior[tw]["r50"].append(len(r_old50.intersection(gt50[tw])) / 50.0)
                old_behavior[tw]["r200"].append(len(r_old200.intersection(gt200[tw])) / 200.0)
                old_behavior[tw]["lat"].append(dt_total_old)

            if (q_idx + 1) % 10 == 0:
                print(f"  Scoring & Recall done {q_idx+1}/{n_queries} queries (Elapsed: {time.perf_counter() - t_p2:.1f}s)", flush=True)

        print(f"Phase 2 completed in {time.perf_counter() - t_p2:.1f}s (Total case time: {time.perf_counter() - t_case_start:.1f}s)", flush=True)
        results_all[case_code] = {"uncapped": case_results, "old": old_behavior}

    # Step 4: Summary Tables
    print("\n" + "=" * 80, flush=True)
    print("FINAL RESULTS SUMMARY: RECALL SWEEP (N=50 DISTINCT QUERIES EACH)", flush=True)
    print("=" * 80, flush=True)

    for case_code, case_desc, _ in case_definitions:
        res = results_all[case_code]
        print(f"\n--- CASE {case_code.upper()}: {case_desc} ---", flush=True)
        print(f"{'tag_weight':<10} {'k':<8} {'Pool':<10} {'R@50 Mean':<12} {'R@50 p5':<12} {'R@200 Mean':<13} {'R@200 p5':<12} {'p50 Lat (ms)':<14} {'p95 Lat (ms)':<14}")
        print("-" * 98)

        for tw in tw_values:
            old_r50_m = np.mean(res["old"][tw]["r50"]) * 100
            old_r50_p5 = np.percentile(res["old"][tw]["r50"], 5) * 100
            old_r200_m = np.mean(res["old"][tw]["r200"]) * 100
            old_r200_p5 = np.percentile(res["old"][tw]["r200"], 5) * 100
            old_p50 = np.median(res["old"][tw]["lat"])
            old_p95 = np.percentile(res["old"][tw]["lat"], 95)
            print(f"{tw:<10.1f} {'20k(old)':<8} {'2,000':<10} {old_r50_m:<12.1f}% {old_r50_p5:<12.1f}% {old_r200_m:<13.1f}% {old_r200_p5:<12.1f}% {old_p50:<14.1f} {old_p95:<14.1f}")

            for k in k_values:
                d = res["uncapped"][tw][k]
                r50_m = np.mean(d["r50"]) * 100
                r50_p5 = np.percentile(d["r50"], 5) * 100
                r200_m = np.mean(d["r200"]) * 100
                r200_p5 = np.percentile(d["r200"], 5) * 100
                p50 = np.median(d["lat"])
                p95 = np.percentile(d["lat"], 95)
                est_pool = f"~{k*2:,}"
                print(f"{tw:<10.1f} {k:<8} {est_pool:<10} {r50_m:<12.1f}% {r50_p5:<12.1f}% {r200_m:<13.1f}% {r200_p5:<12.1f}% {p50:<14.1f} {p95:<14.1f}")
            print("-" * 98)

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [float(x) for x in obj]
        return float(obj)

    with open("recall_sweep_v2_results.json", "w") as f:
        json.dump(clean(results_all), f, indent=2)
    print("\nSaved full results to recall_sweep_v2_results.json", flush=True)

if __name__ == "__main__":
    run_sweep(n_queries=50, rng_seed=42)
