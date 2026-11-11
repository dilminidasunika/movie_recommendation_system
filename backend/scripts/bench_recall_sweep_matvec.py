import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
import json
import ctypes
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True

from app.db.pool import open_pool
from app.db.queries import Filters
from app.stores import artifacts, meta_index, chroma
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def get_free_ram_gb():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024**3)

def run_sweep(n_queries=50, rng_seed=42):
    print("=" * 80, flush=True)
    print("ITEM 3: RECALL & LATENCY SWEEP WITH MATVEC TAG CANDIDATE GENERATION (FLAG ON)", flush=True)
    print(f"N = {n_queries} distinct queries per case. Start Free RAM: {get_free_ram_gb():.2f} GB", flush=True)
    print("=" * 80, flush=True)

    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    idx = meta_index.get_index()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_ids = np.load(artifacts.settings.artifact_path / "plot_embeddings_movie_ids.npy")

    # Warmup Chroma plots collection
    print("Pre-warming Chroma plots collection...", flush=True)
    dummy_v = np.zeros(384, dtype=np.float32); dummy_v[0] = 1.0
    _ = chroma.query_ids(settings.plots_collection, dummy_v, 2000)
    print("Chroma plots collection warmed.", flush=True)

    # UNIFIED UNIVERSE DEFINITION
    set_tag = set(tag_ids)
    set_plot = set(plot_ids)
    unified_ids = np.array(sorted(list(set_tag | set_plot)))
    u_n = len(unified_ids)
    print(f"Unified Scoreable Catalog: {u_n:,} movies (Tags: {len(set_tag):,}, Plots: {len(set_plot):,})", flush=True)

    # Pre-compute indexing structures for fast vectorized Ground Truth over Unified Universe
    u_has_tag = np.array([mid in id_to_row for mid in unified_ids], dtype=bool)
    u_tag_rows = np.array([id_to_row[mid] if mid in id_to_row else 0 for mid in unified_ids], dtype=np.int32)

    u_has_plot = np.array([mid in plot_id_to_r for mid in unified_ids], dtype=bool)
    u_plot_rows = np.array([plot_id_to_r[mid] if mid in plot_id_to_r else 0 for mid in unified_ids], dtype=np.int32)

    u_id_to_pos = {int(mid): i for i, mid in enumerate(unified_ids)}

    common_ids = np.array(sorted(list(set_tag & set_plot)))
    rng = np.random.default_rng(rng_seed)

    k_tag_values = [2000, 5000, 10000]
    k_plot_values = [2000, 5000, 10000]
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

    results_all = {}

    for case_code, case_desc, req_fn in case_definitions:
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING CASE {case_code.upper()}: {case_desc}", flush=True)
        print(f"=======================================================", flush=True)

        t_case_start = time.perf_counter()

        # Step 1: Generate 50 distinct queries
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
            case_u_mask = np.array([mid in matching_ids for mid in unified_ids], dtype=bool)
            case_u_ids = unified_ids[case_u_mask]
            case_u_has_tag = u_has_tag[case_u_mask]
            case_u_tag_rows = u_tag_rows[case_u_mask]
            case_u_has_plot = u_has_plot[case_u_mask]
            case_u_plot_rows = u_plot_rows[case_u_mask]
            case_u_id_to_pos = {int(mid): i for i, mid in enumerate(case_u_ids)}
            print(f"Universe for Case {case_code}: {len(case_u_ids):,} movies matching filter", flush=True)
        else:
            mask = None
            matching_ids = None
            case_u_ids = unified_ids
            case_u_has_tag = u_has_tag
            case_u_tag_rows = u_tag_rows
            case_u_has_plot = u_has_plot
            case_u_plot_rows = u_plot_rows
            case_u_id_to_pos = u_id_to_pos
            print(f"Universe for Case {case_code}: Full unified catalog ({len(case_u_ids):,} movies)", flush=True)

        # Step 2: Retrieve candidates and measure latency
        # For each query, fetch max k (10,000) for plot ANN and tag matvec
        # Then slice for (k_tag, k_plot) combinations
        print(f"Running candidate retrieval and latency measurement for {n_queries} queries...", flush=True)
        query_data = []

        for q_idx, (seeds, req, q) in enumerate(queries):
            exclude = set(seeds)

            # Measure candidate retrieval latency
            # 1. Tag matvec
            t0 = time.perf_counter()
            dots_tag = (tag_m @ q.tag_sparse.T).toarray().ravel()
            valid_t = tag_norms > 0
            sims_tag = np.zeros(len(tag_norms), dtype=np.float32)
            sims_tag[valid_t] = dots_tag[valid_t] / tag_norms[valid_t]
            scores_pick = sims_tag.copy()
            for s in seeds:
                if s in id_to_row:
                    scores_pick[id_to_row[s]] = -1.0
            top10k_tag_idx = np.argpartition(-scores_pick, 10000)[:10000]
            top10k_tag_sorted = top10k_tag_idx[np.argsort(-scores_pick[top10k_tag_idx])]
            raw_tag_10k = [int(tag_ids[r]) for r in top10k_tag_sorted if scores_pick[r] > -0.5]
            t_tag_matvec = (time.perf_counter() - t0) * 1000.0

            # 2. Chroma plot ANN
            t0 = time.perf_counter()
            raw_plot_10k = chroma.query_ids(settings.plots_collection, q.plot_centroid, 10000)
            t_plot_ann = (time.perf_counter() - t0) * 1000.0

            # Under concurrent execution on POOL, candidate generation time is max(t_tag_matvec, t_plot_ann)
            t_cand_concurrent = max(t_tag_matvec, t_plot_ann)

            # Ground Truth computation
            c = np.asarray(q.plot_centroid, dtype=np.float32)
            cn = np.linalg.norm(c)
            c_unit = (c / cn) if cn > 0 else c
            sims_plot = np.empty(n_plot_rows, dtype=np.float32)
            for ch_start in range(0, n_plot_rows, CHUNK):
                ch_end = min(ch_start + CHUNK, n_plot_rows)
                sims_plot[ch_start:ch_end] = plot_mem[ch_start:ch_end] @ c_unit

            # Populate unified ground truth vectors
            u_t = np.zeros(len(case_u_ids), dtype=np.float32)
            u_t[case_u_has_tag] = np.clip(sims_tag[case_u_tag_rows[case_u_has_tag]], 0.0, 1.0)

            u_p = np.zeros(len(case_u_ids), dtype=np.float32)
            u_p[case_u_has_plot] = np.clip(sims_plot[case_u_plot_rows[case_u_has_plot]], 0.0, 1.0)

            for s in seeds:
                if s in case_u_id_to_pos:
                    u_t[case_u_id_to_pos[s]] = -1.0
                    u_p[case_u_id_to_pos[s]] = -1.0

            gt50 = {}
            gt200 = {}
            for tw in tw_values:
                combined = tw * u_t + (1.0 - tw) * u_p
                top200_idx = np.argpartition(-combined, 200)[:200]
                top200_sorted = top200_idx[np.argsort(-combined[top200_idx])]
                gt50[tw] = set(case_u_ids[top200_sorted[:50]])
                gt200[tw] = set(case_u_ids[top200_sorted[:200]])

            query_data.append({
                "seeds": seeds,
                "q": q,
                "sims_tag": sims_tag,
                "raw_tag_10k": raw_tag_10k,
                "raw_plot_10k": raw_plot_10k,
                "t_cand": t_cand_concurrent,
                "gt50": gt50,
                "gt200": gt200,
            })

            if (q_idx + 1) % 10 == 0:
                print(f"  Processed {q_idx+1}/{n_queries} queries (Elapsed: {time.perf_counter() - t_case_start:.1f}s)", flush=True)

        # Step 3: Evaluate all (k_tag, k_plot) x tw combinations
        print(f"Evaluating candidate pools across {len(k_tag_values)*len(k_plot_values)} (k_tag, k_plot) grids...", flush=True)

        case_results = {}
        for kt in k_tag_values:
            for kp in k_plot_values:
                grid_key = f"kt_{kt}_kp_{kp}"
                case_results[grid_key] = {tw: {"r50": [], "r200": [], "lat": []} for tw in tw_values}

                for qd in query_data:
                    raw_t = qd["raw_tag_10k"][:kt]
                    raw_p = qd["raw_plot_10k"][:kp]

                    if mask is not None:
                        raw_t = [m for m in raw_t if m in matching_ids]
                        raw_p = [m for m in raw_p if m in matching_ids]

                    pool = list(dict.fromkeys(raw_t + raw_p))
                    exclude = set(qd["seeds"])
                    pool = [m for m in pool if m not in exclude]

                    # Measure scoring time: tag lookup from sims_tag + chunked plot lookup
                    t0 = time.perf_counter()
                    sims_tag = qd["sims_tag"]

                    cand_pos_t = [k for k, m in enumerate(pool) if m in id_to_row]
                    c_tag = np.zeros(len(pool), dtype=np.float32)
                    c_tag_ok = np.zeros(len(pool), dtype=bool)
                    if cand_pos_t:
                        rows_t = [id_to_row[pool[k]] for k in cand_pos_t]
                        c_tag[cand_pos_t] = np.clip(sims_tag[rows_t], 0.0, 1.0)
                        c_tag_ok[cand_pos_t] = tag_norms[rows_t] > 0

                    c_plot = np.zeros(len(pool), dtype=np.float32)
                    c_plot_ok = np.zeros(len(pool), dtype=bool)
                    valid_p = [(plot_id_to_r[m], k) for k, m in enumerate(pool) if m in plot_id_to_r]
                    if valid_p:
                        valid_p.sort(key=lambda x: x[0])
                        p_rows = np.array([x[0] for x in valid_p], dtype=np.int64)
                        p_pos = np.array([x[1] for x in valid_p], dtype=np.int64)
                        c = np.asarray(qd["q"].plot_centroid, dtype=np.float32)
                        cn = np.linalg.norm(c)
                        c_unit = (c / cn) if cn > 0 else c
                        # chunked scoring
                        chunk_size = 8192
                        for ch_s in range(0, len(p_rows), chunk_size):
                            ch_e = min(ch_s + chunk_size, len(p_rows))
                            c_plot[p_pos[ch_s:ch_e]] = np.clip(plot_mem[p_rows[ch_s:ch_e]] @ c_unit, 0.0, 1.0)
                            c_plot_ok[p_pos[ch_s:ch_e]] = True

                    t_score = (time.perf_counter() - t0) * 1000.0

                    # Fusion
                    for tw in tw_values:
                        t0_f = time.perf_counter()
                        final = tw * c_tag + (1.0 - tw) * c_plot
                        final[~(c_tag_ok | c_plot_ok)] = -1.0
                        order = np.lexsort((np.asarray(pool), -final))[:200]
                        top_ids = [pool[i] for i in order if final[i] >= 0]
                        t_fuse = (time.perf_counter() - t0_f) * 1000.0

                        total_lat = qd["t_cand"] + t_score + t_fuse

                        c_top50 = set(top_ids[:50])
                        c_top200 = set(top_ids[:200])

                        r50 = len(c_top50 & qd["gt50"][tw]) / float(len(qd["gt50"][tw])) if qd["gt50"][tw] else 1.0
                        r200 = len(c_top200 & qd["gt200"][tw]) / float(len(qd["gt200"][tw])) if qd["gt200"][tw] else 1.0

                        case_results[grid_key][tw]["r50"].append(r50)
                        case_results[grid_key][tw]["r200"].append(r200)
                        case_results[grid_key][tw]["lat"].append(total_lat)

        results_all[case_code] = case_results
        print(f"Case {case_code.upper()} completed in {time.perf_counter() - t_case_start:.1f}s", flush=True)

    end_free_ram = get_free_ram_gb()
    print(f"\nSweep completed! Free RAM at end: {end_free_ram:.2f} GB", flush=True)

    # Save raw results
    out_file = Path("recall_sweep_matvec_results.json")
    # Convert lists to serializable summary
    summary_export = {}
    for case_code, grids in results_all.items():
        summary_export[case_code] = {}
        for grid_key, tw_dict in grids.items():
            summary_export[case_code][grid_key] = {}
            for tw, metric_dict in tw_dict.items():
                r50 = np.array(metric_dict["r50"]) * 100.0
                r200 = np.array(metric_dict["r200"]) * 100.0
                lats = np.array(metric_dict["lat"])
                summary_export[case_code][grid_key][str(tw)] = {
                    "r50_mean": float(np.mean(r50)),
                    "r50_p5": float(np.percentile(r50, 5)),
                    "r200_mean": float(np.mean(r200)),
                    "r200_p5": float(np.percentile(r200, 5)),
                    "lat_p50": float(np.percentile(lats, 50)),
                    "lat_p95": float(np.percentile(lats, 95)),
                }

    with open(out_file, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"Saved summary to {out_file}", flush=True)

    # Print summary tables
    print("\n" + "=" * 95, flush=True)
    print("RECALL & LATENCY SWEEP SUMMARY TABLE (FLAG ON: MATVEC TAG + CHROMA PLOT)", flush=True)
    print("=" * 95, flush=True)

    for case_code in ["a", "d", "f"]:
        print(f"\n--- CASE {case_code.upper()} ---")
        header = f"{'kt':>5} {'kp':>5} | {'tw':>4} | {'Mean R@50':>10} {'p5 R@50':>9} | {'Mean R@200':>11} {'p5 R@200':>10} | {'p50 Lat':>8} {'p95 Lat':>8}"
        print(header)
        print("-" * len(header))
        for kt in k_tag_values:
            for kp in k_plot_values:
                grid_key = f"kt_{kt}_kp_{kp}"
                for tw in [0.0, 0.5, 1.0]:
                    d = summary_export[case_code][grid_key][str(tw)]
                    print(f"{kt:5d} {kp:5d} | {tw:4.1f} | {d['r50_mean']:9.2f}% {d['r50_p5']:8.2f}% | {d['r200_mean']:10.2f}% {d['r200_p5']:9.2f}% | {d['lat_p50']:6.1f}ms {d['lat_p95']:6.1f}ms")
                if kp == 10000:
                    print("." * len(header))

if __name__ == "__main__":
    run_sweep(n_queries=50)
