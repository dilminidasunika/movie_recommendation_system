import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import time
import json
import sqlite3
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.stores import artifacts, chroma
from app.config import settings

def print_chroma_configs():
    print("=" * 60, flush=True)
    print("CHROMA COLLECTION CONFIGURATIONS", flush=True)
    print("=" * 60, flush=True)
    sqlite_path = os.path.join(settings.chroma_path, "chroma.sqlite3")
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("SELECT name, schema_str FROM collections")
    for name, sstr in cur.fetchall():
        cfg = json.loads(sstr) if sstr else {}
        vec_cfg = cfg.get("defaults", {}).get("float_list", {}).get("vector_index", {}).get("config", {})
        hnsw = vec_cfg.get("hnsw", {})
        space = vec_cfg.get("space", "unknown")
        print(f"Collection: {name}", flush=True)
        print(f"  space:              {space}", flush=True)
        print(f"  hnsw:search_ef:     {hnsw.get('ef_search')}", flush=True)
        print(f"  M (max_neighbors):  {hnsw.get('max_neighbors')}", flush=True)
        print(f"  construction_ef:    {hnsw.get('ef_construction')}", flush=True)
        print(f"  num_threads:        {hnsw.get('num_threads')}", flush=True)
        print(f"  sync_threshold:     {hnsw.get('sync_threshold')}", flush=True)
    conn.close()
    print("-" * 60, flush=True)

def run_per_signal_diagnostic(n_queries=50, rng_seed=42):
    print_chroma_configs()
    
    print("\nLoading artifacts...", flush=True)
    artifacts.load_all()
    
    # Pre-fetch handles
    id_to_row = artifacts.id_to_row()
    tag_m = artifacts.tag_matrix()
    tag_ids = artifacts.matrix_ids()
    tag_norms = artifacts.row_norms()
    svd_model = artifacts.svd()
    
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_ids = np.empty(len(plot_id_to_r), dtype=np.int64)
    for mid, r in plot_id_to_r.items():
        plot_ids[r] = mid
    
    print(f"Tag matrix: {tag_m.shape}, Plot memmap: {plot_mem.shape}", flush=True)
    
    # Find candidate pool of movie IDs that have BOTH tag row and plot embedding
    common_ids = np.array(list(set(id_to_row.keys()).intersection(set(plot_id_to_r.keys()))))
    common_ids.sort()
    print(f"Found {len(common_ids):,} movies with both tag and plot signals", flush=True)
    
    rng = np.random.default_rng(rng_seed)
    
    # Generate 50 queries, each using 3 random seed movies
    queries = []
    for q_idx in range(n_queries):
        seeds = rng.choice(common_ids, size=3, replace=False).tolist()
        
        # Build tag sparse query (mean of L2-normalized rows)
        s_rows = [id_to_row[s] for s in seeds]
        m = normalize(tag_m[s_rows], norm="l2")
        mean_s = sp.csr_matrix(np.full((1, len(s_rows)), 1.0 / len(s_rows))) @ m
        tag_sparse = normalize(mean_s, norm="l2")
        
        # Build tag SVD ANN vector
        red = svd_model.transform(tag_sparse)
        tag_ann = normalize(red, norm="l2")[0].astype(np.float32)
        
        # Build plot centroid query (mean of unit plot vectors, then normalized)
        p_rows = [plot_id_to_r[s] for s in seeds]
        p_vecs = plot_mem[p_rows]
        p_mean = np.mean(p_vecs, axis=0)
        p_norm = np.linalg.norm(p_mean)
        plot_centroid = (p_mean / p_norm).astype(np.float32) if p_norm > 0 else p_mean.astype(np.float32)
        
        queries.append({
            "idx": q_idx,
            "seeds": seeds,
            "tag_sparse": tag_sparse,
            "tag_ann": tag_ann,
            "plot_centroid": plot_centroid,
        })
    
    print(f"Generated {len(queries)} test queries with fixed RNG seed {rng_seed}", flush=True)
    
    # -------------------------------------------------------------
    # PLOT SIGNAL DIAGNOSTIC
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("RUNNING PLOT SIGNAL ANN RECALL DIAGNOSTIC", flush=True)
    print("=" * 60, flush=True)
    
    plot_r50_in_2k = []
    plot_r200_in_2k = []
    plot_r50_in_10k = []
    plot_r200_in_10k = []
    plot_k2k_returned = []
    plot_k10k_returned = []
    
    # Pre-cache memmap rows as a chunked dot product
    CHUNK = 100_000
    n_plot_rows = len(plot_ids)
    
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        p_vec = q["plot_centroid"]
        
        # 1. Exact top-200 by plot cosine over all 1.02M plot memmap rows
        scores = np.empty(n_plot_rows, dtype=np.float32)
        for start in range(0, n_plot_rows, CHUNK):
            end = min(start + CHUNK, n_plot_rows)
            scores[start:end] = plot_mem[start:end] @ p_vec
        
        # Exclude seed movies
        for s in q["seeds"]:
            if s in plot_id_to_r:
                scores[plot_id_to_r[s]] = -1.0
        
        top200_idx = np.argpartition(-scores, 200)[:200]
        top200_sorted_idx = top200_idx[np.argsort(-scores[top200_idx])]
        exact_plot_top50 = set(plot_ids[top200_sorted_idx[:50]])
        exact_plot_top200 = set(plot_ids[top200_sorted_idx[:200]])
        
        # 2. ANN query at k=2000
        ann_2k = chroma.query_ids(settings.plots_collection, p_vec, 2000)
        ann_2k_set = set(ann_2k) - set(q["seeds"])
        plot_k2k_returned.append(len(ann_2k))
        
        r50_2k = len(ann_2k_set.intersection(exact_plot_top50)) / 50.0
        r200_2k = len(ann_2k_set.intersection(exact_plot_top200)) / 200.0
        plot_r50_in_2k.append(r50_2k)
        plot_r200_in_2k.append(r200_2k)
        
        # 3. ANN query at k=10000
        ann_10k = chroma.query_ids(settings.plots_collection, p_vec, 10000)
        ann_10k_set = set(ann_10k) - set(q["seeds"])
        plot_k10k_returned.append(len(ann_10k))
        
        r50_10k = len(ann_10k_set.intersection(exact_plot_top50)) / 50.0
        r200_10k = len(ann_10k_set.intersection(exact_plot_top200)) / 200.0
        plot_r50_in_10k.append(r50_10k)
        plot_r200_in_10k.append(r200_10k)
        
        if (i + 1) % 10 == 0 or i == n_queries - 1:
            print(f"  [Plot] Done {i+1}/{n_queries}: mean R@200 in k=2k: {np.mean(plot_r200_in_2k):.4f}, in k=10k: {np.mean(plot_r200_in_10k):.4f}", flush=True)
            
    print(f"Plot diagnostic completed in {time.perf_counter() - t_start:.2f}s", flush=True)
    
    # -------------------------------------------------------------
    # TAG SIGNAL DIAGNOSTIC
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("RUNNING TAG SIGNAL ANN RECALL DIAGNOSTIC (SVD vs Exact Sparse)", flush=True)
    print("=" * 60, flush=True)
    
    tag_r50_in_2k = []
    tag_r200_in_2k = []
    tag_r50_in_10k = []
    tag_r200_in_10k = []
    tag_k2k_returned = []
    tag_k10k_returned = []
    
    # Also evaluate SVD exact recall (exact top-k in 300-d SVD space vs exact sparse) to isolate SVD vs HNSW!
    tag_exact_svd_r200_in_2k = []
    
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        t_sp = q["tag_sparse"]
        t_ann = q["tag_ann"]
        
        # 1. Exact top-200 by exact sparse tag cosine over all 1.06M rows
        dots = (tag_m @ t_sp.T).toarray().ravel()
        valid = tag_norms > 0
        scores = np.zeros(len(tag_norms), dtype=np.float32)
        scores[valid] = dots[valid] / tag_norms[valid]
        
        # Exclude seed movies
        for s in q["seeds"]:
            if s in id_to_row:
                scores[id_to_row[s]] = -1.0
                
        top200_idx = np.argpartition(-scores, 200)[:200]
        top200_sorted_idx = top200_idx[np.argsort(-scores[top200_idx])]
        exact_tag_top50 = set(tag_ids[top200_sorted_idx[:50]])
        exact_tag_top200 = set(tag_ids[top200_sorted_idx[:200]])
        
        # 2. ANN query at k=2000 in tags_vectors
        ann_2k = chroma.query_ids(settings.tags_collection, t_ann, 2000)
        ann_2k_set = set(ann_2k) - set(q["seeds"])
        tag_k2k_returned.append(len(ann_2k))
        
        r50_2k = len(ann_2k_set.intersection(exact_tag_top50)) / 50.0
        r200_2k = len(ann_2k_set.intersection(exact_tag_top200)) / 200.0
        tag_r50_in_2k.append(r50_2k)
        tag_r200_in_2k.append(r200_2k)
        
        # 3. ANN query at k=10000 in tags_vectors
        ann_10k = chroma.query_ids(settings.tags_collection, t_ann, 10000)
        ann_10k_set = set(ann_10k) - set(q["seeds"])
        tag_k10k_returned.append(len(ann_10k))
        
        r50_10k = len(ann_10k_set.intersection(exact_tag_top50)) / 50.0
        r200_10k = len(ann_10k_set.intersection(exact_tag_top200)) / 200.0
        tag_r50_in_10k.append(r50_10k)
        tag_r200_in_10k.append(r200_10k)
        
        if (i + 1) % 10 == 0 or i == n_queries - 1:
            print(f"  [Tag] Done {i+1}/{n_queries}: mean R@200 in k=2k: {np.mean(tag_r200_in_2k):.4f}, in k=10k: {np.mean(tag_r200_in_10k):.4f}", flush=True)
            
    print(f"Tag diagnostic completed in {time.perf_counter() - t_start:.2f}s", flush=True)
    
    # -------------------------------------------------------------
    # REPORT RESULTS
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY REPORT - PER-SIGNAL ANN RECALL DIAGNOSTIC (N=50)", flush=True)
    print("=" * 60, flush=True)
    
    print("\n[PLOT SIGNAL] (Dense 384-d Cosine vs Exact Memmap 1.02M)")
    print(f"  ANN k=2000 return count:  min={np.min(plot_k2k_returned)}, max={np.max(plot_k2k_returned)} (Target: 2000)")
    print(f"  ANN k=10000 return count: min={np.min(plot_k10k_returned)}, max={np.max(plot_k10k_returned)} (Target: 10000)")
    print(f"  Recall@50  in k=2000:   Mean={np.mean(plot_r50_in_2k)*100:.2f}%, p5={np.percentile(plot_r50_in_2k, 5)*100:.2f}%")
    print(f"  Recall@200 in k=2000:   Mean={np.mean(plot_r200_in_2k)*100:.2f}%, p5={np.percentile(plot_r200_in_2k, 5)*100:.2f}%")
    print(f"  Recall@50  in k=10000:  Mean={np.mean(plot_r50_in_10k)*100:.2f}%, p5={np.percentile(plot_r50_in_10k, 5)*100:.2f}%")
    print(f"  Recall@200 in k=10000:  Mean={np.mean(plot_r200_in_10k)*100:.2f}%, p5={np.percentile(plot_r200_in_10k, 5)*100:.2f}%")
    
    print("\n[TAG SIGNAL] (300-d SVD Chroma ANN vs Exact Sparse Cosine 1.06M)")
    print(f"  ANN k=2000 return count:  min={np.min(tag_k2k_returned)}, max={np.max(tag_k2k_returned)} (Target: 2000)")
    print(f"  ANN k=10000 return count: min={np.min(tag_k10k_returned)}, max={np.max(tag_k10k_returned)} (Target: 10000)")
    print(f"  Recall@50  in k=2000:   Mean={np.mean(tag_r50_in_2k)*100:.2f}%, p5={np.percentile(tag_r50_in_2k, 5)*100:.2f}%")
    print(f"  Recall@200 in k=2000:   Mean={np.mean(tag_r200_in_2k)*100:.2f}%, p5={np.percentile(tag_r200_in_2k, 5)*100:.2f}%")
    print(f"  Recall@50  in k=10000:  Mean={np.mean(tag_r50_in_10k)*100:.2f}%, p5={np.percentile(tag_r50_in_10k, 5)*100:.2f}%")
    print(f"  Recall@200 in k=10000:  Mean={np.mean(tag_r200_in_10k)*100:.2f}%, p5={np.percentile(tag_r200_in_10k, 5)*100:.2f}%")
    
    # Save results to json for reporting
    res = {
        "plot": {
            "r50_2k_mean": float(np.mean(plot_r50_in_2k)),
            "r50_2k_p5": float(np.percentile(plot_r50_in_2k, 5)),
            "r200_2k_mean": float(np.mean(plot_r200_in_2k)),
            "r200_2k_p5": float(np.percentile(plot_r200_in_2k, 5)),
            "r50_10k_mean": float(np.mean(plot_r50_in_10k)),
            "r50_10k_p5": float(np.percentile(plot_r50_in_10k, 5)),
            "r200_10k_mean": float(np.mean(plot_r200_in_10k)),
            "r200_10k_p5": float(np.percentile(plot_r200_in_10k, 5)),
        },
        "tag": {
            "r50_2k_mean": float(np.mean(tag_r50_in_2k)),
            "r50_2k_p5": float(np.percentile(tag_r50_in_2k, 5)),
            "r200_2k_mean": float(np.mean(tag_r200_in_2k)),
            "r200_2k_p5": float(np.percentile(tag_r200_in_2k, 5)),
            "r50_10k_mean": float(np.mean(tag_r50_in_10k)),
            "r50_10k_p5": float(np.percentile(tag_r50_in_10k, 5)),
            "r200_10k_mean": float(np.mean(tag_r200_in_10k)),
            "r200_10k_p5": float(np.percentile(tag_r200_in_10k, 5)),
        }
    }
    with open("signal_recall_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nResults saved to signal_recall_results.json", flush=True)

if __name__ == "__main__":
    run_per_signal_diagnostic(n_queries=50, rng_seed=42)
