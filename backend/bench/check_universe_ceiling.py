import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.db.pool import open_pool
from app.stores import artifacts, chroma, meta_index

def check():
    print("Loading artifacts and index...", flush=True)
    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    tag_m = artifacts.tag_matrix()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_ids = np.load(settings.artifact_path / "plot_embeddings_movie_ids.npy")
    idx = meta_index.get_index()

    print(f"MetaIndex total ids: {len(idx.ids):,}", flush=True)
    print(f"Tag matrix ids: {len(tag_ids):,}", flush=True)
    print(f"Plot memmap ids: {len(plot_ids):,}", flush=True)

    # Check Chroma plots collection count and IDs
    coll_plot = chroma.get_collection(settings.plots_collection)
    coll_plot_count = coll_plot.count()
    print(f"Chroma plots_embeddings count: {coll_plot_count:,}", flush=True)

    coll_tag = chroma.get_collection(settings.tags_collection)
    coll_tag_count = coll_tag.count()
    print(f"Chroma tags_vectors count: {coll_tag_count:,}", flush=True)

    # Check ID sets
    set_meta = set(idx.ids)
    set_tag = set(tag_ids)
    set_plot = set(plot_ids)

    print("\n--- Set Comparisons ---", flush=True)
    print(f"Tag & Plot intersection: {len(set_tag & set_plot):,}", flush=True)
    print(f"Plot only (in plot, not in tag): {len(set_plot - set_tag):,}", flush=True)
    print(f"Tag only (in tag, not in plot): {len(set_tag - set_plot):,}", flush=True)
    print(f"Union (in either tag or plot): {len(set_tag | set_plot):,}", flush=True)
    print(f"MetaIndex vs Union difference (Meta - Union): {len(set_meta - (set_tag | set_plot)):,}", flush=True)
    print(f"Union - Meta: {len((set_tag | set_plot) - set_meta):,}", flush=True)

    # Verify if Chroma plots IDs match plot_ids
    # We can inspect Chroma IDs using coll_plot.get(limit=10)
    sample_chroma_plot = coll_plot.get(limit=1000, include=[])["ids"]
    sample_chroma_plot_ints = {int(x) for x in sample_chroma_plot}
    missing_in_plot_mem = sample_chroma_plot_ints - set_plot
    print(f"Sample 1000 Chroma plot IDs missing in plot memmap: {len(missing_in_plot_mem)}", flush=True)

    # Check if coll_plot count == len(plot_ids)
    if coll_plot_count == len(plot_ids):
        print("Chroma plots_embeddings count exactly matches plot_embeddings_movie_ids.npy count!", flush=True)
    else:
        print(f"DISCREPANCY: Chroma count ({coll_plot_count}) != plot memmap count ({len(plot_ids)})", flush=True)

    # Now verify the ceiling explanation with a direct experiment:
    # Compute ground truth for tw=0.0 over FULL plot_ids (1,023,484) vs over common_ids (915,185)
    # for 20 sample queries and check where top 200 candidates fall!
    rng = np.random.default_rng(42)
    sample_seeds = rng.choice(plot_ids, size=(10, 3), replace=False)

    print("\n--- Testing tw=0.0 Recall Against Full Plot Universe vs Common IDs ---", flush=True)
    for q_i, seeds in enumerate(sample_seeds[:5]):
        # plot centroid
        s_rows = [plot_id_to_r[s] for s in seeds]
        centroid = plot_mem[s_rows].mean(axis=0)
        norm = np.linalg.norm(centroid)
        c_unit = centroid / norm if norm > 0 else centroid

        # ANN from Chroma
        raw_ann = chroma.query_ids(settings.plots_collection, c_unit, 2000)
        raw_ann_filtered = [m for m in raw_ann if m not in set(seeds)]
        top200_ann = set(raw_ann_filtered[:200])

        # Exact plot cosine over FULL plot_mem (1,023,484)
        all_sims = plot_mem @ c_unit
        for s in seeds:
            all_sims[plot_id_to_r[s]] = -1.0
        gt200_full_idx = np.argpartition(-all_sims, 200)[:200]
        gt200_full_sorted = gt200_full_idx[np.argsort(-all_sims[gt200_full_idx])]
        gt200_full = set(plot_ids[gt200_full_sorted])

        # Exact plot cosine over common_ids (915,185)
        common_sims = all_sims.copy()
        # mask out non-common
        plot_only_mids = set_plot - set_tag
        for m in plot_only_mids:
            common_sims[plot_id_to_r[m]] = -1.0
        gt200_common_idx = np.argpartition(-common_sims, 200)[:200]
        gt200_common_sorted = gt200_common_idx[np.argsort(-common_sims[gt200_common_idx])]
        gt200_common = set(plot_ids[gt200_common_sorted])

        # Compare recall
        r_against_full = len(top200_ann & gt200_full) / 200.0
        r_against_common = len(top200_ann & gt200_common) / 200.0
        plot_only_in_gt_full = len(gt200_full & plot_only_mids)

        print(f"Query {q_i+1}:")
        print(f"  Recall@200 against FULL plot universe (1.02M): {r_against_full*100:.1f}%")
        print(f"  Recall@200 against COMMON IDs universe (915k):  {r_against_common*100:.1f}%")
        print(f"  Number of plot-only movies in GT-full:          {plot_only_in_gt_full}/200 ({plot_only_in_gt_full/200*100:.1f}%)")

if __name__ == "__main__":
    check()
