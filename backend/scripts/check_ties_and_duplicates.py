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

from app.stores import artifacts

def check_ties():
    artifacts.load_all()
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_ids = np.load(artifacts.settings.artifact_path / "plot_embeddings_movie_ids.npy")

    rng = np.random.default_rng(42)
    sample_seeds = rng.choice(plot_ids, size=(10, 3), replace=False)

    print("Checking score ties and duplicate embeddings in top 200 for 10 sample queries...", flush=True)
    all_ties = []
    all_dupes = []

    for q_i, seeds in enumerate(sample_seeds):
        s_rows = [plot_id_to_r[s] for s in seeds]
        centroid = plot_mem[s_rows].mean(axis=0)
        norm = np.linalg.norm(centroid)
        c_unit = centroid / norm if norm > 0 else centroid

        sims = plot_mem @ c_unit
        for s in seeds:
            sims[plot_id_to_r[s]] = -1.0

        top200_idx = np.argpartition(-sims, 200)[:200]
        top200_sorted = top200_idx[np.argsort(-sims[top200_idx])]
        top200_scores = sims[top200_sorted]
        top200_ids = plot_ids[top200_sorted]

        # Check for exact duplicate scores
        diffs = np.diff(top200_scores)
        exact_ties = np.count_nonzero(diffs == 0.0)
        near_ties = np.count_nonzero(np.abs(diffs) < 1e-6)

        # Check for duplicate vectors in top 200
        vecs = plot_mem[top200_sorted]
        # Pairwise diffs of consecutive vectors
        v_diffs = np.linalg.norm(vecs[:-1] - vecs[1:], axis=1)
        dupe_vecs = np.count_nonzero(v_diffs < 1e-5)

        all_ties.append(exact_ties)
        all_dupes.append(dupe_vecs)
        print(f"Query {q_i+1}: exact_ties={exact_ties}, near_ties (<1e-6)={near_ties}, dupe_vecs={dupe_vecs}", flush=True)

    print(f"Average exact ties in top 200: {np.mean(all_ties):.2f}", flush=True)
    print(f"Average duplicate vectors in top 200: {np.mean(all_dupes):.2f}", flush=True)

if __name__ == "__main__":
    check_ties()
