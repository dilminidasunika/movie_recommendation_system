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

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import query_builder, scoring

def test_equivalence():
    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    idx = meta_index.get_index()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()

    rng = np.random.default_rng(42)
    sample_seeds = rng.choice(tag_ids, size=3, replace=False).tolist()
    req = RecommendRequest(movie_ids=sample_seeds, tag_weight=0.5)
    q = query_builder.build(req)

    # 1. Full matvec
    dots_full = (tag_m @ q.tag_sparse.T).toarray().ravel()
    valid_full = tag_norms > 0
    full_sims = np.zeros(len(tag_norms), dtype=np.float32)
    full_sims[valid_full] = dots_full[valid_full] / tag_norms[valid_full]

    # Test on a pool of 2000 IDs and 55000 IDs
    for pool_size in [10, 2000, 55000]:
        pool_ids = rng.choice(tag_ids, size=pool_size, replace=False).tolist()

        # Existing scoring method
        s_old = scoring.score(pool_ids, q)

        # New method reusing full_sims
        pos = [k for k, mid in enumerate(pool_ids) if mid in id_to_row]
        rows = [id_to_row[pool_ids[k]] for k in pos]
        new_tag = np.zeros(len(pool_ids), dtype=np.float32)
        new_valid = np.zeros(len(pool_ids), dtype=bool)
        if pos:
            pos_arr = np.asarray(pos)
            new_tag[pos_arr] = full_sims[rows]
            new_valid[pos_arr] = tag_norms[rows] > 0

        diff = np.abs(s_old.tag - new_tag)
        max_diff = np.max(diff)
        print(f"Pool size {pool_size:,}: max tag score difference = {max_diff:.2e}")
        assert max_diff < 1e-6, f"Max diff {max_diff} exceeds 1e-6 tolerance!"
        assert np.array_equal(s_old.tag_ok, new_valid), "tag_ok mismatch!"
    print("ALL SCORE EQUIVALENCE CHECKS PASSED (diff < 1e-6)!")

if __name__ == "__main__":
    test_equivalence()
