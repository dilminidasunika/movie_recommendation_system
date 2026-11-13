import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.stores import artifacts
from app.schemas import RecommendRequest
from app.services import query_builder

artifacts.load_all()
tag_m = artifacts.tag_matrix()
tag_norms = artifacts.row_norms()
tag_ids = artifacts.matrix_ids()
id_to_row = artifacts.id_to_row()

req = RecommendRequest(movie_ids=[238, 240, 278], tag_weight=1.0)
q = query_builder.build(req)

# Exact matvec
dots = (tag_m @ q.tag_sparse.T).toarray().ravel()
valid = tag_norms > 0
sims = np.zeros(len(tag_norms), dtype=np.float32)
sims[valid] = dots[valid] / tag_norms[valid]

# Exclude seeds
scores_pick = sims.copy()
for s in [238, 240, 278]:
    scores_pick[id_to_row[s]] = -1.0

# Method A: Top 200 directly from scores_pick
top200_a_idx = np.argpartition(-scores_pick, 200)[:200]
top200_a_sorted = top200_a_idx[np.argsort(-scores_pick[top200_a_idx])]
top200_a_ids = tag_ids[top200_a_sorted]
top200_a_scores = scores_pick[top200_a_sorted]

# Method B: Top 10k then take top 200
top10k_b_idx = np.argpartition(-scores_pick, 10000)[:10000]
top10k_b_sorted = top10k_b_idx[np.argsort(-scores_pick[top10k_b_idx])]
top10k_b_ids = tag_ids[top10k_b_sorted]
top10k_b_scores = scores_pick[top10k_b_sorted]

print("Overlap between Top-200 A and Top-200 of Top-10k B:", len(set(top200_a_ids) & set(top10k_b_ids[:200])))
print("Score at rank 199 in A:", top200_a_scores[199])
print("Score at rank 200 in A:", top200_a_scores[200] if len(top200_a_scores) > 200 else "N/A")
print("Min score in top 200:", np.min(top200_a_scores[:200]))
print("Max score in top 200:", np.max(top200_a_scores[:200]))

# Check how ground truth in bench_recall_sweep_matvec_fast.py was constructed:
set_tag = set(tag_ids)
set_plot = set(artifacts.plot_memmap()[1].keys())
unified_ids = np.array(sorted(list(set_tag | set_plot)))
u_has_tag = np.array([mid in id_to_row for mid in unified_ids], dtype=bool)
u_tag_rows = np.array([id_to_row[mid] if mid in id_to_row else 0 for mid in unified_ids], dtype=np.int32)
u_id_to_pos = {int(mid): i for i, mid in enumerate(unified_ids)}

u_t = np.zeros(len(unified_ids), dtype=np.float32)
u_t[u_has_tag] = np.clip(sims[u_tag_rows[u_has_tag]], 0.0, 1.0)
for s in [238, 240, 278]:
    u_t[u_id_to_pos[s]] = -1.0

gt200_idx = np.argpartition(-u_t, 200)[:200]
gt200_sorted = gt200_idx[np.argsort(-u_t[gt200_idx])]
gt200_ids = unified_ids[gt200_sorted]
gt200_scores = u_t[gt200_sorted]

print("Overlap between Method A (from tag_matrix) and Ground Truth (unified catalog):", len(set(top200_a_ids) & set(gt200_ids)))
if len(set(top200_a_ids) & set(gt200_ids)) < 200:
    diff_a_gt = set(top200_a_ids) - set(gt200_ids)
    diff_gt_a = set(gt200_ids) - set(top200_a_ids)
    print(f"Difference count: {len(diff_a_gt)}")
    sample_diff_a = list(diff_a_gt)[:5]
    sample_diff_gt = list(diff_gt_a)[:5]
    print("Scores of sample items in A not in GT:", [scores_pick[id_to_row[m]] for m in sample_diff_a])
    print("Scores of sample items in GT not in A:", [u_t[u_id_to_pos[m]] for m in sample_diff_gt])
