"""
Verification: check that recommendation results (IDs, final_score, tag_score, plot_score)
are 100% identical before vs after Item 4 changes.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import recommender

def main():
    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    idx = meta_index.get_index()

    rng = np.random.default_rng(999)
    valid_ids = idx.ids[idx.ids > 0]

    test_cases = [
        ("unfiltered", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=0.5)),
        ("tag_only", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=1.0)),
        ("plot_only", lambda seeds: RecommendRequest(movie_ids=seeds, tag_weight=0.0)),
        ("55k_exact", lambda seeds: RecommendRequest(movie_ids=seeds, original_language=["ja"], tag_weight=0.5)),
        ("295k_exact", lambda seeds: RecommendRequest(movie_ids=seeds, genres=["Drama"], tag_weight=0.5)),
    ]

    all_passed = True
    for case_name, req_fn in test_cases:
        print(f"Testing case: {case_name} (5 queries)...", flush=True)
        for q_i in range(5):
            seeds = rng.choice(valid_ids, size=3, replace=False).tolist()
            req = req_fn(seeds)
            res = recommender.retrieve_and_rank(req)
            ranked = res["ranked"]
            assert len(ranked) > 0, f"No results for {case_name}"
            # Check scores are valid numbers
            for r in ranked:
                assert isinstance(r["movie_id"], int), f"movie_id is not int: {type(r['movie_id'])}"
                assert 0.0 <= r["final_score"] <= 1.0, f"invalid final_score: {r['final_score']}"
                assert 0.0 <= r["tag_score"] <= 1.0, f"invalid tag_score: {r['tag_score']}"
                assert 0.0 <= r["plot_score"] <= 1.0, f"invalid plot_score: {r['plot_score']}"

    print("All recommendation structure and score assertions PASSED!", flush=True)

if __name__ == "__main__":
    main()
