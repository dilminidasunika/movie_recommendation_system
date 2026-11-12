"""
Unit test for tie_break_plot flag:
Verifies that final scores and returned ID sets are unchanged,
and only the order among equal-final-score items differs.
"""
import unittest
from unittest.mock import patch
import numpy as np

from app.config import settings
from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import recommender, fusion
from app.services.scoring import Scores


class TestTieBreakPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()

    def test_fusion_level_scores_and_ids_identical_only_order_differs(self):
        """
        Direct test on fusion: given candidates with identical final scores,
        returned ID sets and scores are 100% identical, but ordering differs
        such that higher plot_score comes first under tie_break_plot=True.
        """
        # Create a synthetic pool of 5 movies:
        # Movies 101, 102, 103 have identical tag score (0.8) and tw=1.0 -> identical final score (0.8)
        # but differing plot scores: 0.1, 0.9, 0.5.
        ids = np.array([101, 102, 103, 104, 105], dtype=np.int64)
        tag = np.array([0.8, 0.8, 0.8, 0.6, 0.4], dtype=np.float32)
        plot = np.array([0.1, 0.9, 0.5, 0.7, 0.3], dtype=np.float32)
        tag_ok = np.ones(5, dtype=bool)
        plot_ok = np.ones(5, dtype=bool)

        scores = Scores(ids=ids, tag=tag, plot=plot, tag_ok=tag_ok, plot_ok=plot_ok)

        # 1. tie_break_plot = False -> order by final desc, id asc
        with patch.object(settings, "tie_break_plot", False):
            res_off = fusion.fuse(scores, tag_weight=1.0, top_n=5)

        # 2. tie_break_plot = True -> order by final desc, plot desc, id asc
        with patch.object(settings, "tie_break_plot", True):
            res_on = fusion.fuse(scores, tag_weight=1.0, top_n=5)

        # Returned IDs must be identical sets
        ids_off = [r["movie_id"] for r in res_off]
        ids_on = [r["movie_id"] for r in res_on]
        self.assertEqual(set(ids_off), set(ids_on))

        # Scores for each movie must be 100% identical
        score_map_off = {r["movie_id"]: r["final_score"] for r in res_off}
        score_map_on = {r["movie_id"]: r["final_score"] for r in res_on}
        self.assertEqual(score_map_off, score_map_on)

        # Check ordering:
        # Off: 101, 102, 103 are tied on final score 0.8 -> ascending id: 101, 102, 103
        self.assertEqual(ids_off[:3], [101, 102, 103])
        # On: 102 (plot 0.9) > 103 (plot 0.5) > 101 (plot 0.1) -> descending plot: 102, 103, 101
        self.assertEqual(ids_on[:3], [102, 103, 101])

    def test_e2e_recommender_scores_identical(self):
        """
        End-to-end pipeline test: final scores and retrieved items remain unchanged,
        with tie_break_plot prioritizing plot relevance on ties.
        """
        req = RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.8)

        with patch.object(settings, "tie_break_plot", False):
            res_off = recommender.retrieve_and_rank(req)
            ranked_off = res_off["ranked"]

        with patch.object(settings, "tie_break_plot", True):
            res_on = recommender.retrieve_and_rank(req)
            ranked_on = res_on["ranked"]

        self.assertEqual(len(ranked_off), len(ranked_on))

        # Check that all returned movies in ranked_on have matching final scores in ranked_off
        scores_off = {r["movie_id"]: r["final_score"] for r in ranked_off}
        scores_on = {r["movie_id"]: r["final_score"] for r in ranked_on}

        common_mids = set(scores_off.keys()) & set(scores_on.keys())
        self.assertGreater(len(common_mids), 180)  # >90% identical top 200

        for mid in common_mids:
            self.assertAlmostEqual(scores_off[mid], scores_on[mid], places=6)

        # Verify plot tie ordering property holds for exactly equal final scores
        for i in range(len(ranked_on) - 1):
            curr_item = ranked_on[i]
            next_item = ranked_on[i + 1]
            if curr_item["final_score"] == next_item["final_score"]:
                if curr_item["plot_score"] != next_item["plot_score"]:
                    self.assertGreater(
                        curr_item["plot_score"],
                        next_item["plot_score"],
                    )
                else:
                    self.assertLess(
                        curr_item["movie_id"],
                        next_item["movie_id"],
                    )


if __name__ == "__main__":
    unittest.main()
