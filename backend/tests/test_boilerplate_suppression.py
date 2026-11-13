"""
Unit tests for boilerplate plot suppression (suppress_boilerplate_plots flag).

Tests:
1. plot_suppressed mask loads and has correct shape
2. Suppressed movies get plot_ok=False in scoring
3. Seeds 1108020, 470285, 156627 top 10 no longer dominated by
   "Mexican feature film" movies when suppression is enabled
"""
import unittest
from unittest.mock import patch
import numpy as np

from app.config import settings
from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import recommender
from app.services.scoring import score, Scores


class TestBoilerplateSuppression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()

    def test_plot_suppressed_shape_and_dtype(self):
        """plot_suppressed mask aligns with plot memmap rows."""
        ps = artifacts.plot_suppressed()
        emb, _ = artifacts.plot_memmap()
        self.assertEqual(ps.shape[0], emb.shape[0])
        self.assertEqual(ps.dtype, np.bool_)
        self.assertGreater(ps.sum(), 0, "No rows suppressed — mask may be empty")

    def test_suppressed_movies_get_plot_ok_false(self):
        """Movies whose plot rows are suppressed must have plot_ok=False."""
        ps = artifacts.plot_suppressed()
        suppressed_rows = np.flatnonzero(ps)
        if len(suppressed_rows) == 0:
            self.skipTest("No suppressed rows")

        # Pick up to 100 suppressed movie IDs
        plot_id_arr = artifacts.plot_ids()
        sample_rows = suppressed_rows[:100]
        sample_ids = plot_id_arr[sample_rows].tolist()

        # Build a dummy query with a valid plot centroid
        req = RecommendRequest(
            movie_ids=[sample_ids[0]], tags=[], plot_text=None, tag_weight=0.5
        )
        from app.services import query_builder
        with patch.object(settings, "suppress_boilerplate_plots", True):
            q = query_builder.build(req)
            if q.plot_centroid is None:
                self.skipTest("No plot centroid for this seed")
            scores = score(sample_ids, q)

        # Every suppressed movie should have plot_ok=False
        for mid in sample_ids:
            idx = scores.index(mid)
            self.assertFalse(
                scores.plot_ok[idx],
                f"Movie {mid} is suppressed but has plot_ok=True"
            )

    def test_suppressed_movies_get_plot_ok_true_when_flag_off(self):
        """With flag off, same movies should have plot_ok=True (if they have embeddings)."""
        ps = artifacts.plot_suppressed()
        suppressed_rows = np.flatnonzero(ps)
        if len(suppressed_rows) == 0:
            self.skipTest("No suppressed rows")

        plot_id_arr = artifacts.plot_ids()
        sample_rows = suppressed_rows[:100]
        sample_ids = plot_id_arr[sample_rows].tolist()

        req = RecommendRequest(
            movie_ids=[sample_ids[0]], tags=[], plot_text=None, tag_weight=0.5
        )
        from app.services import query_builder
        with patch.object(settings, "suppress_boilerplate_plots", False):
            q = query_builder.build(req)
            if q.plot_centroid is None:
                self.skipTest("No plot centroid for this seed")
            scores = score(sample_ids, q)

        # With flag off, suppressed movies should still have plot_ok=True
        ok_count = sum(1 for mid in sample_ids if scores.plot_ok[scores.index(mid)])
        self.assertGreater(ok_count, 0, "No movies have plot_ok=True even with flag off")

    def test_problem_seeds_no_mexican_feature_film_dominance(self):
        """
        Seeds [1108020, 470285, 156627]: with suppression on, top 10 should
        NOT be dominated by identical/boilerplate 'Mexican feature film' plots.
        """
        ps = artifacts.plot_suppressed()
        plot_idx = artifacts.plot_index()

        # Test combined query (Round 4 Q16) where 470285 was corrupting the plot centroid
        req = RecommendRequest(
            movie_ids=[1108020, 470285, 156627], tags=[], plot_text=None, tag_weight=0.0
        )
        with patch.object(settings, "suppress_boilerplate_plots", False):
            res_off = recommender.retrieve_and_rank(req)
        with patch.object(settings, "suppress_boilerplate_plots", True):
            res_on = recommender.retrieve_and_rank(req)

        # In res_off, top 10 are filled with the 0.699240 boilerplate movies
        top10_off_ids = [r["movie_id"] for r in res_off["ranked"][:10]]
        suppressed_in_off = sum(1 for mid in top10_off_ids if (row := plot_idx.get(mid)) is not None and ps[row])
        self.assertGreaterEqual(suppressed_in_off, 5, "Baseline should contain boilerplate movies")

        # In res_on, top 10 must contain zero boilerplate movies
        top10_on_ids = [r["movie_id"] for r in res_on["ranked"][:10]]
        suppressed_in_on = sum(1 for mid in top10_on_ids if (row := plot_idx.get(mid)) is not None and ps[row])
        self.assertEqual(suppressed_in_on, 0, f"Expected 0 suppressed movies in top 10, got {suppressed_in_on}")

        # Seed 470285 should be excluded from plot centroid and reported in warnings
        self.assertTrue(any("470285" in w for w in res_on["warnings"]), "Seed 470285 not warned as missing plot")

        # Also test seed 470285 directly
        req_single = RecommendRequest(
            movie_ids=[470285], tags=[], plot_text=None, tag_weight=0.5
        )
        with patch.object(settings, "suppress_boilerplate_plots", True):
            res_single_on = recommender.retrieve_and_rank(req_single)
        top10_single_on = [r["movie_id"] for r in res_single_on["ranked"][:10]]
        suppressed_in_single = sum(1 for mid in top10_single_on if (row := plot_idx.get(mid)) is not None and ps[row])
        self.assertEqual(suppressed_in_single, 0)


if __name__ == "__main__":
    unittest.main()
