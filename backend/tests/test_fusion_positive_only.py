import unittest
from types import SimpleNamespace
import numpy as np

from app.services import fusion


class TestFusionPositiveOnly(unittest.TestCase):
    def test_tag_weight_0_excludes_tag_only_movies(self):
        """At tag_weight 0.0, tag-only movies have final_score == 0.0 and must NOT be returned."""
        # Movie 1: dual signal (plot=0.8, tag=0.5)
        # Movie 2: tag-only (plot_ok=False, tag=0.9)
        # Movie 3: plot-only (tag_ok=False, plot=0.7)
        # Movie 4: neither signal
        s = SimpleNamespace(
            ids=np.array([1, 2, 3, 4], dtype=np.int64),
            tag=np.array([0.5, 0.9, 0.0, 0.0], dtype=np.float32),
            tag_ok=np.array([True, True, False, False]),
            plot=np.array([0.8, 0.0, 0.7, 0.0], dtype=np.float32),
            plot_ok=np.array([True, False, True, False]),
        )

        res = fusion.fuse(s, tag_weight=0.0, top_n=10)
        returned_ids = [r["movie_id"] for r in res]

        # Movie 2 is tag-only, must NOT be returned
        self.assertNotIn(2, returned_ids)
        # Movie 4 has neither signal, must NOT be returned
        self.assertNotIn(4, returned_ids)
        # Movies 1 and 3 have plot signal (> 0), must be returned
        self.assertIn(1, returned_ids)
        self.assertIn(3, returned_ids)
        self.assertEqual(len(res), 2)

        # All returned items must have strictly positive final_score
        for r in res:
            self.assertGreater(r["final_score"], 0.0)

    def test_tag_weight_1_excludes_plot_only_movies(self):
        """At tag_weight 1.0, plot-only movies have final_score == 0.0 and must NOT be returned."""
        # Movie 1: dual signal (tag=0.6, plot=0.8)
        # Movie 2: tag-only (tag=0.9, plot_ok=False)
        # Movie 3: plot-only (tag_ok=False, plot=0.95)
        # Movie 4: neither signal
        s = SimpleNamespace(
            ids=np.array([1, 2, 3, 4], dtype=np.int64),
            tag=np.array([0.6, 0.9, 0.0, 0.0], dtype=np.float32),
            tag_ok=np.array([True, True, False, False]),
            plot=np.array([0.8, 0.0, 0.95, 0.0], dtype=np.float32),
            plot_ok=np.array([True, False, True, False]),
        )

        res = fusion.fuse(s, tag_weight=1.0, top_n=10)
        returned_ids = [r["movie_id"] for r in res]

        # Movie 3 is plot-only, must NOT be returned
        self.assertNotIn(3, returned_ids)
        # Movie 4 has neither signal, must NOT be returned
        self.assertNotIn(4, returned_ids)
        # Movies 1 and 2 have tag signal (> 0), must be returned
        self.assertIn(1, returned_ids)
        self.assertIn(2, returned_ids)
        self.assertEqual(len(res), 2)

        for r in res:
            self.assertGreater(r["final_score"], 0.0)

    def test_list_shorter_than_top_n_only_when_fewer_positive_candidates(self):
        """The list may be shorter than top_n only when fewer positive-score candidates exist."""
        # Create 100 candidates: 15 positive, 85 zero/negative
        n = 100
        ids = np.arange(1, n + 1, dtype=np.int64)
        tag = np.zeros(n, dtype=np.float32)
        plot = np.zeros(n, dtype=np.float32)
        tag_ok = np.zeros(n, dtype=bool)
        plot_ok = np.zeros(n, dtype=bool)

        # 15 positive candidates
        tag[:15] = np.linspace(0.1, 0.9, 15)
        tag_ok[:15] = True
        plot[:15] = np.linspace(0.1, 0.9, 15)
        plot_ok[:15] = True

        # Rest have tag=0 or tag_ok=False
        tag_ok[15:50] = True  # score 0.0
        plot_ok[50:80] = True  # score 0.0

        s = SimpleNamespace(
            ids=ids,
            tag=tag,
            tag_ok=tag_ok,
            plot=plot,
            plot_ok=plot_ok,
        )

        # When top_n=50 (> 15 positive candidates), result must have length 15
        res_50 = fusion.fuse(s, tag_weight=0.5, top_n=50)
        self.assertEqual(len(res_50), 15)
        for r in res_50:
            self.assertGreater(r["final_score"], 0.0)

        # When top_n=10 (< 15 positive candidates), result must have exactly top_n = 10
        res_10 = fusion.fuse(s, tag_weight=0.5, top_n=10)
        self.assertEqual(len(res_10), 10)
        for r in res_10:
            self.assertGreater(r["final_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
