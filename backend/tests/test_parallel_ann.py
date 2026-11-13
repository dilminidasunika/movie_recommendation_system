from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from app.core.pool import POOL
from app.services import candidates


class TestParallelANN(unittest.TestCase):
    def setUp(self):
        self.mock_q = SimpleNamespace(
            tag_ann=[0.1] * 300,
            plot_centroid=[0.2] * 384,
            movie_ids=[1, 2, 3],
        )

    @patch("app.stores.chroma.query_ids")
    def test_ordering(self, mock_query):
        """Verify that tag results are returned first and plot results second."""
        def side_effect(col_name, vec, k, where=None):
            if "tags" in col_name:
                return [101, 102, 103]
            elif "plots" in col_name:
                return [201, 202, 203]
            return []

        mock_query.side_effect = side_effect
        timings = {}
        lists = candidates._ann(self.mock_q, where=None, timings=timings)

        self.assertEqual(len(lists), 2)
        self.assertEqual(lists[0], [101, 102, 103])  # Tag first
        self.assertEqual(lists[1], [201, 202, 203])  # Plot second
        self.assertIn("ann_tag", timings)
        self.assertIn("ann_plot", timings)

    @patch("app.core.pool.POOL.submit")
    @patch("app.stores.chroma.query_ids")
    def test_single_signal_inline_tag_only(self, mock_query, mock_submit):
        """Verify single signal (tag only) runs inline without submitting to POOL."""
        q_tag_only = SimpleNamespace(
            tag_ann=[0.1] * 300,
            plot_centroid=None,
            movie_ids=[1],
        )
        mock_query.return_value = [11, 12, 13]

        timings = {}
        lists = candidates._ann(q_tag_only, where=None, timings=timings)

        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0], [11, 12, 13])
        mock_submit.assert_not_called()  # Inline execution
        self.assertIn("ann_tag", timings)
        self.assertNotIn("ann_plot", timings)

    @patch("app.core.pool.POOL.submit")
    @patch("app.stores.chroma.query_ids")
    def test_single_signal_inline_plot_only(self, mock_query, mock_submit):
        """Verify single signal (plot only) runs inline without submitting to POOL."""
        q_plot_only = SimpleNamespace(
            tag_ann=None,
            plot_centroid=[0.2] * 384,
            movie_ids=[1],
        )
        mock_query.return_value = [21, 22, 23]

        timings = {}
        lists = candidates._ann(q_plot_only, where=None, timings=timings)

        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0], [21, 22, 23])
        mock_submit.assert_not_called()  # Inline execution
        self.assertIn("ann_plot", timings)
        self.assertNotIn("ann_tag", timings)

    @patch("app.stores.chroma.query_ids")
    def test_exception_propagation_tag_failure(self, mock_query):
        """Verify exceptions from tag worker propagate out."""
        def side_effect(col_name, vec, k, where=None):
            if "tags" in col_name:
                raise RuntimeError("Chroma tag index error")
            return [201, 202]

        mock_query.side_effect = side_effect
        with self.assertRaises(RuntimeError) as ctx:
            candidates._ann(self.mock_q, where=None)
        self.assertIn("Chroma tag index error", str(ctx.exception))

    @patch("app.stores.chroma.query_ids")
    def test_exception_propagation_plot_failure(self, mock_query):
        """Verify exceptions from plot worker propagate out."""
        def side_effect(col_name, vec, k, where=None):
            if "plots" in col_name:
                raise RuntimeError("Chroma plot index error")
            return [101, 102]

        mock_query.side_effect = side_effect
        with self.assertRaises(RuntimeError) as ctx:
            candidates._ann(self.mock_q, where=None)
        self.assertIn("Chroma plot index error", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
