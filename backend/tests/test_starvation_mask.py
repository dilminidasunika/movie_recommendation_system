import unittest
from datetime import date
from unittest.mock import patch
import numpy as np

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import candidates, query_builder
from app.stores import artifacts, chroma, meta_index


class TestStarvationMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()
        chroma.get_collection(settings.plots_collection)

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_tag_to_meta_row_integrity(self):
        """Verify precomputed tag_to_meta_row is int32 and correctly aligned."""
        t2m = meta_index.tag_to_meta_row()
        tag_ids = artifacts.matrix_ids()
        meta_idx = meta_index.get_index()

        self.assertIsInstance(t2m, np.ndarray)
        self.assertEqual(t2m.dtype, np.int32)
        self.assertEqual(len(t2m), len(tag_ids))

        # Check random sample of rows
        sample_indices = np.random.choice(len(t2m), size=50, replace=False)
        for r in sample_indices:
            m_row = t2m[r]
            if m_row >= 0:
                self.assertEqual(tag_ids[r], meta_idx.ids[m_row])

    def test_matvec_tag_with_mask(self):
        """_matvec_tag with mask selects candidates strictly matching the mask."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],  # English movies
            original_language=["fr"],
            tag_weight=0.5,
        )
        q = query_builder.build(req)
        f = candidates.Filters.from_request(req)
        mask = meta_index.mask_for(f)
        idx = meta_index.get_index()
        fr_code = idx.lang_to_code["fr"]

        # Directly call _matvec_tag with mask
        cands, dt = candidates._matvec_tag(q, k=2000, exclude=set(req.movie_ids), mask=mask)
        self.assertGreater(len(cands), 0)

        # Every candidate must be a French movie
        cand_meta_rows = idx.id_to_row.rows_for(cands)
        self.assertTrue(np.all(cand_meta_rows >= 0))
        self.assertTrue(np.all(idx.languages[cand_meta_rows] == fr_code))

    def test_starvation_large_branch_e2e(self):
        """Seeds from one language (en), filter for another (fr) matching > exact_limit movies:
        pool has >= min(200, n_matches) candidates and every returned movie satisfies the filter."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],  # English movies
            original_language=["fr"],
            date_from=date(1950, 1, 1),
            date_to=date(2024, 12, 31),
            tag_weight=0.5,
        )
        q = query_builder.build(req)
        f = candidates.Filters.from_request(req)
        mask = meta_index.mask_for(f)
        idx = meta_index.get_index()
        n_matches = int(mask.sum())
        fr_code = idx.lang_to_code["fr"]

        self.assertGreater(n_matches, 200)

        # Lower exact_limit to force into large branch (n_matches > exact_limit)
        with patch.object(settings, "exact_limit", 5000):
            with patch.object(settings, "use_matvec_tag_candidates", True):
                self.assertGreater(n_matches, settings.exact_limit)
                pool = candidates.generate(req, q)

        # 1. Pool must have >= min(200, n_matches) candidates
        self.assertGreaterEqual(len(pool), min(200, n_matches))

        # 2. Every returned movie satisfies the filter
        pool_meta_rows = idx.id_to_row.rows_for(pool)
        self.assertTrue(np.all(pool_meta_rows >= 0))
        self.assertTrue(np.all(idx.languages[pool_meta_rows] == fr_code))

        # 3. Excluded seeds must not be in pool
        for seed in req.movie_ids:
            self.assertNotIn(seed, pool)

    def test_plot_retry_when_survivors_below_min_survivors(self):
        """When plot survivors < min_survivors, Chroma is retried with 4x k_plot."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],
            original_language=["fr"],
            tag_weight=0.5,
        )
        q = query_builder.build(req)

        # Force large branch and ensure min_survivors triggers retry
        with patch.object(settings, "exact_limit", 100):
            with patch.object(settings, "k_plot", 500):
                with patch.object(settings, "min_survivors", 1000):
                    with patch("app.stores.chroma.query_ids", wraps=chroma.query_ids) as mock_chroma:
                        pool = candidates.generate(req, q)
                        # Chroma should be called twice (initial + retry with 4x k_plot = 2000)
                        self.assertEqual(mock_chroma.call_count, 2)
                        # First call: k=500, second call: k=2000
                        self.assertEqual(mock_chroma.call_args_list[0][0][2], 500)
                        self.assertEqual(mock_chroma.call_args_list[1][0][2], 2000)


if __name__ == "__main__":
    unittest.main()
