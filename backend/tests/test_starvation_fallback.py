import unittest
from datetime import date
from unittest.mock import patch

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import candidates, query_builder
from app.stores import artifacts, chroma, meta_index


class TestStarvationFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()
        # Need tags_vectors for the ANN fallback path tested below
        chroma.get_collection(settings.plots_collection)
        chroma.get_collection(settings.tags_collection)

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_starvation_fallback_triggered(self):
        """Force ANN path on seed-language/filter-language mismatch (Case e style).
        Verify that starvation fallback triggers and returns full exact masked set."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],
            original_language=["fr"],
            date_from=date(1950, 1, 1),
            date_to=date(2024, 12, 31),
            tag_weight=0.5,
        )

        # Force the SVD/ANN path (not matvec) and lower exact_limit to force large branch
        with patch.object(settings, "use_matvec_tag_candidates", False):
            q = query_builder.build(req)
            f = candidates.Filters.from_request(req)
            mask = meta_index.mask_for(f)
            idx = meta_index.get_index()
            n_matches = int(mask.sum())
            self.assertGreater(n_matches, 50_000)

            with patch.object(settings, "exact_limit", 100):
                with patch.object(settings, "min_survivors", 1000):
                    with patch.object(settings, "starvation_hard_cap", 300_000):
                        pool = candidates.generate(req, q)

        # Due to starvation (only 30 tag survivors from Chroma for French),
        # it should fall back to exact masked set (57,612 movies minus seeds)
        expected_size = n_matches - len(set(req.movie_ids) & set(idx.ids[mask]))
        self.assertEqual(len(pool), expected_size)


if __name__ == "__main__":
    unittest.main()
