import unittest
import numpy as np

from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest, TagIn
from app.services import candidates, query_builder
from app.stores import artifacts, meta_index


class TestPositiveSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_rare_chip_query_no_cap_warning_and_bounded_pool(self):
        """A chips-only query with a rare director (fewer than k movies with overlap)
        returns a pool no larger than the number of positive-similarity movies
        and logs no cap warning."""
        req = RecommendRequest(
            tags=[TagIn(tag_type="director", name="Aarne Tarkas")],
            tag_weight=1.0,
        )
        q = query_builder.build(req)

        self.assertIsNotNone(q.tag_sparse)

        with self.assertLogs("app.services.candidates", level="WARNING") as cm:
            # Add a dummy warning so assertLogs doesn't fail if no warnings are logged
            import logging
            logging.getLogger("app.services.candidates").warning("DUMMY_GUARD_WARNING")
            pool = candidates.generate(req, q)

        # Check logs: MUST NOT contain "exceeded safety cap"
        cap_warnings = [m for m in cm.output if "exceeded safety cap" in m]
        self.assertEqual(
            cap_warnings,
            [],
            f"Cap warning was unexpectedly logged: {cap_warnings}",
        )

        # Count positive-similarity movies
        sims = q.tag_full_sims
        self.assertIsNotNone(sims)
        n_pos = int(np.count_nonzero(sims > 0))

        # Must be fewer than k_tag (2000)
        self.assertLess(n_pos, 2000)
        self.assertGreater(n_pos, 0)

        # Pool size must be <= n_pos
        self.assertLessEqual(len(pool), n_pos)
        self.assertGreater(len(pool), 0)

    def test_matvec_tag_strictly_positive_similarity(self):
        """Direct test on _matvec_tag to ensure kth <= 0 never includes 0-score ties."""
        req = RecommendRequest(
            tags=[TagIn(tag_type="director", name="Aarne Tarkas")],
            tag_weight=1.0,
        )
        q = query_builder.build(req)
        cands, dt = candidates._matvec_tag(q, k=2000, exclude=set())

        sims = q.tag_full_sims
        n_pos = int(np.count_nonzero(sims > 0))

        self.assertLess(n_pos, 2000)
        self.assertEqual(len(cands), n_pos)
        # All returned candidates must have sims > 0
        id_to_row = artifacts.id_to_row()
        cand_rows = id_to_row.rows_for(cands)
        self.assertTrue(np.all(sims[cand_rows] > 0))


if __name__ == "__main__":
    unittest.main()
