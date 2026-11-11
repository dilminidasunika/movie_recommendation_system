import unittest
import numpy as np
from types import SimpleNamespace

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.stores import artifacts, meta_index, chroma
from app.services import query_builder, candidates, scoring
from app.schemas import RecommendRequest


class TestMatvecTagScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_matvec_score_equivalence_2000_and_55000(self):
        """Verify reused full matvec tag scores match CSR slicing within 1e-6."""
        tag_m = artifacts.tag_matrix()
        tag_norms = artifacts.row_norms()
        tag_ids = artifacts.matrix_ids()
        id_to_row = artifacts.id_to_row()

        rng = np.random.default_rng(42)
        sample_seeds = rng.choice(tag_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=sample_seeds, tag_weight=0.5)
        q = query_builder.build(req)

        # Full matvec
        dots_full = (tag_m @ q.tag_sparse.T).toarray().ravel()
        valid_full = tag_norms > 0
        full_sims = np.zeros(len(tag_norms), dtype=np.float32)
        full_sims[valid_full] = dots_full[valid_full] / tag_norms[valid_full]
        q.tag_full_sims = full_sims

        for pool_size in [2000, 55000]:
            pool_ids = rng.choice(tag_ids, size=pool_size, replace=False).tolist()

            # Baseline scorer without tag_full_sims
            q_baseline = SimpleNamespace(tag_sparse=q.tag_sparse, plot_centroid=None, tag_full_sims=None)
            s_baseline = scoring.score(pool_ids, q_baseline)

            # Reused matvec scorer
            q_reused = SimpleNamespace(tag_sparse=q.tag_sparse, plot_centroid=None, tag_full_sims=full_sims)
            s_reused = scoring.score(pool_ids, q_reused)

            max_diff = np.max(np.abs(s_baseline.tag - s_reused.tag))
            self.assertLess(max_diff, 1e-6, f"Tag score difference {max_diff} exceeded 1e-6 for {pool_size} IDs")
            np.testing.assert_array_equal(s_baseline.tag_ok, s_reused.tag_ok)

    def test_matvec_flag_pipeline_end_to_end(self):
        """Verify query_builder, candidates, and scoring behave correctly with flag ON."""
        orig_flag = settings.use_matvec_tag_candidates
        orig_ktag = settings.k_tag
        orig_kplot = settings.k_plot
        try:
            settings.use_matvec_tag_candidates = True
            settings.k_tag = 100
            settings.k_plot = 100

            tag_ids = artifacts.matrix_ids()
            rng = np.random.default_rng(123)
            seeds = rng.choice(tag_ids, size=3, replace=False).tolist()
            req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)

            # 1. query_builder must NOT produce tag_ann (SVD skipped)
            q = query_builder.build(req)
            self.assertIsNone(q.tag_ann, "tag_ann must be None when use_matvec_tag_candidates=True")
            self.assertIsNotNone(q.tag_sparse, "tag_sparse must be present")

            # 2. candidates.generate generates pool via matvec + Chroma plot ANN
            pool = candidates.generate(req, q)
            self.assertGreater(len(pool), 0)
            for s in seeds:
                self.assertNotIn(s, pool, "Seed movies must be excluded from candidate pool")

            # 3. q.tag_full_sims was populated during candidate generation
            self.assertIsNotNone(q.tag_full_sims, "tag_full_sims must be populated on q")

            # 4. scoring reuses tag_full_sims and matches CSR slicing within 1e-6
            s_reused = scoring.score(pool, q)

            q_csr = SimpleNamespace(tag_sparse=q.tag_sparse, plot_centroid=q.plot_centroid, tag_full_sims=None)
            s_csr = scoring.score(pool, q_csr)

            max_diff = np.max(np.abs(s_csr.tag - s_reused.tag))
            self.assertLess(max_diff, 1e-6, f"Score difference {max_diff} exceeded 1e-6")
            np.testing.assert_allclose(s_reused.plot, s_csr.plot, atol=1e-6)
        finally:
            settings.use_matvec_tag_candidates = orig_flag
            settings.k_tag = orig_ktag
            settings.k_plot = orig_kplot


if __name__ == "__main__":
    unittest.main()
