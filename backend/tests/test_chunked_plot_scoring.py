import unittest
import numpy as np

from app.stores import artifacts
from app.services import scoring
from types import SimpleNamespace


class TestChunkedPlotScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts.load_all()

    def _old_plot_score(self, pool_ids: list[int], c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        cn = np.linalg.norm(c)
        plot = np.zeros(len(pool_ids))
        if cn > 0:
            c = c / cn
            embs = artifacts.get_plot_embeddings(pool_ids)
            if embs:
                pos = [k for k, i in enumerate(pool_ids) if i in embs]
                mat = np.stack([embs[pool_ids[k]] for k in pos])
                plot[pos] = mat @ c
        return plot

    def test_chunked_vs_old_2000_ids(self):
        """Verify chunked direct scoring matches old path within 1e-6 on 2,000 IDs."""
        emb, id_to_r = artifacts.plot_memmap()
        pool_ids = list(id_to_r.keys())[:2000]

        np.random.seed(42)
        c = np.random.randn(384).astype(np.float32)
        c /= np.linalg.norm(c)

        q = SimpleNamespace(tag_sparse=None, plot_centroid=c)
        scores_new = scoring.score(pool_ids, q)
        plot_old = self._old_plot_score(pool_ids, c)

        np.testing.assert_allclose(scores_new.plot, plot_old, atol=1e-6)

    def test_chunked_vs_old_55000_ids(self):
        """Verify chunked direct scoring matches old path within 1e-6 on 55,000 IDs."""
        emb, id_to_r = artifacts.plot_memmap()
        pool_ids = list(id_to_r.keys())[:55000]

        np.random.seed(42)
        c = np.random.randn(384).astype(np.float32)
        c /= np.linalg.norm(c)

        q = SimpleNamespace(tag_sparse=None, plot_centroid=c)
        scores_new = scoring.score(pool_ids, q)
        plot_old = self._old_plot_score(pool_ids, c)

        np.testing.assert_allclose(scores_new.plot, plot_old, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
