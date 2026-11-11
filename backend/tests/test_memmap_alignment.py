import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.stores import artifacts, chroma


class TestMemmapAlignment(unittest.TestCase):
    def test_memmap_vectors_equal_chroma_vectors(self):
        """Verify that plot_memmap id/row alignment matches Chroma vectors for 200 random ids."""
        emb, id_to_r = artifacts.plot_memmap()
        self.assertGreater(len(id_to_r), 0)

        rng = np.random.default_rng(12345)
        all_ids = list(id_to_r.keys())
        sample_ids = rng.choice(all_ids, size=min(200, len(all_ids)), replace=False).tolist()

        chroma_embs = chroma.get_plot_embeddings(sample_ids)
        self.assertEqual(len(chroma_embs), len(sample_ids))

        memmap_embs = artifacts.get_plot_embeddings(sample_ids)
        self.assertEqual(len(memmap_embs), len(sample_ids))

        for mid in sample_ids:
            v_chroma = chroma_embs[mid]
            v_memmap = memmap_embs[mid]
            np.testing.assert_allclose(
                v_memmap,
                v_chroma,
                atol=1e-6,
                err_msg=f"Memmap and Chroma vectors mismatch for movie_id {mid}",
            )


if __name__ == "__main__":
    unittest.main()
