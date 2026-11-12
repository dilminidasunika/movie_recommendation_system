import unittest
import numpy as np

from app.schemas import RecommendRequest
from app.services import query_builder, scoring, fusion
from app.services.scoring import Scores
from app.stores import artifacts


class TestMissingPlotEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts.load_all()

    def test_missing_plot_embedding_scoring(self):
        """Verify movies missing from the plot memmap have plot_ok=False and no zero-penalty on tag score."""
        emb, id_to_r = artifacts.plot_memmap()
        valid_plot_id = int(id_to_r.sorted_ids[0])
        missing_plot_id = 999_999_999  # Guaranteed absent from plot memmap

        self.assertIn(valid_plot_id, id_to_r)
        self.assertNotIn(missing_plot_id, id_to_r)

        # 1. Test get_plot_embeddings drops missing IDs safely
        embs = artifacts.get_plot_embeddings([valid_plot_id, missing_plot_id])
        self.assertIn(valid_plot_id, embs)
        self.assertNotIn(missing_plot_id, embs)

        # 2. Test scoring behavior
        req = RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.5)
        q = query_builder.build(req)

        scores = scoring.score([valid_plot_id, missing_plot_id], q)
        idx_missing = list(scores.ids).index(missing_plot_id)
        idx_valid = list(scores.ids).index(valid_plot_id)

        self.assertFalse(scores.plot_ok[idx_missing])
        self.assertEqual(scores.plot[idx_missing], 0.0)
        self.assertTrue(scores.plot_ok[idx_valid])

        # 3. Test fusion behavior: missing plot does not corrupt tag score
        ranked = fusion.fuse(scores, tag_weight=0.5, top_n=10)
        ranked_map = {r["movie_id"]: r for r in ranked}

        if missing_plot_id in ranked_map:
            r_missing = ranked_map[missing_plot_id]
            expected_final = 0.5 * scores.tag[idx_missing]
            self.assertAlmostEqual(r_missing["final_score"], expected_final, places=6)
            self.assertEqual(r_missing["plot_score"], 0.0)

    def test_neither_signal_dropped(self):
        """Movies with neither tag nor plot data are assigned -1 and dropped by fusion."""
        ids = np.array([123456789], dtype=np.int64)
        tag = np.array([0.0], dtype=np.float32)
        plot = np.array([0.0], dtype=np.float32)
        tag_ok = np.array([False])
        plot_ok = np.array([False])

        s = Scores(ids, tag, plot, tag_ok, plot_ok)
        ranked = fusion.fuse(s, tag_weight=0.5, top_n=10)
        self.assertEqual(len(ranked), 0)


if __name__ == "__main__":
    unittest.main()
