import concurrent.futures
import unittest

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import recommender
from app.stores import artifacts, meta_index, chroma


class TestConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()
        chroma.get_collection(settings.plots_collection)
        chroma.get_collection(settings.tags_collection)

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_8_thread_concurrent_retrieve_and_rank(self):
        """Run 8 concurrent retrieve_and_rank calls. Verify 0 exceptions and identical outputs."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],
            tag_weight=0.5,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(recommender.retrieve_and_rank, req) for _ in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 8)
        first_ranked = results[0]["ranked"]
        self.assertGreater(len(first_ranked), 0)

        for i in range(1, 8):
            cur_ranked = results[i]["ranked"]
            self.assertEqual(len(cur_ranked), len(first_ranked))
            for item_cur, item_first in zip(cur_ranked, first_ranked):
                self.assertEqual(item_cur["movie_id"], item_first["movie_id"])
                self.assertAlmostEqual(item_cur["final_score"], item_first["final_score"], places=6)
                self.assertAlmostEqual(item_cur["tag_score"], item_first["tag_score"], places=6)
                self.assertAlmostEqual(item_cur["plot_score"], item_first["plot_score"], places=6)


if __name__ == "__main__":
    unittest.main()
