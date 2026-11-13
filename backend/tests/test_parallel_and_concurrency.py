import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import candidates, query_builder, recommender
from app.stores import artifacts, chroma, text_embedder


class TestParallelAndConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        chroma.get_collection("plots_embeddings")
        chroma.get_collection("tags_vectors")
        text_embedder.get_sentence_transformer()

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_ann_ordering_and_singles(self):
        """Verify _ann returns [tag_list, plot_list] in correct order, handles single and empty signals."""
        # Force the SVD / ANN path so tag_ann is populated
        with patch.object(settings, "use_matvec_tag_candidates", False):
            req_both = RecommendRequest(movie_ids=[238, 240, 278])
            q_both = query_builder.build(req_both)

            # Both signals
            lists_both = candidates._ann(q_both, where=None)
            self.assertEqual(len(lists_both), 2)
            # Tag ids list and plot ids list
            self.assertGreater(len(lists_both[0]), 0)
            self.assertGreater(len(lists_both[1]), 0)

            # Only tag signal (simulate plot_centroid is None)
            q_tag_only = query_builder.Query(
                tag_sparse=q_both.tag_sparse,
                tag_ann=q_both.tag_ann,
                plot_centroid=None,
                movie_ids=q_both.movie_ids,
                chips=[],
            )
            lists_tag = candidates._ann(q_tag_only, where=None)
            self.assertEqual(len(lists_tag), 1)
            self.assertEqual(lists_tag[0], lists_both[0])

            # Only plot signal (simulate tag_ann is None)
            q_plot_only = query_builder.Query(
                tag_sparse=None,
                tag_ann=None,
                plot_centroid=q_both.plot_centroid,
                movie_ids=q_both.movie_ids,
                chips=[],
            )
            lists_plot = candidates._ann(q_plot_only, where=None)
            self.assertEqual(len(lists_plot), 1)
            self.assertEqual(lists_plot[0], lists_both[1])

            # Neither signal
            q_neither = query_builder.Query(
                tag_sparse=None,
                tag_ann=None,
                plot_centroid=None,
                movie_ids=q_both.movie_ids,
                chips=[],
            )
            lists_none = candidates._ann(q_neither, where=None)
            self.assertEqual(len(lists_none), 0)

    def test_concurrency_smoke_8_threads(self):
        """8 simultaneous calls to retrieve_and_rank from threads must return identical results with no exceptions."""
        req = RecommendRequest(movie_ids=[238, 240, 278])

        # First run one baseline call to compare against
        base_res = recommender.retrieve_and_rank(req)
        base_ranked = base_res["ranked"]
        self.assertGreater(len(base_ranked), 0)

        results = []
        errors = []

        def worker():
            try:
                res = recommender.retrieve_and_rank(req)
                results.append(res["ranked"])
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker) for _ in range(8)]
            for f in futures:
                f.result()

        self.assertEqual(len(errors), 0, f"Errors in concurrent runs: {errors}")
        self.assertEqual(len(results), 8)

        for idx, r in enumerate(results):
            self.assertEqual(len(r), len(base_ranked), f"Thread {idx} result length mismatch")
            for rank_i, (b_item, cur_item) in enumerate(zip(base_ranked, r)):
                self.assertEqual(
                    cur_item["movie_id"],
                    b_item["movie_id"],
                    f"Thread {idx} rank {rank_i} movie_id mismatch: {cur_item['movie_id']} != {b_item['movie_id']}",
                )
                self.assertAlmostEqual(
                    cur_item["final_score"],
                    b_item["final_score"],
                    places=5,
                    msg=f"Thread {idx} rank {rank_i} final_score mismatch",
                )


if __name__ == "__main__":
    unittest.main()
