import unittest
from unittest.mock import patch
import numpy as np

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import candidates, query_builder, recommender
from app.stores import artifacts, chroma, meta_index


class TestZeroWeightSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        artifacts.load_all()
        meta_index.load_index()
        chroma.get_collection(settings.plots_collection)

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_tag_weight_0_skips_tag_matvec(self):
        """At tag_weight == 0, tag matvec is skipped during candidate generation."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],
            tag_weight=0.0,
        )
        q = query_builder.build(req)

        with patch("app.services.candidates._matvec_tag") as mock_matvec:
            raw_tag, raw_plot = candidates._generate_matvec(
                q, exclude=set(req.movie_ids), tag_weight=0.0
            )
            # Matvec tag must NOT have been called
            mock_matvec.assert_not_called()
            self.assertEqual(len(raw_tag), 0)
            self.assertGreater(len(raw_plot), 0)

    def test_tag_weight_1_skips_plot_ann_but_scores_plot_for_tie_breaking(self):
        """At tag_weight == 1, plot ANN is skipped during candidate generation,
        but plot is still scored on the candidate pool for tie-breaking."""
        req = RecommendRequest(
            movie_ids=[238, 240, 278],
            tag_weight=1.0,
        )
        q = query_builder.build(req)

        with patch("app.stores.chroma.query_ids") as mock_chroma_query:
            raw_tag, raw_plot = candidates._generate_matvec(
                q, exclude=set(req.movie_ids), tag_weight=1.0
            )
            # Chroma query_ids must NOT have been called
            mock_chroma_query.assert_not_called()
            self.assertGreater(len(raw_tag), 0)
            self.assertEqual(len(raw_plot), 0)

        # Full end-to-end retrieve_and_rank test at tag_weight=1.0
        with patch.object(settings, "tie_break_plot", True):
            res = recommender.retrieve_and_rank(req)
            ranked = res["ranked"]
            self.assertGreater(len(ranked), 0)

            # Confirm plot scores are populated on the ranked items
            has_plot_scores = any(r["plot_score"] > 0 for r in ranked)
            self.assertTrue(has_plot_scores, "Plot scores should be scored for tie-breaking")

            # Check tie breaking: if any adjacent items have equal final score (which is tag_score at tw=1),
            # the earlier item must have plot_score >= later item.
            for i in range(len(ranked) - 1):
                if ranked[i]["final_score"] == ranked[i + 1]["final_score"]:
                    self.assertGreaterEqual(
                        ranked[i]["plot_score"],
                        ranked[i + 1]["plot_score"],
                        "Equal final score items must be ordered by descending plot score",
                    )


if __name__ == "__main__":
    unittest.main()
