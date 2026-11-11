import unittest
from unittest import mock
import numpy as np
from pydantic import ValidationError

from app.config import settings
from app.schemas import RecommendRequest, TagIn
from app.services.query_builder import build, EmptyQueryError
from app.services import recommender
from app.stores import chroma


class TestPlotTextSchema(unittest.TestCase):
    """Test schema validation rules for plot_text and tag_weight."""

    def test_length_29_rejected(self):
        text_29 = "a" * 29
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(plot_text=text_29)
        self.assertIn("at least 30 characters", str(ctx.exception))

    def test_length_30_accepted(self):
        text_30 = "a" * 30
        req = RecommendRequest(plot_text=text_30)
        self.assertEqual(req.plot_text, text_30)

    def test_length_1000_accepted(self):
        text_1000 = "a" * 1000
        req = RecommendRequest(plot_text=text_1000)
        self.assertEqual(req.plot_text, text_1000)

    def test_length_1001_rejected(self):
        text_1001 = "a" * 1001
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(plot_text=text_1001)
        self.assertIn("not exceed 1000 characters", str(ctx.exception))

    def test_whitespace_collapsing(self):
        # 30 chars with lots of extra internal/external spaces
        raw = "   hello   " + ("word " * 6) + "   test   "
        req = RecommendRequest(plot_text=raw)
        self.assertNotIn("  ", req.plot_text)
        self.assertEqual(req.plot_text, req.plot_text.strip())

    def test_whitespace_empty_treated_as_none(self):
        # Empty or whitespace-only without other fields raises error
        with self.assertRaises(ValidationError):
            RecommendRequest(plot_text="    ")

        # If tags are provided, whitespace-only plot_text becomes None without error
        req = RecommendRequest(tags=[TagIn(tag_type="genres", name="Action")], plot_text="   \n\t  ")
        self.assertIsNone(req.plot_text)

    def test_empty_request_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest()
        self.assertIn("Provide at least one of movie_ids, tags, or plot_text", str(ctx.exception))

    def test_tag_weight_rule(self):
        # 1. Tags only: tag_weight forced to 1.0
        req_tags = RecommendRequest(tags=[TagIn(tag_type="genres", name="Action")], tag_weight=0.4)
        self.assertEqual(req_tags.tag_weight, 1.0)

        # 2. Text only: tag_weight preserved
        req_text = RecommendRequest(plot_text="A suspenseful mystery thriller about a detective" * 2, tag_weight=0.3)
        self.assertEqual(req_text.tag_weight, 0.3)

        # 3. Movies only: tag_weight preserved
        req_movies = RecommendRequest(movie_ids=[238], tag_weight=0.7)
        self.assertEqual(req_movies.tag_weight, 0.7)

        # 4. Text + Movies: tag_weight preserved
        req_both = RecommendRequest(
            movie_ids=[238],
            plot_text="A suspenseful mystery thriller about a detective" * 2,
            tag_weight=0.45,
        )
        self.assertEqual(req_both.tag_weight, 0.45)


class TestPlotQueryVector(unittest.TestCase):
    """Test plot query vector calculation: text only, movies only, both."""

    def test_text_only_vector(self):
        text = "A retired detective solves an impossible puzzle in a foggy coastal town."
        req = RecommendRequest(plot_text=text)
        q = build(req)

        self.assertIsNotNone(q.plot_centroid)
        self.assertEqual(q.plot_centroid.shape, (384,))
        self.assertEqual(q.plot_centroid.dtype, np.float32)
        norm = np.linalg.norm(q.plot_centroid)
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertIsNone(q.tag_sparse)
        self.assertIsNone(q.tag_ann)

    def test_movies_only_vector(self):
        req = RecommendRequest(movie_ids=[238])
        q = build(req)

        self.assertIsNotNone(q.plot_centroid)
        self.assertEqual(q.plot_centroid.shape, (384,))
        self.assertEqual(q.plot_centroid.dtype, np.float32)
        norm = np.linalg.norm(q.plot_centroid)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_text_and_movies_combined_vector(self):
        text = "A retired detective solves an impossible puzzle in a foggy coastal town."
        req_text = RecommendRequest(plot_text=text)
        q_text = build(req_text)

        req_movie = RecommendRequest(movie_ids=[238])
        q_movie = build(req_movie)

        req_both = RecommendRequest(movie_ids=[238], plot_text=text)
        q_both = build(req_both)

        self.assertIsNotNone(q_both.plot_centroid)
        norm = np.linalg.norm(q_both.plot_centroid)
        self.assertAlmostEqual(norm, 1.0, places=4)

        # Verify combined vector equals normalized(0.5 * text + 0.5 * centroid)
        expected = 0.5 * q_text.plot_centroid + 0.5 * q_movie.plot_centroid
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(q_both.plot_centroid, expected, atol=1e-5)


class TestRetrievalAndRanking(unittest.TestCase):
    """Test retrieval and ranking with text only, text + filters, text + movies."""

    def test_retrieval_text_only(self):
        req = RecommendRequest(
            plot_text="A gripping mystery about a detective hunting a serial killer in a dark rainy city",
            tag_weight=0.0,
            top_n=5,
        )
        out = recommender.retrieve_and_rank(req)
        self.assertIn("ranked", out)
        self.assertGreater(len(out["ranked"]), 0)
        # All ranked items should have plot_score computed
        for item in out["ranked"]:
            self.assertIsNotNone(item["plot_score"])

    @mock.patch("app.services.candidates.queries.gate_ids")
    def test_retrieval_text_and_filters(self, mock_gate):
        sample = chroma.get_collection(settings.plots_collection).get(limit=10)
        sample_ids = [int(i) for i in sample["ids"]]
        mock_gate.return_value = sample_ids

        req = RecommendRequest(
            plot_text="A spaceship crew encounters a mysterious extraterrestrial organism in deep space",
            original_language=["en"],
            genres=["horror"],
            tag_weight=0.0,
            top_n=5,
        )
        out = recommender.retrieve_and_rank(req)
        self.assertIn("ranked", out)
        self.assertGreater(len(out["ranked"]), 0)

    def test_retrieval_text_and_movies(self):
        seed_id = 238
        req = RecommendRequest(
            movie_ids=[seed_id],
            plot_text="An organized crime dynasty transfers control to a reluctant younger heir",
            tag_weight=0.5,
            top_n=5,
        )
        out = recommender.retrieve_and_rank(req)
        self.assertIn("ranked", out)
        self.assertGreater(len(out["ranked"]), 0)
        # Seed movie must never leak into recommendations
        recommended_ids = [r["movie_id"] for r in out["ranked"]]
        self.assertNotIn(seed_id, recommended_ids)


if __name__ == "__main__":
    unittest.main()
