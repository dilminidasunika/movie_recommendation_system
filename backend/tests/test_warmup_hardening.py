import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from app.config import settings
from app import main


class TestWarmupHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need artifacts.matrix_ids() and artifacts.id_to_row() to work
        from app.stores import artifacts
        artifacts.load_all()

    def test_warmup_prod_failure_raises_runtime_error(self):
        """When in prod and all warmup requests fail, RuntimeError must be raised."""
        with patch.object(settings, "app_env", "prod"):
            with patch("app.services.recommender.retrieve_and_rank", side_effect=Exception("Chroma down")):
                with patch.object(main.log, "exception") as mock_log_exc:
                    with self.assertRaises(RuntimeError) as ctx:
                        main._warmup()
                    self.assertIn("No warmup requests succeeded at boot in production", str(ctx.exception))
                    self.assertEqual(mock_log_exc.call_count, 5)

    def test_warmup_non_prod_failure_logs_but_does_not_raise(self):
        """In non-prod (dev/staging), failure should log exceptions but not raise."""
        with patch.object(settings, "app_env", "development"):
            with patch("app.services.recommender.retrieve_and_rank", side_effect=Exception("Transient error")):
                with patch.object(main.log, "exception") as mock_log_exc:
                    # Should not raise
                    main._warmup()
                    self.assertEqual(mock_log_exc.call_count, 5)

    def test_warmup_seeds_fallback_when_seeds_missing(self):
        """If seeds 238, 240, 278 are missing from id_to_row, use first 3 matrix IDs."""
        empty_id_to_row = {}
        with patch("app.stores.artifacts.id_to_row", return_value=empty_id_to_row):
            with patch.object(settings, "app_env", "development"):
                with patch("app.services.recommender.retrieve_and_rank") as mock_rec:
                    with patch.object(main.log, "warning") as mock_warn:
                        main._warmup()
                        mock_warn.assert_called_once()
                        self.assertIn("Default warmup seeds missing", mock_warn.call_args[0][0])
                        self.assertEqual(mock_rec.call_count, 5)


if __name__ == "__main__":
    unittest.main()
