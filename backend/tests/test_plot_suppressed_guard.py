import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from app.config import settings
from app.stores import artifacts

orig_p = artifacts._p


class TestPlotSuppressedGuard(unittest.TestCase):
    def setUp(self):
        artifacts.plot_suppressed.cache_clear()
        artifacts.plot_ids.cache_clear()

    def tearDown(self):
        artifacts.plot_suppressed.cache_clear()
        artifacts.plot_ids.cache_clear()

    def test_missing_artifact_raises_runtime_error(self):
        """When suppress_boilerplate_plots=True and plot_suppressed.npy is missing, raise RuntimeError."""
        def mock_p(name):
            if name == "plot_suppressed.npy":
                return Path("non_existent_plot_suppressed.npy")
            return orig_p(name)

        with patch.object(settings, "suppress_boilerplate_plots", True):
            with patch("app.stores.artifacts._p", side_effect=mock_p):
                with self.assertRaises(RuntimeError) as ctx:
                    artifacts.plot_suppressed()
                self.assertIn("Boilerplate plot suppression is enabled", str(ctx.exception))
                self.assertIn("does not exist", str(ctx.exception))

    def test_mismatched_length_raises_runtime_error(self):
        """When suppress_boilerplate_plots=True and row count != plot row count, raise RuntimeError."""
        def mock_p(name):
            if name == "plot_suppressed.npy":
                fake_path = MagicMock(spec=Path)
                fake_path.exists.return_value = True
                fake_path.__str__.return_value = "fake_suppressed.npy"
                return fake_path
            return orig_p(name)

        wrong_size_array = np.zeros(10, dtype=bool)
        mock_plot_ids = np.arange(100, dtype=np.int64)  # length 100 != 10

        with patch.object(settings, "suppress_boilerplate_plots", True):
            with patch("app.stores.artifacts._p", side_effect=mock_p):
                with patch("app.stores.artifacts.plot_ids", return_value=mock_plot_ids):
                    with patch("numpy.load", return_value=wrong_size_array):
                        with self.assertRaises(RuntimeError) as ctx:
                            artifacts.plot_suppressed()
                        self.assertIn("Boilerplate plot suppression artifact mismatch", str(ctx.exception))
                        self.assertIn("expected 100 rows", str(ctx.exception))

    def test_valid_artifact_returns_correct_mask(self):
        """When valid, returns the real mask without error."""
        with patch.object(settings, "suppress_boilerplate_plots", True):
            mask = artifacts.plot_suppressed()
            self.assertEqual(len(mask), len(artifacts.plot_ids()))
            self.assertEqual(mask.dtype, bool)


if __name__ == "__main__":
    unittest.main()
