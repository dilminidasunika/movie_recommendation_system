import unittest
from types import SimpleNamespace
import numpy as np

from app.services import fusion


class TestFusionFloat64(unittest.TestCase):
    def test_float64_precision_and_stability(self):
        # Create synthetic scores
        np.random.seed(42)
        n = 1000
        tag_scores = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)
        plot_scores = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)
        tag_ok = np.random.choice([True, False], size=n, p=[0.9, 0.1])
        plot_ok = np.random.choice([True, False], size=n, p=[0.9, 0.1])
        ids = np.arange(1000, 1000 + n, dtype=np.int64)

        s = SimpleNamespace(
            ids=ids,
            tag=tag_scores,
            tag_ok=tag_ok,
            plot=plot_scores,
            plot_ok=plot_ok,
        )

        final, tn, pn = fusion.compute_final_scores(s, tag_weight=0.5)

        # 1. final, tn, pn must be float64
        self.assertEqual(final.dtype, np.float64)
        self.assertEqual(tn.dtype, np.float64)
        self.assertEqual(pn.dtype, np.float64)

        # 2. Scores must differ by <= 1e-6 from pure float32 computation
        final_f32 = (0.5 * fusion._scale(tag_scores, tag_ok) + 0.5 * fusion._scale(plot_scores, plot_ok)).astype(np.float32)
        valid = tag_ok | plot_ok
        diff = np.abs(final[valid] - final_f32[valid])
        self.assertLessEqual(np.max(diff), 1e-6)

        # 3. Two runs produce identical results
        res1 = fusion.fuse(s, tag_weight=0.5, top_n=50)
        res2 = fusion.fuse(s, tag_weight=0.5, top_n=50)
        self.assertEqual(res1, res2)

        # 4. Scores in the API response are rounded to at most 6 decimal places
        for item in res1:
            final_val = item["final_score"]
            tag_val = item["tag_score"]
            plot_val = item["plot_score"]
            self.assertEqual(final_val, round(final_val, 6))
            self.assertEqual(tag_val, round(tag_val, 6))
            self.assertEqual(plot_val, round(plot_val, 6))


if __name__ == "__main__":
    unittest.main()
