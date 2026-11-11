import os
import sys
import unittest
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.stores import artifacts
from app.services.query_builder import _movie_tag_part, _chip_part


class TestSvdSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts.load_all()

    def test_svd_transform_sparse_50_cases(self):
        """Compare svd_transform_sparse against svd().transform on 50 varied query vectors."""
        model = artifacts.svd()
        n_features = model.components_.shape[1]
        n_components = model.components_.shape[0]

        vectors = []
        descriptions = []

        # 1. Empty query vectors (5 cases)
        for i in range(5):
            vectors.append(sp.csr_matrix((1, n_features), dtype=np.float64))
            descriptions.append(f"empty_{i}")

        # 2. Single non-zero (1 nnz) vectors (10 cases)
        rng = np.random.default_rng(1001)
        for i in range(10):
            col = int(rng.integers(0, n_features))
            val = float(rng.uniform(0.1, 2.0))
            v = sp.csr_matrix(([val], [col], [0, 1]), shape=(1, n_features), dtype=np.float64)
            vectors.append(v)
            descriptions.append(f"single_nnz_{i}_col_{col}")

        # 3. Many nnz vectors (15 cases, random nnz from 5 to 500)
        for i in range(15):
            k = int(rng.integers(5, 500))
            cols = np.sort(rng.choice(n_features, size=k, replace=False))
            vals = rng.standard_normal(size=k)
            v = sp.csr_matrix((vals, cols, [0, k]), shape=(1, n_features), dtype=np.float64)
            vectors.append(v)
            descriptions.append(f"many_nnz_{i}_k_{k}")

        # 4. Movie-based query vectors (10 cases)
        all_movie_ids = list(artifacts.id_to_row().keys())
        for i in range(10):
            sample_size = int(rng.integers(1, 6))
            mids = rng.choice(all_movie_ids, size=sample_size, replace=False).tolist()
            mpart, _ = _movie_tag_part(mids)
            if mpart is not None:
                vectors.append(mpart)
                descriptions.append(f"movie_based_{i}_size_{sample_size}")
            else:
                vectors.append(sp.csr_matrix((1, n_features), dtype=np.float64))
                descriptions.append(f"movie_based_empty_{i}")

        # 5. Chips-only query vectors (10 cases)
        sample_chips = [
            [{"tag_type": "genres", "name": "Drama"}],
            [{"tag_type": "genres", "name": "Action"}, {"tag_type": "genres", "name": "Comedy"}],
            [{"tag_type": "genres", "name": "Animation"}, {"tag_type": "cast_members", "name": "Tom Hanks"}],
            [{"tag_type": "genres", "name": "Horror"}, {"tag_type": "director", "name": "Steven Spielberg"}],
            [{"tag_type": "keywords", "name": "space"}, {"tag_type": "genres", "name": "Science Fiction"}],
            [{"tag_type": "genres", "name": "Documentary"}],
            [{"tag_type": "genres", "name": "Romance"}, {"tag_type": "keywords", "name": "love"}],
            [{"tag_type": "genres", "name": "Western"}, {"tag_type": "director", "name": "Clint Eastwood"}],
            [{"tag_type": "genres", "name": "Thriller"}, {"tag_type": "keywords", "name": "murder"}],
            [{"tag_type": "genres", "name": "NonExistentGenreXYZ123"}],
        ]
        for i, chips in enumerate(sample_chips):
            cpart = _chip_part(chips)
            if cpart is not None:
                vectors.append(cpart)
                descriptions.append(f"chips_{i}")
            else:
                vectors.append(sp.csr_matrix((1, n_features), dtype=np.float64))
                descriptions.append(f"chips_empty_{i}")

        self.assertEqual(len(vectors), 50)

        # Verify all 50 query vectors against svd().transform
        for idx, (vec, desc) in enumerate(zip(vectors, descriptions)):
            res_fast = artifacts.svd_transform_sparse(vec)
            res_orig = model.transform(vec)

            self.assertEqual(res_fast.shape, (1, n_components), f"Shape mismatch in {desc}")
            self.assertEqual(res_fast.dtype, np.float64, f"Dtype mismatch in {desc}")

            max_err = float(np.max(np.abs(res_fast - res_orig)))
            self.assertLess(
                max_err,
                1e-9,
                f"SVD transform mismatch in {desc} (#{idx}): max abs error = {max_err:.2e} >= 1e-9",
            )


if __name__ == "__main__":
    unittest.main()
