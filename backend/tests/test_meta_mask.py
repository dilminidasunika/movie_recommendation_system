from __future__ import annotations

import os
import sys
import unittest
from datetime import date
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.pool import open_pool, close_pool, current_pool
from app.db.queries import Filters, _where
from app.stores import meta_index


class TestMetaMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        open_pool()
        meta_index.load_index()

    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_mask_matches_postgres_50_combinations(self):
        """Compare in-memory mask_for against real Postgres queries for 50+ filter combinations."""
        idx = meta_index.get_index()
        genres_sample = ["Drama", "Action", "Comedy", "Animation", "Horror", "Documentary", "Romance", "Western"]
        langs_sample = ["en", "fr", "es", "ja", "de", "zh", "ru", "ko", "it"]

        rng = random.Random(42)

        test_cases = [
            # 1. Empty filter
            Filters(),
            # 2. Non-existent genre
            Filters(genres=("NonExistentGenreXYZ",)),
            # 3. Non-existent language
            Filters(languages=("non_existent_lang",)),
            # 4. Old date range
            Filters(date_from=date(1890, 1, 1), date_to=date(1910, 12, 31)),
            # 5. Future date range
            Filters(date_from=date(2035, 1, 1), date_to=date(2040, 12, 31)),
            # 6. Single genre Drama
            Filters(genres=("Drama",)),
            # 7. Single language Japanese
            Filters(languages=("ja",)),
            # 8. Date from only
            Filters(date_from=date(2022, 1, 1)),
            # 9. Date to only
            Filters(date_to=date(1950, 12, 31)),
            # 10. Specific month/day bounds
            Filters(date_from=date(1972, 3, 24), date_to=date(1972, 3, 25)),
        ]

        # Generate 45 additional randomized filter combinations
        for i in range(45):
            use_lang = rng.random() < 0.6
            use_genre = rng.random() < 0.6
            use_date = rng.random() < 0.6

            langs = tuple(rng.sample(langs_sample, k=rng.randint(1, 3))) if use_lang else ()
            genres = tuple(rng.sample(genres_sample, k=rng.randint(1, 2))) if use_genre else ()

            d_from = None
            d_to = None
            if use_date:
                y1 = rng.randint(1950, 2020)
                y2 = rng.randint(y1, min(y1 + 25, 2024))
                m1, d1 = rng.randint(1, 12), rng.randint(1, 28)
                m2, d2 = rng.randint(1, 12), rng.randint(1, 28)
                d_from = date(y1, m1, d1)
                d_to = date(y2, m2, d2)
                if d_from > d_to:
                    d_from, d_to = d_to, d_from

            test_cases.append(Filters(languages=langs, genres=genres, date_from=d_from, date_to=d_to))

        self.assertGreaterEqual(len(test_cases), 50)

        with current_pool().connection() as conn:
            for c_idx, f in enumerate(test_cases):
                # 1. In-memory mask
                m = meta_index.mask_for(f)
                mask_ids = set(idx.ids[m].tolist())

                # 2. Postgres query
                clauses, params = _where(f)
                sql = "SELECT id FROM movies"
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                rows = conn.execute(sql, params).fetchall()
                pg_ids = set(int(r["id"]) for r in rows)

                diff = mask_ids ^ pg_ids
                self.assertEqual(
                    len(diff),
                    0,
                    f"Test case #{c_idx} mismatch for filters {f}! Mask={len(mask_ids)}, Postgres={len(pg_ids)}. Mismatched IDs: {list(diff)[:10]}",
                )

        print(f"\n[OK] All {len(test_cases)} randomized filter test cases matched Postgres 100.000%!")


if __name__ == "__main__":
    unittest.main()
