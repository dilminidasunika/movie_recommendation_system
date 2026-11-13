import unittest
from unittest.mock import patch, MagicMock

from app.stores import meta_index


class TestGenreLimit(unittest.TestCase):
    def test_genre_limit_exceeded(self):
        # Create 33 mock genres
        mock_genres = [f"genre_{i}" for i in range(33)]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [{"original_language": "en"}],
            [{"g": g} for g in mock_genres],
        ]
        # Safeguard against infinite loops if exception is not raised
        mock_cursor.fetchmany.return_value = []

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn

        with patch("app.stores.meta_index.current_pool", return_value=mock_pool):
            with self.assertRaises(RuntimeError) as ctx:
                meta_index.load_index()

            self.assertIn("exceeding the 32-genre limit of uint32 bitmask", str(ctx.exception))
            self.assertIn("33 distinct genres", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
