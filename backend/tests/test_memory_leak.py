"""
Item 0: Memory leak tests.
(a) 250 distinct requests — working-set growth < 60 MB once cache is full.
(b) No cached Query attribute is a numpy array > 100 KB.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings


def _get_rss_mb():
    """Return current process RSS in MB (Windows & Linux)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p

    class PMCEX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMCEX), wintypes.DWORD]
    p = PMCEX()
    p.cb = ctypes.sizeof(PMCEX)
    res = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    if res:
        # On Windows under high memory load, the OS aggressively trims WorkingSetSize.
        # PrivateUsage measures committed process memory, deterministically detecting memory leaks.
        return p.PrivateUsage / (1024 * 1024)
    return 0.0


class TestMemoryLeak(unittest.TestCase):
    """Verify that cached Query objects do not retain large numpy arrays."""

    @classmethod
    def setUpClass(cls):
        from app.db.pool import open_pool
        from app.stores import artifacts, meta_index, chroma, text_embedder

        open_pool()
        artifacts.load_all()
        meta_index.load_index()
        chroma.get_collection(settings.plots_collection)
        text_embedder.get_sentence_transformer()

        emb, _ = artifacts.plot_memmap()
        _ = np.sum(emb[::2])  # Pre-fault memmap pages so OS cold-page faults do not distort growth

        tag_ids_set = set(artifacts.matrix_ids())
        _, id_to_r = artifacts.plot_memmap()
        common = tag_ids_set & set(id_to_r.keys())
        cls.common_ids = np.array(sorted(common))

    def test_no_large_numpy_on_cached_query(self):
        """(b) After retrieve_and_rank, no Query attr is a numpy array > 100 KB."""
        from app.schemas import RecommendRequest
        from app.services import recommender

        rng = np.random.default_rng(99)
        for _ in range(5):
            seeds = rng.choice(self.common_ids, size=3, replace=False).tolist()
            req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
            out = recommender.retrieve_and_rank(req)
            q = out["query"]
            for attr_name in vars(q):
                val = getattr(q, attr_name)
                if isinstance(val, np.ndarray) and val.nbytes > 100 * 1024:
                    self.fail(
                        f"Query.{attr_name} is a numpy array of {val.nbytes / 1024:.0f} KB "
                        f"(limit 100 KB); it should be released after scoring."
                    )

    def test_working_set_growth_bounded(self):
        """(a) 250 distinct requests — working-set growth < 60 MB once cache is full."""
        import gc
        from app.schemas import RecommendRequest
        from app.services import recommender, paging

        # Use patch to run test with k=2000 so that 450 requests finish in ~2 minutes
        # while still exercising full 1.06M matvec, full tag_full_sims allocation and release.
        with patch.object(settings, "k_tag", 2000), patch.object(settings, "k_plot", 2000):
            rng = np.random.default_rng(42)

            # Fill cache to maxsize (200 entries) with distinct requests
            for i in range(200):
                seeds = rng.choice(self.common_ids, size=3, replace=False).tolist()
                req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
                out = recommender.retrieve_and_rank(req)
                rid = paging.request_id(req)
                paging.put(rid, {
                    "ranked": out["ranked"], "query": out["query"],
                    "warnings": out["warnings"], "pool_size": out["pool_size"],
                    "timings": out["timings"], "explanations": {},
                })

            gc.collect()
            rss_before = _get_rss_mb()

            # Now do 250 MORE distinct requests (continuous cache churn/eviction)
            for i in range(250):
                seeds = rng.choice(self.common_ids, size=3, replace=False).tolist()
                req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
                out = recommender.retrieve_and_rank(req)
                rid = paging.request_id(req)
                paging.put(rid, {
                    "ranked": out["ranked"], "query": out["query"],
                    "warnings": out["warnings"], "pool_size": out["pool_size"],
                    "timings": out["timings"], "explanations": {},
                })

            gc.collect()
            rss_after = _get_rss_mb()
            growth = rss_after - rss_before
            print(f"\n  [Memory Leak Test] RSS before={rss_before:.1f} MB, after={rss_after:.1f} MB, growth={growth:.1f} MB")
            self.assertLess(growth, 60.0,
                            f"Working set grew {growth:.1f} MB after 250 cache-churn requests (limit 60 MB).")


if __name__ == "__main__":
    unittest.main()
