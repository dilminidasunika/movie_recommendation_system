from __future__ import annotations

import logging
import time
from dataclasses import dataclass
import numpy as np

from app.config import settings
from app.stores import artifacts, meta_index

log = logging.getLogger(__name__)


@dataclass
class Scores:
    ids: np.ndarray      # int64 array
    tag: np.ndarray      # float32 array
    plot: np.ndarray     # float32 array
    tag_ok: np.ndarray   # bool array
    plot_ok: np.ndarray  # bool array

    def index(self, movie_id: int) -> int:
        """Helper for test compatibility (returns index of movie_id in ids)."""
        matches = np.flatnonzero(self.ids == movie_id)
        if len(matches) == 0:
            raise ValueError(f"{movie_id} is not in ids")
        return int(matches[0])


def score(pool_ids: list[int] | np.ndarray, q, timings: dict | None = None) -> Scores:
    pool_arr = np.asarray(pool_ids, dtype=np.int64)
    n = len(pool_arr)
    tag = np.zeros(n, dtype=np.float32)
    plot = np.zeros(n, dtype=np.float32)
    tag_ok = np.zeros(n, dtype=bool)
    plot_ok = np.zeros(n, dtype=bool)

    if n == 0:
        return Scores(pool_arr, tag, plot, tag_ok, plot_ok)

    # Determine precomputed tag and plot row alignments
    meta_rows = getattr(q, "_meta_rows", None)
    if meta_rows is not None and len(meta_rows) == n:
        tag_rows = meta_index.meta_to_tag_row()[meta_rows]
        plot_rows = meta_index.meta_to_plot_row()[meta_rows]
    else:
        tag_rows = artifacts.tag_rows_for(pool_arr)
        plot_rows = artifacts.plot_rows_for(pool_arr)

    # Exact tag cosine on the SPARSE matrix (never the SVD space)
    t0 = time.perf_counter()
    if q.tag_sparse is not None:
        valid = tag_rows >= 0
        if np.any(valid):
            if getattr(q, "tag_full_sims", None) is not None:
                tag[valid] = q.tag_full_sims[tag_rows[valid]]
                tag_ok[valid] = artifacts.row_norms()[tag_rows[valid]] > 0
            elif n > 20_000:
                # Large exact pool: compute full dense matvec once and index into it.
                # Bypasses CSR row-slicing and per-candidate sparse matmul.
                tag_m = artifacts.tag_matrix()
                inv = artifacts.inv_norms()
                V = tag_m.shape[1]
                qd = np.zeros(V, dtype=np.float64)
                qd[q.tag_sparse.indices] = q.tag_sparse.data
                full_sims = ((tag_m @ qd) * inv).astype(np.float32)
                tag[valid] = full_sims[tag_rows[valid]]
                tag_ok[valid] = artifacts.row_norms()[tag_rows[valid]] > 0
            else:
                pos = np.flatnonzero(valid)
                rows = tag_rows[valid]
                norms = artifacts.row_norms()[rows]
                dots = (artifacts.tag_matrix()[rows] @ q.tag_sparse.T).toarray().ravel()
                nz = norms > 0
                tag[pos[nz]] = dots[nz] / norms[nz]
                tag_ok[pos[nz]] = True
    score_tag_time = round(time.perf_counter() - t0, 4)
    if timings is not None:
        timings["score_tag"] = score_tag_time
    log.debug("timing score_tag=%.4fs", score_tag_time)

    # Exact plot cosine: embeddings are normalized, so normalize the centroid too
    t0 = time.perf_counter()
    if q.plot_centroid is not None:
        c = np.asarray(q.plot_centroid, dtype=np.float32)
        cn = np.linalg.norm(c)
        if cn > 0:
            c = c / cn
            emb, _ = artifacts.plot_memmap()
            valid_mask = plot_rows >= 0
            if np.any(valid_mask):
                valid_pos = np.flatnonzero(valid_mask)
                valid_rows = plot_rows[valid_mask]

                # Vectorized row sort for sequential memmap disk access
                order = np.argsort(valid_rows)
                sorted_rows = valid_rows[order]
                orig_pos = valid_pos[order]

                chunk_size = 8192
                total_valid = len(sorted_rows)
                for start in range(0, total_valid, chunk_size):
                    end = min(start + chunk_size, total_valid)
                    chunk_vecs = emb[sorted_rows[start:end]]
                    plot[orig_pos[start:end]] = chunk_vecs @ c
                    plot_ok[orig_pos[start:end]] = True
    score_plot_time = round(time.perf_counter() - t0, 4)
    if timings is not None:
        timings["score_plot"] = score_plot_time
    log.debug("timing score_plot=%.4fs", score_plot_time)

    return Scores(pool_arr, tag, plot, tag_ok, plot_ok)
