from __future__ import annotations

import logging
import time
from dataclasses import dataclass
import numpy as np

from app.config import settings
from app.stores import artifacts

log = logging.getLogger(__name__)


@dataclass
class Scores:
    ids: list[int]
    tag: np.ndarray
    plot: np.ndarray
    tag_ok: np.ndarray
    plot_ok: np.ndarray


def score(pool_ids: list[int], q, timings: dict | None = None) -> Scores:
    n = len(pool_ids)
    tag, plot = np.zeros(n), np.zeros(n)
    tag_ok, plot_ok = np.zeros(n, bool), np.zeros(n, bool)

    # Exact tag cosine on the SPARSE matrix (never the SVD space)
    t0 = time.perf_counter()
    if q.tag_sparse is not None:
        i2r = artifacts.id_to_row()
        pos = [k for k, i in enumerate(pool_ids) if i in i2r]
        if pos:
            rows = [i2r[pool_ids[k]] for k in pos]
            if getattr(q, "tag_full_sims", None) is not None:
                sims = q.tag_full_sims[rows]
                valid = artifacts.row_norms()[rows] > 0
            elif len(pos) > 20_000:
                # Large exact pool: compute full dense matvec once and index into it.
                # This avoids CSR row-slicing + sparse matmul for each pool member.
                tag_m = artifacts.tag_matrix()
                inv = artifacts.inv_norms()
                V = tag_m.shape[1]
                qd = np.zeros(V, dtype=np.float64)
                qd[q.tag_sparse.indices] = q.tag_sparse.data
                full_sims = ((tag_m @ qd) * inv).astype(np.float32)
                sims = full_sims[rows]
                valid = artifacts.row_norms()[rows] > 0
                # Do not store on q: it will be released after scoring anyway
            else:
                norms = artifacts.row_norms()[rows]
                dots = (artifacts.tag_matrix()[rows] @ q.tag_sparse.T).toarray().ravel()
                valid = norms > 0
                sims = np.zeros(len(rows))
                sims[valid] = dots[valid] / norms[valid]   # q is already unit norm
            pos = np.asarray(pos)
            tag[pos], tag_ok[pos] = sims, valid
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
            c = c / cn                                  # mean of unit vectors is not unit
            emb, id_to_r = artifacts.plot_memmap()
            valid_items = [(id_to_r[mid], k) for k, mid in enumerate(pool_ids) if mid in id_to_r]
            if valid_items:
                valid_items.sort(key=lambda x: x[0])
                rows = np.array([x[0] for x in valid_items], dtype=np.int64)
                orig_pos = np.array([x[1] for x in valid_items], dtype=np.int64)
                chunk_size = 8192
                total_valid = len(rows)
                for start in range(0, total_valid, chunk_size):
                    end = min(start + chunk_size, total_valid)
                    chunk_vecs = emb[rows[start:end]]
                    plot[orig_pos[start:end]] = chunk_vecs @ c
                    plot_ok[orig_pos[start:end]] = True
    score_plot_time = round(time.perf_counter() - t0, 4)
    if timings is not None:
        timings["score_plot"] = score_plot_time
    log.debug("timing score_plot=%.4fs", score_plot_time)

    return Scores(pool_ids, tag, plot, tag_ok, plot_ok)
