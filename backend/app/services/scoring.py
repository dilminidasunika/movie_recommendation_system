from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from app.stores import artifacts, chroma


@dataclass
class Scores:
    ids: list[int]
    tag: np.ndarray
    plot: np.ndarray
    tag_ok: np.ndarray
    plot_ok: np.ndarray


def score(pool_ids: list[int], q) -> Scores:
    n = len(pool_ids)
    tag, plot = np.zeros(n), np.zeros(n)
    tag_ok, plot_ok = np.zeros(n, bool), np.zeros(n, bool)

    # Exact tag cosine on the SPARSE matrix (never the SVD space)
    if q.tag_sparse is not None:
        i2r = artifacts.id_to_row()
        pos = [k for k, i in enumerate(pool_ids) if i in i2r]
        if pos:
            rows = [i2r[pool_ids[k]] for k in pos]
            norms = artifacts.row_norms()[rows]
            dots = (artifacts.tag_matrix()[rows] @ q.tag_sparse.T).toarray().ravel()
            valid = norms > 0
            sims = np.zeros(len(rows))
            sims[valid] = dots[valid] / norms[valid]   # q is already unit norm
            pos = np.asarray(pos)
            tag[pos], tag_ok[pos] = sims, valid

    # Exact plot cosine: embeddings are normalized, so normalize the centroid too
    if q.plot_centroid is not None:
        c = np.asarray(q.plot_centroid, dtype=float)
        cn = np.linalg.norm(c)
        if cn > 0:
            c = c / cn                                  # mean of unit vectors is not unit
            embs = chroma.get_plot_embeddings(pool_ids) # missing ids simply absent
            if embs:
                pos = [k for k, i in enumerate(pool_ids) if i in embs]
                mat = np.stack([embs[pool_ids[k]] for k in pos])
                plot[pos] = mat @ c
                plot_ok[pos] = True

    return Scores(pool_ids, tag, plot, tag_ok, plot_ok)