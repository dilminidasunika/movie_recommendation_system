from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import scipy.sparse as sp

from app.config import settings

log = logging.getLogger(__name__)

TAG_MATRIX = "tag_matrix.npz"                  # CSR, rows aligned with matrix_ids.npy
MATRIX_IDS = "tag_matrix_movie_ids.npy"        # int64 movie ids, one per row
VECTORIZER_HIGH = "tag_vectorizer_high.joblib"  # fitted CountVectorizer for HIGH_FIELDS
VECTORIZER_MID  = "tag_vectorizer_mid.joblib"   # fitted CountVectorizer for MID_FIELDS
VECTORIZER_LOW  = "tag_vectorizer_low.joblib"   # fitted CountVectorizer for LOW_FIELDS
SVD = "tag_svd_model.joblib"                   # fitted TruncatedSVD
PLOT_EMB = "plot_embeddings.npy"               # optional runtime use only (memmap path)
PLOT_IDS = "plot_embeddings_movie_ids.npy"     # int64 movie ids, one per row


def _p(name: str):
    return settings.artifact_path / name


class SortedIdIndex:
    """Vectorized ID-to-row index using sorted IDs and np.searchsorted.

    Eliminates Python dict overhead (~80-100 MB per 1M items) while providing
    O(log N) vectorized lookups and drop-in dict compatibility (__contains__,
    __getitem__, get, keys, len).
    """
    __slots__ = ("sorted_ids", "order")

    def __init__(self, ids: np.ndarray):
        ids = np.asarray(ids, dtype=np.int64)
        self.order = np.argsort(ids, kind="stable").astype(np.int32)
        self.sorted_ids = ids[self.order]

    def rows_for(self, query_ids) -> np.ndarray:
        """Vectorized lookup: returns int32 row indices, with -1 for absent IDs."""
        q = np.asarray(query_ids, dtype=np.int64)
        if q.size == 0:
            return np.empty(q.shape, dtype=np.int32)
        idx = np.searchsorted(self.sorted_ids, q)
        in_bounds = idx < len(self.sorted_ids)
        valid = in_bounds & (self.sorted_ids[np.where(in_bounds, idx, 0)] == q)
        res = np.full(q.shape, -1, dtype=np.int32)
        res[valid] = self.order[idx[valid]]
        return res

    def __contains__(self, item) -> bool:
        try:
            mid = int(item)
        except (ValueError, TypeError):
            return False
        idx = np.searchsorted(self.sorted_ids, mid)
        return idx < len(self.sorted_ids) and self.sorted_ids[idx] == mid

    def __getitem__(self, item) -> int:
        mid = int(item)
        idx = np.searchsorted(self.sorted_ids, mid)
        if idx < len(self.sorted_ids) and self.sorted_ids[idx] == mid:
            return int(self.order[idx])
        raise KeyError(item)

    def get(self, item, default=None):
        mid = int(item)
        idx = np.searchsorted(self.sorted_ids, mid)
        if idx < len(self.sorted_ids) and self.sorted_ids[idx] == mid:
            return int(self.order[idx])
        return default

    def __len__(self) -> int:
        return len(self.sorted_ids)

    def keys(self):
        return self.sorted_ids


@lru_cache(maxsize=1)
def tag_matrix() -> sp.csr_matrix:
    m = sp.load_npz(_p(TAG_MATRIX)).tocsr()
    m.sort_indices()
    return m


@lru_cache(maxsize=1)
def matrix_ids() -> np.ndarray:
    return np.load(_p(MATRIX_IDS))


@lru_cache(maxsize=1)
def tag_index() -> SortedIdIndex:
    return SortedIdIndex(matrix_ids())


def id_to_row() -> SortedIdIndex:
    """Returns the vectorized ID-to-row index for the tag matrix."""
    return tag_index()


def tag_rows_for(ids) -> np.ndarray:
    """Vectorized lookup for tag matrix rows (-1 if absent)."""
    return tag_index().rows_for(ids)


@lru_cache(maxsize=1)
def vectorizers() -> list:
    return [
        joblib.load(_p(VECTORIZER_HIGH)),
        joblib.load(_p(VECTORIZER_MID)),
        joblib.load(_p(VECTORIZER_LOW)),
    ]


@lru_cache(maxsize=1)
def svd():
    return joblib.load(_p(SVD))


def svd_transform_sparse(x_csr: sp.csr_matrix | None) -> np.ndarray:
    """Fast TruncatedSVD transform for a 1xV sparse CSR vector by taking the dot product
    only over active sub-columns of components_, avoiding streaming the full dense matrix.
    Returns (1, n_components) float64 array.
    """
    model = svd()
    n_components = model.components_.shape[0]
    if x_csr is None or x_csr.nnz == 0:
        return np.zeros((1, n_components), dtype=np.float64)

    if not sp.isspmatrix_csr(x_csr):
        x_csr = x_csr.tocsr()

    cols = x_csr.indices
    data = x_csr.data
    red = model.components_[:, cols] @ data
    return red.reshape(1, n_components).astype(np.float64)


@lru_cache(maxsize=1)
def plot_ids() -> np.ndarray:
    return np.load(_p(PLOT_IDS))


@lru_cache(maxsize=1)
def plot_index() -> SortedIdIndex:
    return SortedIdIndex(plot_ids())


def plot_rows_for(ids) -> np.ndarray:
    """Vectorized lookup for plot memmap rows (-1 if absent)."""
    return plot_index().rows_for(ids)


@lru_cache(maxsize=1)
def plot_suppressed() -> np.ndarray:
    """Bool mask aligned with plot memmap rows: True = boilerplate/suppressed."""
    p = _p("plot_suppressed.npy")
    expected_rows = plot_ids().shape[0]

    if getattr(settings, "suppress_boilerplate_plots", True):
        if not p.exists():
            raise RuntimeError(
                f"Boilerplate plot suppression is enabled (suppress_boilerplate_plots=True) "
                f"but artifact '{p}' does not exist. Run backend/scripts/build_plot_suppressed.py to generate it."
            )
        arr = np.load(p)
        if len(arr) != expected_rows:
            raise RuntimeError(
                f"Boilerplate plot suppression artifact mismatch: '{p}' has {len(arr)} rows, "
                f"expected {expected_rows} rows aligned with plot embeddings catalog."
            )
        return arr

    if p.exists():
        arr = np.load(p)
        if len(arr) == expected_rows:
            return arr
    return np.zeros(expected_rows, dtype=bool)


@lru_cache(maxsize=1)
def plot_memmap():
    emb = np.load(_p(PLOT_EMB), mmap_mode="r")
    return emb, plot_index()


def get_plot_embeddings(movie_ids: list[int] | np.ndarray) -> dict[int, np.ndarray]:
    emb, idx = plot_memmap()
    mids = np.asarray(movie_ids, dtype=np.int64)
    rows = idx.rows_for(mids)
    valid = rows >= 0
    if not np.any(valid):
        return {}
    valid_mids = mids[valid]
    valid_rows = rows[valid]
    order = np.argsort(valid_rows)
    sorted_rows = valid_rows[order]
    sorted_mids = valid_mids[order]
    vecs = np.asarray(emb[sorted_rows], dtype=np.float32)
    return {int(mid): vecs[i] for i, mid in enumerate(sorted_mids)}


@lru_cache(maxsize=1)
def row_norms() -> np.ndarray:
    m = tag_matrix()
    return np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())


@lru_cache(maxsize=1)
def inv_norms() -> np.ndarray:
    """Precomputed reciprocal norms: 0 where norm is 0 (avoids division in matvec)."""
    norms = row_norms()
    inv = np.zeros_like(norms)
    valid = norms > 0
    inv[valid] = 1.0 / norms[valid]
    return inv


def load_all() -> None:
    """Call once from the FastAPI lifespan so the first request isn't slow
    and a missing or corrupt file fails at boot, not mid-request."""
    m, ids = tag_matrix(), matrix_ids()
    if m.shape[0] != len(ids):
        raise RuntimeError(f"tag matrix rows {m.shape[0]} != matrix_ids {len(ids)}")
    vs = vectorizers()
    if len(vs) != 3:
        raise RuntimeError(f"expected 3 tier vectorizers, got {len(vs)}")
    total_vocab = sum(len(v.vocabulary_) for v in vs)
    if total_vocab != m.shape[1]:
        raise RuntimeError(f"vocab {total_vocab} != matrix cols {m.shape[1]}")
    if not settings.use_matvec_tag_candidates:
        svd()
    tag_index()
    plot_index()
    emb, _ = plot_memmap()
    _ = np.sum(emb[:, 0])  # pre-fault plot memmap into OS page cache
    row_norms()             # pre-compute and cache row norms
    inv_norms()             # pre-compute and cache inverse norms
    ps = plot_suppressed()  # load boilerplate-plot suppression mask
    log.info("plot_suppressed: %d / %d rows marked", int(ps.sum()), len(ps))
    svd_shape = svd().components_.shape if not settings.use_matvec_tag_candidates else "skipped"
    log.info("artifacts loaded: %s rows, %s cols, svd=%s",
             m.shape[0], m.shape[1], svd_shape)
