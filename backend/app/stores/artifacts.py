from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import scipy.sparse as sp

from app.config import settings

log = logging.getLogger(__name__)

TAG_MATRIX = "tag_matrix.npz"        # CSR, rows aligned with matrix_ids.npy
MATRIX_IDS = "tag_matrix_movie_ids.npy"        # int64 movie ids, one per row
VECTORIZER_HIGH = "tag_vectorizer_high.joblib"  # fitted CountVectorizer for HIGH_FIELDS
VECTORIZER_MID  = "tag_vectorizer_mid.joblib"   # fitted CountVectorizer for MID_FIELDS
VECTORIZER_LOW  = "tag_vectorizer_low.joblib"    # fitted CountVectorizer for LOW_FIELDS
SVD = "tag_svd_model.joblib"                   # fitted TruncatedSVD
PLOT_EMB = "plot_embeddings.npy"     # optional runtime use only (memmap path)
PLOT_IDS = "plot_embeddings_movie_ids.npy"     # int64 movie ids, one per row


def _p(name: str):
    return settings.artifact_path / name


@lru_cache(maxsize=1)
def tag_matrix() -> sp.csr_matrix:
    m = sp.load_npz(_p(TAG_MATRIX)).tocsr()
    m.sort_indices()
    return m


@lru_cache(maxsize=1)
def matrix_ids() -> np.ndarray:
    return np.load(_p(MATRIX_IDS))


@lru_cache(maxsize=1)
def id_to_row() -> dict[int, int]:
    return {int(i): r for r, i in enumerate(matrix_ids())}


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
def plot_memmap():
    emb = np.load(_p(PLOT_EMB), mmap_mode="r")
    ids = np.load(_p(PLOT_IDS))
    return emb, {int(i): r for r, i in enumerate(ids)}


def get_plot_embeddings(movie_ids: list[int]) -> dict[int, np.ndarray]:
    emb, id_to_r = plot_memmap()
    unique_ids = list(dict.fromkeys(movie_ids))
    pairs = [(mid, id_to_r[mid]) for mid in unique_ids if mid in id_to_r]
    if not pairs:
        return {}
    pairs.sort(key=lambda x: x[1])
    sorted_rows = [p[1] for p in pairs]
    sorted_mids = [p[0] for p in pairs]
    vecs = np.asarray(emb[sorted_rows], dtype=np.float32)
    return {mid: vecs[i] for i, mid in enumerate(sorted_mids)}


@lru_cache(maxsize=1)
def row_norms() -> np.ndarray:
    m = tag_matrix()
    return np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())


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
    svd()
    id_to_row()
    emb, _ = plot_memmap()
    _ = np.sum(emb[:, 0])  # pre-fault plot memmap into OS page cache
    row_norms()             # pre-compute and cache row norms
    log.info("artifacts loaded: %s rows, %s cols, svd=%s",
             m.shape[0], m.shape[1], svd().components_.shape)
