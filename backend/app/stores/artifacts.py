from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import scipy.sparse as sp

from app.config import settings

log = logging.getLogger(__name__)

TAG_MATRIX = "tag_matrix.npz"        # CSR, rows aligned with matrix_ids.npy
MATRIX_IDS = "matrix_ids.npy"        # int64 movie ids, one per row
VECTORIZERS = "vectorizers.joblib"   # list of 3 fitted CountVectorizers, same order as column blocks
SVD = "svd.joblib"                   # fitted TruncatedSVD
PLOT_EMB = "plot_embeddings.npy"     # optional runtime use only (memmap path)
PLOT_IDS = "plot_ids.npy"            # must exist if you use the memmap path


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
    return joblib.load(_p(VECTORIZERS))


@lru_cache(maxsize=1)
def svd():
    return joblib.load(_p(SVD))


@lru_cache(maxsize=1)
def plot_memmap():
    """Only if Chroma batch fetch is too slow for the small branch (step 33)."""
    emb = np.load(_p(PLOT_EMB), mmap_mode="r")
    ids = np.load(_p(PLOT_IDS))
    return emb, {int(i): r for r, i in enumerate(ids)}

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
    log.info("artifacts loaded: %s rows, %s cols, svd=%s",
             m.shape[0], m.shape[1], svd().components_.shape)