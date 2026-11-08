from __future__ import annotations

import logging
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    log.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_plot_text(text: str) -> np.ndarray:
    model = get_sentence_transformer()
    emb = model.encode(text, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)
