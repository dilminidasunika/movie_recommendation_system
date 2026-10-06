from __future__ import annotations

from functools import lru_cache

import chromadb
import numpy as np

from app.config import settings

@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(settings.chroma_path))


@lru_cache(maxsize=4)
def get_collection(name: str):
    # get_collection, NOT get_or_create: the runtime must never silently
    # create an empty collection with default HNSW params.
    return _client().get_collection(name)


def build_where(languages: list[str] | None,
                year_from: int | None,
                year_to: int | None) -> dict | None:
    """Chroma rejects an empty where and a single-item $and."""
    clauses = []
    if languages:
        clauses.append({"original_language": {"$in": [l.lower() for l in languages]}})
    if year_from is not None:
        clauses.append({"release_year": {"$gte": int(year_from)}})
    if year_to is not None:
        clauses.append({"release_year": {"$lte": int(year_to)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def query_ids(collection_name: str, vec: np.ndarray, k: int,
              where: dict | None = None) -> list[int]:
    """Ids only, rank order preserved, cast to int at the boundary.
    May return fewer than k with a selective where."""
    col = get_collection(collection_name)
    kwargs = {"query_embeddings": [vec.astype(np.float32).tolist()],
              "n_results": k,
              "include": []}
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    return [int(i) for i in res["ids"][0]]


def get_plot_embeddings(ids: list[int], batch: int = 5000) -> dict[int, np.ndarray]:
    col = get_collection(settings.plots_collection)
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        res = col.get(ids=[str(x) for x in chunk], include=["embeddings"])
        embs = res["embeddings"]
        if embs is None or len(embs) == 0:
            continue
        for rid, emb in zip(res["ids"], embs):
            out[int(rid)] = np.asarray(emb, dtype=np.float32)
    return out