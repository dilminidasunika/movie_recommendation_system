# app/services/query_builder.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from app.schemas import RecommendRequest
from app.stores import artifacts, chroma
from app.services.soup import build_tier_docs

CHIP_WEIGHT = 1.0   # relative weight of the tag-chip part vs the movie part


class EmptyQueryError(ValueError):
    """No usable tag or plot signal could be built. Map to HTTP 422."""


@dataclass
class Query:
    tag_sparse: sp.csr_matrix | None      # (1, V) L2-normalized, exact scoring
    tag_ann: np.ndarray | None            # (n_components,) float32 L2-normalized, ANN
    plot_centroid: np.ndarray | None      # (384,) float32 L2-normalized
    movie_ids: list[int]
    chips: list[dict]
    missing_tag_ids: list[int] = field(default_factory=list)    # selected movies with no tag row
    missing_plot_ids: list[int] = field(default_factory=list)   # selected movies with no plot embedding


def _movie_tag_part(movie_ids: list[int]):
    id_to_row = artifacts.id_to_row()
    present = [i for i in movie_ids if i in id_to_row]
    missing = [i for i in movie_ids if i not in id_to_row]
    if not present:
        return None, missing

    rows = [id_to_row[i] for i in present]
    m = normalize(artifacts.tag_matrix()[rows], norm="l2")        # per-movie unit rows
    mean = sp.csr_matrix(np.full((1, len(rows)), 1.0 / len(rows))) @ m
    mean = normalize(mean, norm="l2")
    return (mean if mean.nnz else None), missing


def _chip_part(chips: list[dict]):
    if not chips:
        return None
    fields: dict[str, list[str]] = defaultdict(list)
    for c in chips:
        fields[c["tag_type"]].append(c["name"])

    docs = build_tier_docs(fields)                                # same code path as offline
    blocks = [v.transform([d]) for v, d in zip(artifacts.vectorizers(), docs)]
    vec = sp.hstack(blocks, format="csr")
    if vec.nnz == 0:                                              # chips not in vocabulary
        return None
    return normalize(vec, norm="l2")


def _plot_centroid(movie_ids: list[int]):
    if not movie_ids:
        return None, []
    embs = chroma.get_plot_embeddings(movie_ids)                  # {id: np.ndarray}
    missing = [i for i in movie_ids if i not in embs]
    if not embs:
        return None, missing
    c = np.mean(np.stack(list(embs.values())), axis=0, keepdims=True)
    c = normalize(c, norm="l2")[0].astype(np.float32)
    return c, missing


def build(req: RecommendRequest) -> Query:
    movie_ids = list(dict.fromkeys(req.movie_ids))                # dedupe, keep order
    chips = [{"tag_type": t.tag_type, "name": t.name} for t in req.tags]

    movie_part, missing_tag = _movie_tag_part(movie_ids)
    chip_part = _chip_part(chips)

    tag_sparse = None
    parts = []
    if movie_part is not None:
        parts.append(movie_part)
    if chip_part is not None:
        parts.append(chip_part * CHIP_WEIGHT)
    if parts:
        tag_sparse = normalize(sum(parts[1:], parts[0]), norm="l2")

    tag_ann = None
    if tag_sparse is not None:
        red = artifacts.svd().transform(tag_sparse)               # (1, n_components)
        tag_ann = normalize(red, norm="l2")[0].astype(np.float32)

    plot_centroid, missing_plot = _plot_centroid(movie_ids)

    if tag_sparse is None and plot_centroid is None:
        raise EmptyQueryError(
            "No usable signal: selected movies have no tags/plot data "
            "and the tag chips matched nothing in the vocabulary."
        )

    return Query(
        tag_sparse=tag_sparse,
        tag_ann=tag_ann,
        plot_centroid=plot_centroid,
        movie_ids=movie_ids,
        chips=chips,
        missing_tag_ids=missing_tag,
        missing_plot_ids=missing_plot,
    )