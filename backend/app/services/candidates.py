from __future__ import annotations

import itertools
import logging
import time

from app.config import settings
from app.db import queries
from app.db.queries import Filters
from app.stores import chroma

log = logging.getLogger(__name__)


def _interleave(lists: list[list[int]], cap: int, exclude: set[int]) -> list[int]:
    """Round-robin over ranked lists, dedupe, skip excluded, stop at cap."""
    seen: set[int] = set()
    out: list[int] = []
    for group in itertools.zip_longest(*lists):
        for i in group:
            if i is None or i in seen or i in exclude:
                continue
            seen.add(i)
            out.append(i)
            if len(out) >= cap:
                return out
    return out


def _ann(q, where: dict | None, timings: dict | None = None) -> list[list[int]]:
    """Ranked id lists, one per available signal. Skips a signal that has no query vector."""
    lists = []
    if q.tag_ann is not None:
        t0 = time.perf_counter()
        tag_res = chroma.query_ids(settings.tags_collection, q.tag_ann,
                                  settings.ann_k, where)
        ann_tag_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["ann_tag"] = ann_tag_time
        log.debug("timing ann_tag=%.4fs", ann_tag_time)
        lists.append(tag_res)
    if q.plot_centroid is not None:                       # step 30
        t0 = time.perf_counter()
        plot_res = chroma.query_ids(settings.plots_collection, q.plot_centroid,
                                   settings.ann_k, where)
        ann_plot_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["ann_plot"] = ann_plot_time
        log.debug("timing ann_plot=%.4fs", ann_plot_time)
        lists.append(plot_res)
    return lists


def generate(req, q, timings: dict | None = None) -> list[int]:
    """Gate, then small branch (exact) or large branch (filtered ANN + post-filter)."""
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    # No filters: skip the gate entirely (step 26)
    if not f.active:
        pool = _interleave(_ann(q, None, timings=timings), settings.pool_cap, exclude)
        log.debug("branch=no_filters pool=%d", len(pool))
        return pool

    # Gate (step 27)
    t0 = time.perf_counter()
    gate = queries.gate_ids(f, settings.exact_limit + 1)
    gate_time = round(time.perf_counter() - t0, 4)
    if timings is not None:
        timings["gate"] = gate_time
    log.debug("timing gate=%.4fs", gate_time)

    # Small branch: whole filtered set is the pool, no ANN (step 28)
    if len(gate) <= settings.exact_limit:
        pool = [i for i in gate if i not in exclude]
        log.debug("branch=small gate=%d pool=%d", len(gate), len(pool))
        return pool

    # Large branch: filtered ANN, post-filter, then cap (step 29)
    where = chroma.build_where(
        list(f.languages),
        f.date_from.year if f.date_from else None,
        f.date_to.year if f.date_to else None,
    )
    lists = _ann(q, where, timings=timings)
    union = list(dict.fromkeys(i for lst in lists for i in lst))
    t0 = time.perf_counter()
    keep = queries.filter_ids(union, f)                   # BEFORE the cap
    filter_time = round(time.perf_counter() - t0, 4)
    if timings is not None:
        timings["filter_ids"] = filter_time
    log.debug("timing filter_ids=%.4fs", filter_time)
    lists = [[i for i in lst if i in keep] for lst in lists]
    pool = _interleave(lists, settings.pool_cap, exclude)
    log.debug("branch=large ann_union=%d kept=%d pool=%d", len(union), len(keep), len(pool))
    return pool
