from __future__ import annotations

import itertools
import logging
import time

import numpy as np

from app.config import settings
from app.core.pool import POOL
from app.db import queries
from app.db.queries import Filters
from app.stores import chroma, meta_index, artifacts

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


def _ann(q, where: dict | None = None, timings: dict | None = None, k: int | None = None) -> list[list[int]]:
    """Ranked id lists, one per available signal. Skips a signal that has no query vector.

    Runs tag ANN and plot ANN concurrently on the shared thread pool when both signals
    are present, inline when only one is present. Preserves exact list ordering (tag first, plot second)
    and propagates exceptions from either worker.
    """
    ann_k = k if k is not None else settings.ann_k
    has_tag = q.tag_ann is not None
    has_plot = q.plot_centroid is not None

    if not has_tag and not has_plot:
        return []

    # Single signal: run inline (no pool overhead)
    if has_tag and not has_plot:
        t0 = time.perf_counter()
        tag_res = chroma.query_ids(settings.tags_collection, q.tag_ann, ann_k, where)
        ann_tag_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["ann_tag"] = ann_tag_time
        log.debug("timing ann_tag=%.4fs (inline)", ann_tag_time)
        return [tag_res]

    if has_plot and not has_tag:
        t0 = time.perf_counter()
        plot_res = chroma.query_ids(settings.plots_collection, q.plot_centroid, ann_k, where)
        ann_plot_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["ann_plot"] = ann_plot_time
        log.debug("timing ann_plot=%.4fs (inline)", ann_plot_time)
        return [plot_res]

    # Both signals present: run concurrently on shared POOL
    def _run_tag():
        t0 = time.perf_counter()
        res = chroma.query_ids(settings.tags_collection, q.tag_ann, ann_k, where)
        return res, round(time.perf_counter() - t0, 4)

    def _run_plot():
        t0 = time.perf_counter()
        res = chroma.query_ids(settings.plots_collection, q.plot_centroid, ann_k, where)
        return res, round(time.perf_counter() - t0, 4)

    fut_tag = POOL.submit(_run_tag)
    fut_plot = POOL.submit(_run_plot)

    # Propagate exceptions if either worker fails
    tag_res, ann_tag_time = fut_tag.result()
    plot_res, ann_plot_time = fut_plot.result()

    if timings is not None:
        timings["ann_tag"] = ann_tag_time
        timings["ann_plot"] = ann_plot_time
    log.debug("timing parallel ann_tag=%.4fs ann_plot=%.4fs", ann_tag_time, ann_plot_time)

    # Preserve exact deterministic ordering: tag first, plot second
    return [tag_res, plot_res]


def _matvec_tag(q, k: int, exclude: set[int]) -> tuple[list[int], float]:
    """Exact tag cosine over in-memory CSR tag matrix, saves full sims on q.tag_full_sims."""
    t0 = time.perf_counter()
    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()

    dots = (tag_m @ q.tag_sparse.T).toarray().ravel()
    valid = tag_norms > 0
    sims = np.zeros(len(tag_norms), dtype=np.float32)
    sims[valid] = dots[valid] / tag_norms[valid]

    q.tag_full_sims = sims

    scores_for_pick = sims.copy()
    for s in exclude:
        if s in id_to_row:
            scores_for_pick[id_to_row[s]] = -1.0

    if k < len(scores_for_pick):
        top_idx = np.argpartition(-scores_for_pick, k)[:k]
        top_sorted = top_idx[np.argsort(-scores_for_pick[top_idx])]
    else:
        top_sorted = np.argsort(-scores_for_pick)

    tag_candidates = [int(tag_ids[r]) for r in top_sorted if scores_for_pick[r] > -0.5]
    dt = round(time.perf_counter() - t0, 4)
    return tag_candidates, dt


def _generate_matvec(q, exclude: set[int], timings: dict | None = None) -> tuple[list[int], list[int]]:
    """Retrieves candidates using matvec tag and Chroma plot ANN concurrently on POOL."""
    has_tag = q.tag_sparse is not None
    has_plot = q.plot_centroid is not None

    if not has_tag and not has_plot:
        return [], []

    k_tag = getattr(settings, "k_tag", 2000)
    k_plot = getattr(settings, "k_plot", 2000)

    raw_tag: list[int] = []
    raw_plot: list[int] = []

    def _run_plot():
        t0 = time.perf_counter()
        res = chroma.query_ids(settings.plots_collection, q.plot_centroid, k_plot)
        return res, round(time.perf_counter() - t0, 4)

    if has_tag and has_plot:
        fut_tag = POOL.submit(_matvec_tag, q, k_tag, exclude)
        fut_plot = POOL.submit(_run_plot)
        raw_tag, t_tag = fut_tag.result()
        raw_plot, t_plot = fut_plot.result()
        if timings is not None:
            timings["matvec_tag"] = t_tag
            timings["ann_plot"] = t_plot
        log.debug("timing parallel matvec_tag=%.4fs ann_plot=%.4fs", t_tag, t_plot)
    elif has_tag:
        raw_tag, t_tag = _matvec_tag(q, k_tag, exclude)
        if timings is not None:
            timings["matvec_tag"] = t_tag
        log.debug("timing matvec_tag=%.4fs (inline)", t_tag)
    elif has_plot:
        raw_plot, t_plot = _run_plot()
        if timings is not None:
            timings["ann_plot"] = t_plot
        log.debug("timing ann_plot=%.4fs (inline)", t_plot)

    return raw_tag, raw_plot


def generate(req, q, timings: dict | None = None) -> list[int]:
    """Candidate generation using in-memory metadata mask and parallel ANN or matvec."""
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)

    # No filters: skip gate/mask entirely
    if not f.active:
        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            pool = list(dict.fromkeys(raw_tag + raw_plot))
            pool = [mid for mid in pool if mid not in exclude]
            log.debug("branch=no_filters matvec_tag pool=%d", len(pool))
            return pool
        if settings.use_uncapped_pool:
            raw_lists = _ann(q, None, timings=timings, k=settings.ann_k)
            pool = list(dict.fromkeys(i for lst in raw_lists for i in lst if i not in exclude))
            log.debug("branch=no_filters uncapped_pool=%d", len(pool))
            return pool
        pool = _interleave(_ann(q, None, timings=timings), settings.pool_cap, exclude)
        log.debug("branch=no_filters pool=%d", len(pool))
        return pool

    t0 = time.perf_counter()
    if meta_index.is_loaded():
        # Fast in-memory path
        mask = meta_index.mask_for(f)
        idx = meta_index.get_index()
        n_matches = int(np.count_nonzero(mask))
        gate_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["gate"] = gate_time
        log.debug("timing gate=%.4fs matches=%d (meta_index)", gate_time, n_matches)

        if n_matches == 0:
            if timings is not None:
                timings["ann_tag"] = 0.0
                timings["ann_plot"] = 0.0
                timings["filter_ids"] = 0.0
            return []

        # Small/Exact branch: whole filtered set is the pool, no ANN needed
        if n_matches <= settings.exact_limit:
            matching_ids = idx.ids[mask]
            pool = [int(mid) for mid in matching_ids if mid not in exclude]
            if timings is not None:
                timings["ann_tag"] = 0.0
                timings["ann_plot"] = 0.0
                timings["filter_ids"] = 0.0
            log.debug("branch=small n_matches=%d pool=%d", n_matches, len(pool))
            return pool

        # Large branch: n_matches > exact_limit
        i2r = idx.id_to_row
        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            t0 = time.perf_counter()
            filtered_tag = [mid for mid in raw_tag if mask[i2r[mid]]]
            filtered_plot = [mid for mid in raw_plot if mask[i2r[mid]]]
            filter_time = round(time.perf_counter() - t0, 4)
            if timings is not None:
                timings["filter_ids"] = filter_time
            pool = list(dict.fromkeys(filtered_tag + filtered_plot))
            pool = [mid for mid in pool if mid not in exclude]
            log.debug("branch=large matvec_tag n_matches=%d pool=%d", n_matches, len(pool))
            return pool

        if settings.use_uncapped_pool:
            raw_lists = _ann(q, where=None, timings=timings, k=settings.ann_k)
            t0 = time.perf_counter()
            filtered_lists = []
            for lst in raw_lists:
                filtered_lists.append([mid for mid in lst if mask[i2r[mid]]])
            filter_time = round(time.perf_counter() - t0, 4)
            if timings is not None:
                timings["filter_ids"] = filter_time
            pool = list(dict.fromkeys(mid for lst in filtered_lists for mid in lst if mid not in exclude))
        else:
            # Escalating unfiltered ANN schedule (k=2000 -> 8000 -> 20000) + fast bitmask filtering
            target = settings.pool_cap
            ladder = [settings.ann_k, 8000, 20000]
            filtered_lists = []

            for k_step in ladder:
                raw_lists = _ann(q, where=None, timings=timings, k=k_step)
                t0 = time.perf_counter()
                filtered_lists = []
                for lst in raw_lists:
                    filtered_lists.append([mid for mid in lst if mask[i2r[mid]]])
                filter_time = round(time.perf_counter() - t0, 4)
                if timings is not None:
                    timings["filter_ids"] = filter_time

                tag_count = len(filtered_lists[0]) if len(filtered_lists) > 0 else 0
                plot_count = len(filtered_lists[1]) if len(filtered_lists) > 1 else 0
                if (tag_count >= target // 2 and plot_count >= target // 2) or (tag_count + plot_count >= target):
                    break

            pool = _interleave(filtered_lists, settings.pool_cap, exclude)

        # Starvation check: if either signal has fewer than min_survivors or pool < pool_cap
        min_surv = settings.min_survivors
        tag_count = len(filtered_lists[0]) if len(filtered_lists) > 0 else 0
        plot_count = len(filtered_lists[1]) if len(filtered_lists) > 1 else 0
        has_tag = q.tag_ann is not None
        has_plot = q.plot_centroid is not None

        is_starved = (
            (has_tag and tag_count < min_surv) or
            (has_plot and plot_count < min_surv) or
            (len(pool) < settings.pool_cap)
        )

        if is_starved:
            if n_matches <= settings.starvation_hard_cap:
                log.info(
                    "Starvation fallback triggered (tag=%d, plot=%d, pool=%d); using exact masked set (%d rows)",
                    tag_count, plot_count, len(pool), n_matches,
                )
                matching_ids = idx.ids[mask]
                return [int(mid) for mid in matching_ids if mid not in exclude]
            else:
                log.warning(
                    "Starvation detected (tag=%d, plot=%d, pool=%d) but n_matches (%d) exceeds hard cap (%d)",
                    tag_count, plot_count, len(pool), n_matches, settings.starvation_hard_cap,
                )

        log.debug("branch=large n_matches=%d pool=%d", n_matches, len(pool))
        return pool
    else:
        # Fallback for offline tests where meta_index is not loaded
        gate = queries.gate_ids(f, settings.exact_limit + 1)
        gate_time = round(time.perf_counter() - t0, 4)
        if timings is not None:
            timings["gate"] = gate_time
        if len(gate) <= settings.exact_limit:
            pool = [i for i in gate if i not in exclude]
            return pool
        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            union = list(dict.fromkeys(raw_tag + raw_plot))
            t0 = time.perf_counter()
            keep = queries.filter_ids(union, f)
            if timings is not None:
                timings["filter_ids"] = round(time.perf_counter() - t0, 4)
            pool = [mid for mid in union if mid in keep and mid not in exclude]
            return pool
        where = chroma.build_where(
            list(f.languages),
            f.date_from.year if f.date_from else None,
            f.date_to.year if f.date_to else None,
        )
        lists = _ann(q, where, timings=timings)
        union = list(dict.fromkeys(i for lst in lists for i in lst))
        t0 = time.perf_counter()
        keep = queries.filter_ids(union, f)
        if timings is not None:
            timings["filter_ids"] = round(time.perf_counter() - t0, 4)
        lists = [[i for i in lst if i in keep] for lst in lists]
        return _interleave(lists, settings.pool_cap, exclude)
