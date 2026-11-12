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


def _interleave(lists: list[list[int]], cap: int, exclude: set[int]) -> np.ndarray:
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
                return np.array(out, dtype=np.int64)
    return np.array(out, dtype=np.int64)


def _unique_ordered(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Concatenate two int64 arrays, deduplicating while preserving first-seen order."""
    if len(arr1) == 0:
        return arr2
    if len(arr2) == 0:
        return arr1
    combined = np.concatenate([arr1, arr2])
    _, idx = np.unique(combined, return_index=True)
    return combined[np.sort(idx)]


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


def _matvec_tag(q, k: int, exclude: set[int]) -> tuple[np.ndarray, float]:
    """Exact tag cosine over in-memory CSR tag matrix, saves full sims on q.tag_full_sims.

    Optimizations & tie-handling:
    - Dense float64 query vector: CSR @ dense releases GIL for BLAS kernels.
    - Precomputed inv_norms: elementwise multiply instead of boolean-mask division.
    - Zero-copy seed exclusion: temporarily marks excluded seeds as -1.0, restores after selection.
    - Tie-aware boundary coverage: includes ALL candidates whose score >= kth score,
      with a 30,000 safety cap keeping lowest movie IDs if exceeded.
    - Returns int64 numpy array directly without Python int conversion loops.
    """
    t0 = time.perf_counter()
    tag_m = artifacts.tag_matrix()
    tag_ids = artifacts.matrix_ids()
    id_to_row = artifacts.id_to_row()
    inv = artifacts.inv_norms()

    # Build dense float64 query from the sparse representation
    V = tag_m.shape[1]
    qd = np.zeros(V, dtype=np.float64)
    qd[q.tag_sparse.indices] = q.tag_sparse.data

    # CSR @ dense (releases GIL via BLAS); elementwise multiply by precomputed 1/norm
    dots = tag_m @ qd                       # float64, length N
    sims = (dots * inv).astype(np.float32)  # float32, length N

    q.tag_full_sims = sims

    # Exclude seeds in-place without copying the 4.3 MB sims array
    orig_vals: dict[int, float] = {}
    for s in exclude:
        if s in id_to_row:
            r = id_to_row[s]
            orig_vals[r] = float(sims[r])
            sims[r] = -1.0

    if k < len(sims):
        kth = float(np.partition(sims, -k)[-k])
        # Include all candidates whose score >= kth (excluding seeds marked -1.0)
        idx = np.flatnonzero((sims >= kth) & (sims > -0.5))
        cand_ids = tag_ids[idx]
        cand_scores = sims[idx]
        order = np.lexsort((cand_ids, -cand_scores))
        if len(idx) > 30_000:
            log.warning(
                "Tag ties at kth=%.6f exceeded safety cap (%d > 30000); keeping lowest ids",
                kth, len(idx)
            )
            order = order[:30_000]
        top_sorted = idx[order]
    else:
        order = np.lexsort((tag_ids, -sims))
        top_sorted = np.array([r for r in order if sims[r] > -0.5])

    tag_candidates = tag_ids[top_sorted]

    # Restore excluded seed scores so q.tag_full_sims remains true similarities
    for r, val in orig_vals.items():
        sims[r] = val

    dt = round(time.perf_counter() - t0, 4)
    return tag_candidates, dt


def _generate_matvec(q, exclude: set[int], timings: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Retrieves candidates using matvec tag and Chroma plot ANN concurrently on POOL."""
    has_tag = q.tag_sparse is not None
    has_plot = q.plot_centroid is not None

    if not has_tag and not has_plot:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    k_tag = getattr(settings, "k_tag", 2000)
    k_plot = getattr(settings, "k_plot", 2000)

    raw_tag = np.empty(0, dtype=np.int64)
    raw_plot = np.empty(0, dtype=np.int64)

    def _run_plot():
        t0 = time.perf_counter()
        res = chroma.query_ids(settings.plots_collection, q.plot_centroid, k_plot)
        return np.asarray(res, dtype=np.int64), round(time.perf_counter() - t0, 4)

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


def generate(req, q, timings: dict | None = None) -> np.ndarray:
    """Candidate generation using vectorized metadata mask and parallel ANN or matvec.
    Keeps candidates as a numpy int64 array throughout.
    """
    f = Filters.from_request(req)
    exclude = set(q.movie_ids)
    q._meta_rows = None

    # No filters: skip gate/mask entirely
    if not f.active:
        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            pool = _unique_ordered(raw_tag, raw_plot)
            if len(exclude) and len(pool):
                pool = pool[~np.isin(pool, np.array(list(exclude), dtype=np.int64))]
            log.debug("branch=no_filters matvec_tag pool=%d", len(pool))
            return pool
        if settings.use_uncapped_pool:
            raw_lists = _ann(q, None, timings=timings, k=settings.ann_k)
            combined = np.array([i for lst in raw_lists for i in lst], dtype=np.int64)
            if len(combined):
                _, first_idx = np.unique(combined, return_index=True)
                pool = combined[np.sort(first_idx)]
            else:
                pool = np.empty(0, dtype=np.int64)
            if len(exclude) and len(pool):
                pool = pool[~np.isin(pool, np.array(list(exclude), dtype=np.int64))]
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
            return np.empty(0, dtype=np.int64)

        # Small/Exact branch: whole filtered set is the pool, no ANN needed
        if n_matches <= settings.exact_limit:
            meta_rows = np.flatnonzero(mask)
            matching_ids = idx.ids[meta_rows]
            if len(exclude) and len(matching_ids):
                keep_mask = ~np.isin(matching_ids, np.array(list(exclude), dtype=np.int64))
                pool = matching_ids[keep_mask]
                q._meta_rows = meta_rows[keep_mask]
            else:
                pool = matching_ids
                q._meta_rows = meta_rows
            if timings is not None:
                timings["ann_tag"] = 0.0
                timings["ann_plot"] = 0.0
                timings["filter_ids"] = 0.0
            log.debug("branch=small n_matches=%d pool=%d", n_matches, len(pool))
            return pool

        # Large branch: n_matches > exact_limit
        tag_count = 0
        plot_count = 0

        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            t0 = time.perf_counter()

            # Vectorized meta filter for raw_tag
            meta_rows_tag = meta_index.rows_for(raw_tag)
            valid_tag = meta_rows_tag >= 0
            n_missing_tag = int(np.sum(~valid_tag))
            if n_missing_tag > 0:
                log.warning("Dropped %d candidate IDs missing from meta index", n_missing_tag)
            keep_tag = np.zeros(len(raw_tag), dtype=bool)
            keep_tag[valid_tag] = mask[meta_rows_tag[valid_tag]]
            filtered_tag = raw_tag[keep_tag]

            # Vectorized meta filter for raw_plot
            meta_rows_plot = meta_index.rows_for(raw_plot)
            valid_plot = meta_rows_plot >= 0
            n_missing_plot = int(np.sum(~valid_plot))
            if n_missing_plot > 0:
                log.warning("Dropped %d candidate IDs missing from meta index", n_missing_plot)
            keep_plot = np.zeros(len(raw_plot), dtype=bool)
            keep_plot[valid_plot] = mask[meta_rows_plot[valid_plot]]
            filtered_plot = raw_plot[keep_plot]

            filter_time = round(time.perf_counter() - t0, 4)
            if timings is not None:
                timings["filter_ids"] = filter_time

            tag_count = len(filtered_tag)
            plot_count = len(filtered_plot)

            pool = _unique_ordered(filtered_tag, filtered_plot)
            if len(exclude) and len(pool):
                pool = pool[~np.isin(pool, np.array(list(exclude), dtype=np.int64))]
            log.debug("branch=large matvec_tag n_matches=%d pool=%d", n_matches, len(pool))

        elif settings.use_uncapped_pool:
            raw_lists = _ann(q, where=None, timings=timings, k=settings.ann_k)
            t0 = time.perf_counter()
            filtered_parts = []
            for lst in raw_lists:
                arr = np.asarray(lst, dtype=np.int64)
                m_rows = meta_index.rows_for(arr)
                val = m_rows >= 0
                n_miss = int(np.sum(~val))
                if n_miss > 0:
                    log.warning("Dropped %d candidate IDs missing from meta index", n_miss)
                kp = np.zeros(len(arr), dtype=bool)
                kp[val] = mask[m_rows[val]]
                filtered_parts.append(arr[kp])
            filter_time = round(time.perf_counter() - t0, 4)
            if timings is not None:
                timings["filter_ids"] = filter_time

            tag_count = len(filtered_parts[0]) if len(filtered_parts) > 0 else 0
            plot_count = len(filtered_parts[1]) if len(filtered_parts) > 1 else 0

            combined = np.concatenate(filtered_parts) if filtered_parts else np.empty(0, dtype=np.int64)
            if len(combined):
                _, first_idx = np.unique(combined, return_index=True)
                pool = combined[np.sort(first_idx)]
            else:
                pool = np.empty(0, dtype=np.int64)
            if len(exclude) and len(pool):
                pool = pool[~np.isin(pool, np.array(list(exclude), dtype=np.int64))]

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
                    arr = np.asarray(lst, dtype=np.int64)
                    m_rows = meta_index.rows_for(arr)
                    val = m_rows >= 0
                    kp = np.zeros(len(arr), dtype=bool)
                    kp[val] = mask[m_rows[val]]
                    filtered_lists.append(arr[kp].tolist())
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
        has_tag = q.tag_ann is not None or q.tag_sparse is not None
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
                meta_rows = np.flatnonzero(mask)
                matching_ids = idx.ids[meta_rows]
                if len(exclude):
                    keep_mask = ~np.isin(matching_ids, np.array(list(exclude), dtype=np.int64))
                    q._meta_rows = meta_rows[keep_mask]
                    return matching_ids[keep_mask]
                q._meta_rows = meta_rows
                return matching_ids
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
            p_arr = np.asarray([i for i in gate if i not in exclude], dtype=np.int64)
            return p_arr
        if settings.use_matvec_tag_candidates:
            raw_tag, raw_plot = _generate_matvec(q, exclude, timings=timings)
            union = _unique_ordered(raw_tag, raw_plot)
            t0 = time.perf_counter()
            keep = set(queries.filter_ids(union.tolist(), f))
            if timings is not None:
                timings["filter_ids"] = round(time.perf_counter() - t0, 4)
            p_arr = np.asarray([mid for mid in union if mid in keep and mid not in exclude], dtype=np.int64)
            return p_arr
        where = chroma.build_where(
            list(f.languages),
            f.date_from.year if f.date_from else None,
            f.date_to.year if f.date_to else None,
        )
        lists = _ann(q, where, timings=timings)
        union = list(dict.fromkeys(i for lst in lists for i in lst))
        t0 = time.perf_counter()
        keep = set(queries.filter_ids(union, f))
        if timings is not None:
            timings["filter_ids"] = round(time.perf_counter() - t0, 4)
        lists = [[i for i in lst if i in keep] for lst in lists]
        return _interleave(lists, settings.pool_cap, exclude)
