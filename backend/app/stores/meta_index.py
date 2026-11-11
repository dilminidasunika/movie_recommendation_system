from __future__ import annotations

import logging
import time
from typing import NamedTuple

import numpy as np

from app.db.pool import current_pool
from app.db.queries import Filters

log = logging.getLogger(__name__)


class MetaIndex(NamedTuple):
    ids: np.ndarray             # int64, shape (N,)
    id_to_row: dict[int, int]   # movie_id -> row index
    languages: np.ndarray       # uint16, shape (N,)
    lang_to_code: dict[str, int]
    code_to_lang: dict[int, str]
    years: np.ndarray           # int16, shape (N,), sentinel -1 for NULL
    dates: np.ndarray           # int32, shape (N,), ordinal date, sentinel 0 for NULL
    genres: np.ndarray          # uint32, shape (N,), bitmask
    genre_to_bit: dict[str, int]
    bit_to_genre: dict[int, str]
    n_movies: int


_INDEX: MetaIndex | None = None


def load_index() -> MetaIndex:
    """Stream movies metadata from Postgres and build fast in-memory numpy index."""
    global _INDEX
    t0 = time.perf_counter()
    log.info("Loading MetaIndex from Postgres...")

    # Fetch unique genres and languages first to establish code mappings
    with current_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT original_language FROM movies WHERE original_language IS NOT NULL")
            raw_langs = [r["original_language"].strip().lower() for r in cur.fetchall() if r["original_language"]]
            all_langs = sorted(list(set(raw_langs)))

            cur.execute("SELECT DISTINCT unnest(genres) as g FROM movies WHERE genres IS NOT NULL")
            raw_genres = [r["g"].strip().lower() for r in cur.fetchall() if r["g"]]
            all_genres = sorted(list(set(raw_genres)))

    lang_to_code = {l: (i + 1) for i, l in enumerate(all_langs)}
    code_to_lang = {c: l for l, c in lang_to_code.items()}

    genre_to_bit = {g: (1 << i) for i, g in enumerate(all_genres)}
    bit_to_genre = {b: g for g, b in genre_to_bit.items()}

    # Stream full table via server-side cursor
    id_list: list[int] = []
    lang_list: list[int] = []
    year_list: list[int] = []
    date_list: list[int] = []
    genre_list: list[int] = []

    with current_pool().connection() as conn:
        with conn.cursor(name="meta_stream") as cur:
            cur.execute("SELECT id, original_language, release_date, genres FROM movies ORDER BY id")
            while True:
                chunk = cur.fetchmany(50000)
                if not chunk:
                    break
                for r in chunk:
                    mid = int(r["id"])
                    id_list.append(mid)

                    lang = r["original_language"]
                    lang_list.append(lang_to_code.get(lang.strip().lower(), 0) if lang else 0)

                    d = r["release_date"]
                    if d:
                        year_list.append(d.year)
                        date_list.append(d.toordinal())
                    else:
                        year_list.append(-1)
                        date_list.append(0)

                    g_arr = r["genres"]
                    if g_arr:
                        gmask = 0
                        for g in g_arr:
                            b = genre_to_bit.get(g.strip().lower())
                            if b:
                                gmask |= b
                        genre_list.append(gmask)
                    else:
                        genre_list.append(0)

    n = len(id_list)
    ids = np.array(id_list, dtype=np.int64)
    languages = np.array(lang_list, dtype=np.uint16)
    years = np.array(year_list, dtype=np.int16)
    dates = np.array(date_list, dtype=np.int32)
    genres = np.array(genre_list, dtype=np.uint32)
    id_to_row = {mid: i for i, mid in enumerate(id_list)}

    bytes_total = ids.nbytes + languages.nbytes + years.nbytes + dates.nbytes + genres.nbytes
    mb_arrays = bytes_total / (1024 * 1024)

    _INDEX = MetaIndex(
        ids=ids,
        id_to_row=id_to_row,
        languages=languages,
        lang_to_code=lang_to_code,
        code_to_lang=code_to_lang,
        years=years,
        dates=dates,
        genres=genres,
        genre_to_bit=genre_to_bit,
        bit_to_genre=bit_to_genre,
        n_movies=n,
    )

    t_load = time.perf_counter() - t0
    log.info(
        "MetaIndex loaded in %.2fs: %d movies, arrays=%.2f MB (ids=%.1fMB, lang=%.1fMB, years=%.1fMB, dates=%.1fMB, genres=%.1fMB)",
        t_load,
        n,
        mb_arrays,
        ids.nbytes / (1024 * 1024),
        languages.nbytes / (1024 * 1024),
        years.nbytes / (1024 * 1024),
        dates.nbytes / (1024 * 1024),
        genres.nbytes / (1024 * 1024),
    )
    return _INDEX


def is_loaded() -> bool:
    return _INDEX is not None


def get_index() -> MetaIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index()
    return _INDEX


def mask_for(f: Filters) -> np.ndarray:
    """Compute boolean mask matching EXACT Postgres filter semantics."""
    idx = get_index()
    mask = np.ones(idx.n_movies, dtype=bool)

    if f.languages:
        codes = [idx.lang_to_code[l.lower()] for l in f.languages if l.lower() in idx.lang_to_code]
        if codes:
            mask &= np.isin(idx.languages, codes)
        else:
            return np.zeros(idx.n_movies, dtype=bool)

    if f.genres:
        req_gmask = 0
        for g in f.genres:
            b = idx.genre_to_bit.get(g.lower())
            if b:
                req_gmask |= b
        if req_gmask:
            mask &= ((idx.genres & req_gmask) != 0)
        else:
            return np.zeros(idx.n_movies, dtype=bool)

    if f.date_from:
        d_from = f.date_from.toordinal()
        mask &= (idx.dates >= d_from)

    if f.date_to:
        d_to = f.date_to.toordinal()
        mask &= (idx.dates > 0) & (idx.dates <= d_to)

    return mask


def filter_ids_with_mask(candidate_ids: list[int], f: Filters) -> list[int]:
    """Fast in-memory filter for candidate IDs using the meta index mask."""
    if not candidate_ids:
        return []
    if not f.active:
        return candidate_ids

    idx = get_index()
    # If filter is empty or matches nothing
    mask = mask_for(f)
    i2r = idx.id_to_row

    out = []
    for mid in candidate_ids:
        row = i2r.get(mid)
        if row is not None and mask[row]:
            out.append(mid)
    return out
