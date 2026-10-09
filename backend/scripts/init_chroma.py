"""
Builds chroma_data (tags + plots) from the clean .npy artifacts.
Runs inside the backend container on first startup only.

Usage: python -m scripts.init_chroma
Env:   CHROMA_DIR, ARTIFACTS_DIR, DATABASE_URL, ALLOW_NO_META=1 (skip metadata, NOT recommended)
"""
import os
import sys

import chromadb
import numpy as np

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/chroma_data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/artifacts/v1")
DATABASE_URL = os.environ.get("DATABASE_URL")
ALLOW_NO_META = os.environ.get("ALLOW_NO_META") == "1"
MARKER = os.path.join(CHROMA_DIR, ".linux_ready")

TAGS_NAME, PLOTS_NAME = "tags_vectors", "plots_embeddings"
SPECS = [
    # (collection, vectors file, ids file, expected dim)
    (TAGS_NAME,  "tag_matrix_svd_reduced_clean.npy", "tag_svd_movie_ids_clean.npy",  300),
    (PLOTS_NAME, "plot_embeddings.npy",              "plot_embeddings_movie_ids.npy", 384),
]
BATCH = 5000
MIN_SELF_RECALL = 0.97


def load_metadata() -> dict:
    """id -> {"original_language": str, "release_year": int}; null keys omitted."""
    if not DATABASE_URL:
        if ALLOW_NO_META:
            print("WARNING: no DATABASE_URL, building WITHOUT filter metadata.")
            return {}
        sys.exit("DATABASE_URL not set. Filtered ANN needs metadata. "
                 "Set it, or set ALLOW_NO_META=1 to skip.")
    import psycopg2
    meta = {}
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(name="meta_cur") as cur:  # server-side cursor
            cur.itersize = 10000
            cur.execute("SELECT id, original_language, release_year FROM movies")
            for mid, lang, yr in cur:
                m = {}
                if lang:
                    m["original_language"] = str(lang).lower()
                if yr is not None:
                    m["release_year"] = int(yr)
                if m:
                    meta[int(mid)] = m
    finally:
        conn.close()
    print(f"Loaded metadata for {len(meta):,} movies from Postgres")
    return meta


def load_and_validate(vec_file, ids_file, dim, name):
    X = np.load(os.path.join(ARTIFACTS_DIR, vec_file), mmap_mode="r")
    ids = np.load(os.path.join(ARTIFACTS_DIR, ids_file), allow_pickle=False)

    assert X.ndim == 2 and X.shape[1] == dim, f"[{name}] bad shape {X.shape}, expected dim {dim}"
    assert X.shape[0] == len(ids), f"[{name}] rows {X.shape[0]} != ids {len(ids)}"
    assert len(np.unique(ids)) == len(ids), f"[{name}] duplicate ids"

    for i in range(0, len(ids), 50000):  # touches every page, catches truncation
        C = np.asarray(X[i:i + 50000], dtype=np.float32)
        assert np.isfinite(C).all(), f"[{name}] non-finite values near row {i}"
        n = np.linalg.norm(C, axis=1)
        assert (np.abs(n - 1) < 1e-3).all(), f"[{name}] rows not L2-normalized near row {i}"
    print(f"[{name}] source OK: {X.shape}")
    return X, ids


def build_collection(client, name, X, ids, meta, batch):
    try:
        client.delete_collection(name)
        print(f"[{name}] deleted existing collection")
    except Exception:
        pass

    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    total = len(ids)
    for i in range(0, total, batch):
        j = min(i + batch, total)
        emb = np.asarray(X[i:j], dtype=np.float32)
        sid = [str(int(x)) for x in ids[i:j]]
        md = [meta.get(int(x)) for x in ids[i:j]]

        # Chroma rejects empty metadata dicts, so rows without metadata go in a separate call
        with_idx = [k for k, m in enumerate(md) if m]
        without_idx = [k for k, m in enumerate(md) if not m]
        if with_idx:
            col.add(ids=[sid[k] for k in with_idx],
                    embeddings=emb[with_idx],
                    metadatas=[md[k] for k in with_idx])
        if without_idx:
            col.add(ids=[sid[k] for k in without_idx], embeddings=emb[without_idx])
        if (i // batch) % 20 == 0 or j == total:
            print(f"[{name}] {j:,}/{total:,}")
    return col


def verify(col, X, ids, name):
    n = col.count()
    assert n == len(ids), f"[{name}] count {n} != expected {len(ids)}"

    # every stored vector must be readable and finite
    for off in range(0, n, 5000):
        r = col.get(limit=5000, offset=off, include=["embeddings"])
        E = np.asarray(r["embeddings"], dtype=np.float32)
        assert np.isfinite(E).all(), f"[{name}] unreadable/non-finite rows near offset {off}"

    # HNSW self-recall on a random sample
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(ids), size=min(300, len(ids)), replace=False))
    res = col.query(query_embeddings=np.asarray(X[pick], dtype=np.float32), n_results=1)
    hits = sum(res["ids"][k][0] == str(int(ids[j])) for k, j in enumerate(pick))
    recall = hits / len(pick)
    print(f"[{name}] count OK ({n:,}), self-recall@1 = {recall:.3f}")
    assert recall >= MIN_SELF_RECALL, f"[{name}] self-recall too low: {recall:.3f}"


def main():
    if os.path.exists(MARKER):
        print("ChromaDB already initialized. Skipping.")
        return

    os.makedirs(CHROMA_DIR, exist_ok=True)
    meta = load_metadata()

    # validate ALL sources before writing anything
    loaded = {}
    for name, vec_file, ids_file, dim in SPECS:
        loaded[name] = load_and_validate(vec_file, ids_file, dim, name)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    batch = min(BATCH, client.get_max_batch_size())

    for name, *_ in SPECS:
        X, ids = loaded[name]
        col = build_collection(client, name, X, ids, meta, batch)
        verify(col, X, ids, name)

    if meta:  # filter sanity: must return results
        for name, *_ in SPECS:
            X, _ = loaded[name]
            r = client.get_collection(name).query(
                query_embeddings=np.asarray(X[:1], dtype=np.float32),
                n_results=5, where={"original_language": {"$in": ["en"]}})
            assert r["ids"][0], f"[{name}] filtered query returned nothing"
        print("Filtered query sanity OK")

    with open(MARKER, "w") as f:  # only after everything passed
        f.write("ok")
    print("\nchroma_data built and verified.")


if __name__ == "__main__":
    main()