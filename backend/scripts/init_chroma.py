"""
Builds chroma_data (tags + plots) from the clean .npy artifacts.

Changes vs the old version:
  - each collection is built in its OWN subprocess, so memory is fully released between them
  - per-collection ".done" marker: a finished collection is skipped on rerun (resume)
  - a half-built collection is always deleted and rebuilt, never resumed mid-way
  - progress every 20k rows, smaller batches
  - read-back check uses a random sample instead of paging the whole collection with offset

Usage: python -m scripts.init_chroma
Env:   CHROMA_DIR, ARTIFACTS_DIR, DATABASE_URL, ALLOW_NO_META=1, MIN_RECALL (default 0.85)
"""
import os
import subprocess
import sys

import chromadb
import numpy as np

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/chroma_data")
ARTIFACTS_DIR = os.path.join(os.environ.get("ARTIFACTS_DIR", "/artifacts"), "v1")
DATABASE_URL = os.environ.get("DATABASE_URL")
ALLOW_NO_META = os.environ.get("ALLOW_NO_META") == "1"
MARKER = os.path.join(CHROMA_DIR, ".linux_ready")

SPECS = [
    # (collection, vectors file, ids file, expected dim)
    ("tags_vectors",      "tag_matrix_svd_reduced_clean.npy", "tag_svd_movie_ids_clean.npy",   300),
    ("plots_embeddings",  "plot_embeddings.npy",              "plot_embeddings_movie_ids.npy", 384),
]
BATCH = 2000
LOG_EVERY = 20000
MIN_RECALL = float(os.environ.get("MIN_RECALL", "0.85"))


def done_marker(name):
    return os.path.join(CHROMA_DIR, f".{name}.done")


def load_metadata() -> dict:
    """id -> {"original_language": str, "release_date": int}; null keys omitted."""
    if not DATABASE_URL:
        if ALLOW_NO_META:
            print("WARNING: no DATABASE_URL, building WITHOUT filter metadata.", flush=True)
            return {}
        sys.exit("DATABASE_URL not set. Filtered ANN needs metadata. "
                 "Set it, or set ALLOW_NO_META=1 to skip.")
    import psycopg2
    meta = {}
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(name="meta_cur") as cur:  # server-side cursor
            cur.itersize = 10000
            cur.execute("SELECT id, original_language, release_date FROM movies")
            for mid, lang, rd in cur:
                m = {}
                if lang:
                    m["original_language"] = sys.intern(str(lang).lower())
                if rd is not None:
                    m["release_date"] = rd.year
                if m:
                    meta[int(mid)] = m
    finally:
        conn.close()
    print(f"Loaded metadata for {len(meta):,} movies from Postgres", flush=True)
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
    print(f"[{name}] source OK: {X.shape}", flush=True)
    return X, ids


def create_collection(client, name):
    # Newer Chroma (1.x) uses `configuration`; older versions use hnsw:* metadata.
    try:
        return client.create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine",
                                    "ef_construction": 200,
                                    "ef_search": 100}},
        )
    except Exception as e:
        print(f"[{name}] configuration= not accepted ({type(e).__name__}), using metadata", flush=True)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine",
                      "hnsw:construction_ef": 200,
                      "hnsw:search_ef": 100},
        )


def build_collection(client, name, X, ids, meta, batch):
    try:
        client.delete_collection(name)
        print(f"[{name}] deleted existing collection", flush=True)
    except Exception:
        pass

    col = create_collection(client, name)
    total = len(ids)
    next_log = LOG_EVERY
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
        if j >= next_log or j == total:
            print(f"[{name}] {j:,}/{total:,}", flush=True)
            next_log += LOG_EVERY
    return col


def verify(col, X, ids, name):
    n = col.count()
    assert n == len(ids), f"[{name}] count {n} != expected {len(ids)}"
    rng = np.random.default_rng(0)

    # read-back: random sample of stored vectors must match the source
    samp = np.sort(rng.choice(len(ids), size=min(2000, len(ids)), replace=False))
    r = col.get(ids=[str(int(ids[j])) for j in samp], include=["embeddings"])
    assert len(r["ids"]) == len(samp), f"[{name}] missing rows on read-back"
    got = dict(zip(r["ids"], r["embeddings"]))
    for j in samp[:200]:
        E = np.asarray(got[str(int(ids[j]))], dtype=np.float32)
        assert np.allclose(E, np.asarray(X[j], dtype=np.float32), atol=1e-4),             f"[{name}] stored vector differs from source at row {j}"

    # HNSW recall vs exact brute force, tie-safe (compares distances, not ids)
    pick = np.sort(rng.choice(len(ids), size=min(300, len(ids)), replace=False))
    Q = np.asarray(X[pick], dtype=np.float32)
    best = np.full(len(pick), -np.inf, dtype=np.float32)
    for i in range(0, len(ids), 50000):
        C = np.asarray(X[i:i + 50000], dtype=np.float32)
        best = np.maximum(best, (Q @ C.T).max(axis=1))
    exact_dist = 1.0 - best

    res = col.query(query_embeddings=Q, n_results=1, include=["distances"])
    got_dist = np.array([d[0] for d in res["distances"]])
    recall = int((got_dist <= exact_dist + 1e-4).sum()) / len(pick)
    print(f"[{name}] count OK ({n:,}), recall@1 (tie-safe) = {recall:.3f}", flush=True)
    assert recall >= MIN_RECALL, f"[{name}] recall too low: {recall:.3f} (min {MIN_RECALL})"


def build_one(name):
    """Runs in a child process: builds + verifies one collection, then writes its .done marker."""
    spec = next(s for s in SPECS if s[0] == name)
    _, vec_file, ids_file, dim = spec
    meta = load_metadata()
    X, ids = load_and_validate(vec_file, ids_file, dim, name)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    batch = min(BATCH, client.get_max_batch_size())
    col = build_collection(client, name, X, ids, meta, batch)
    del meta
    verify(col, X, ids, name)

    # Sanity check filter while collection is already open in memory
    if DATABASE_URL:
        print(f"[{name}] checking filter sanity...", flush=True)
        r = col.query(
            query_embeddings=np.asarray(X[:1], dtype=np.float32),
            n_results=5,
            where={"original_language": {"$in": ["en"]}},
        )
        assert r["ids"][0], f"[{name}] filtered query returned nothing"
        print(f"[{name}] filter sanity OK", flush=True)

    with open(done_marker(name), "w") as f:
        f.write("ok")
    print(f"[{name}] verified and done marker written", flush=True)

    # Force cleanup and instant process exit to avoid Python interpreter
    # hanging on ChromaDB's lingering background worker threads / locks.
    if hasattr(client, "_system"):
        try:
            client._system.stop()
        except Exception:
            pass
    del col
    del client
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--only":
        build_one(sys.argv[2])
        return

    if os.path.exists(MARKER):
        print("ChromaDB already initialized. Skipping.")
        return

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # validate ALL sources before writing anything
    for name, vec_file, ids_file, dim in SPECS:
        load_and_validate(vec_file, ids_file, dim, name)

    for name, *_ in SPECS:
        if os.path.exists(done_marker(name)):
            print(f"[{name}] already built and verified, skipping", flush=True)
            continue
        print(f"[{name}] starting build subprocess...", flush=True)
        r = subprocess.run([sys.executable, "-m", "scripts.init_chroma", "--only", name])
        if r.returncode != 0:
            sys.exit(f"[{name}] build failed (exit {r.returncode}); rerun to retry this collection")

    with open(MARKER, "w") as f:  # only after everything passed
        f.write("ok")
    print("\nchroma_data built and verified.", flush=True)


if __name__ == "__main__":
    main()
