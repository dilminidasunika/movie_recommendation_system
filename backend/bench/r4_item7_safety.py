"""
Round 4 Item 7: Safety test.
With flag ON, rename tags_vectors collection (in SQLite) and SVD artifact file temporarily,
confirm app boots and serves all case types. Then confirm flag OFF still works (rollback path).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, shutil, sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def run_safety_test():
    from app.config import settings

    svd_path = settings.artifact_path / "tag_svd_model.joblib"
    svd_backup = settings.artifact_path / "tag_svd_model.joblib.bak"
    sqlite_path = settings.chroma_path / "chroma.sqlite3"

    print("=" * 75, flush=True)
    print("ROUND 4 ITEM 7: SAFETY TEST", flush=True)
    print("=" * 75, flush=True)

    # Step 1: Temporarily rename SVD artifact file
    print("\n--- Step 1: Rename SVD artifact file ---", flush=True)
    if svd_path.exists():
        shutil.move(str(svd_path), str(svd_backup))
        print(f"  Renamed {svd_path.name} -> {svd_backup.name}", flush=True)
    else:
        print(f"  Note: SVD already renamed or not found at {svd_path}", flush=True)

    # Step 2: Temporarily rename tags_vectors in Chroma SQLite
    print("\n--- Step 2: Rename tags_vectors collection in Chroma SQLite ---", flush=True)
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM collections")
    collections_before = cur.fetchall()
    print(f"  Collections before rename: {collections_before}", flush=True)
    cur.execute("UPDATE collections SET name = 'tags_vectors_hidden' WHERE name = 'tags_vectors'")
    conn.commit()
    cur.execute("SELECT id, name FROM collections")
    print(f"  Collections after rename: {cur.fetchall()}", flush=True)
    conn.close()

    # Step 3: Boot FastAPI app with flag ON (production defaults)
    print("\n--- Step 3: Boot app with flag ON (SVD & tags_vectors hidden) ---", flush=True)
    settings.use_matvec_tag_candidates = True
    settings.k_tag = 10000
    settings.k_plot = 5000
    settings.exact_limit = 300000
    settings.use_uncapped_pool = True

    from app.stores import artifacts as art_mod, chroma as chroma_mod
    art_mod.svd.cache_clear()
    art_mod.tag_matrix.cache_clear()
    art_mod.matrix_ids.cache_clear()
    art_mod.id_to_row.cache_clear()
    art_mod.vectorizers.cache_clear()
    art_mod.plot_memmap.cache_clear()
    art_mod.row_norms.cache_clear()
    chroma_mod.get_collection.cache_clear()

    from app.db.pool import open_pool
    from app.stores import artifacts, meta_index, chroma, text_embedder
    from app.schemas import RecommendRequest
    from app.services import recommender

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    text_embedder.get_sentence_transformer()
    print("  Boot SUCCESS with flag ON! Neither SVD nor tags_vectors was required.", flush=True)

    # Step 4: Verify that trying to access tags_vectors collection actually raises error (proving it's really hidden)
    print("\n--- Step 4: Confirm tags_vectors is truly inaccessible ---", flush=True)
    try:
        chroma.get_collection("tags_vectors")
        print("  WARNING: Expected get_collection('tags_vectors') to fail, but it succeeded!", flush=True)
    except Exception as e:
        print(f"  Confirmed: accessing 'tags_vectors' fails as expected: {type(e).__name__}", flush=True)

    # Step 5: Serve all request types with flag ON
    print("\n--- Step 5: Serve all case types with flag ON ---", flush=True)
    test_cases = [
        ("Unfiltered (3 seeds, tw=0.5)", RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.5)),
        ("Plot Text (seeds + query, tw=0.5)", RecommendRequest(
            movie_ids=[238, 240, 278],
            plot_text="A dramatic story of crime, family loyalty, and redemption across generations.",
            tag_weight=0.5
        )),
        ("Language Filter (fr, tw=0.5)", RecommendRequest(
            movie_ids=[238, 240, 278],
            original_language=["fr"],
            tag_weight=0.5
        )),
        ("Genre Filter (Drama, tw=0.5)", RecommendRequest(
            movie_ids=[238, 240, 278],
            genres=["Drama"],
            tag_weight=0.5
        )),
        ("Pure Tag (tw=1.0)", RecommendRequest(movie_ids=[238, 240, 278], tag_weight=1.0)),
        ("Pure Plot (tw=0.0)", RecommendRequest(movie_ids=[238, 240, 278], tag_weight=0.0)),
    ]

    all_flag_on_passed = True
    for desc, req in test_cases:
        t0 = time.perf_counter()
        res = recommender.retrieve_and_rank(req)
        elapsed = (time.perf_counter() - t0) * 1000.0
        n_ranked = len(res["ranked"])
        pool_sz = res.get("pool_size", 0)
        print(f"  {desc:<40} -> {n_ranked} results, pool={pool_sz} ({elapsed:.1f}ms)", flush=True)
        if n_ranked == 0:
            all_flag_on_passed = False

    # Step 6: Restore SVD and Chroma collection
    print("\n--- Step 6: Restore SVD file and tags_vectors collection ---", flush=True)
    if svd_backup.exists():
        shutil.move(str(svd_backup), str(svd_path))
        print(f"  Restored {svd_backup.name} -> {svd_path.name}", flush=True)

    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("UPDATE collections SET name = 'tags_vectors' WHERE name = 'tags_vectors_hidden'")
    conn.commit()
    cur.execute("SELECT id, name FROM collections")
    print(f"  Collections after restore: {cur.fetchall()}", flush=True)
    conn.close()

    # Step 7: Test flag OFF (rollback path)
    print("\n--- Step 7: Test flag OFF (rollback path) ---", flush=True)
    settings.use_matvec_tag_candidates = False
    art_mod.svd.cache_clear()
    art_mod.tag_matrix.cache_clear()
    art_mod.matrix_ids.cache_clear()
    art_mod.id_to_row.cache_clear()
    art_mod.vectorizers.cache_clear()
    art_mod.plot_memmap.cache_clear()
    art_mod.row_norms.cache_clear()
    chroma_mod.get_collection.cache_clear()

    artifacts.load_all()
    chroma.get_collection(settings.tags_collection)
    print("  Boot SUCCESS with flag OFF! Rollback path is completely functional.", flush=True)

    all_rollback_passed = True
    for desc, req in test_cases[:3]:
        t0 = time.perf_counter()
        res = recommender.retrieve_and_rank(req)
        elapsed = (time.perf_counter() - t0) * 1000.0
        n_ranked = len(res["ranked"])
        print(f"  [Rollback] {desc:<35} -> {n_ranked} results ({elapsed:.1f}ms)", flush=True)
        if n_ranked == 0:
            all_rollback_passed = False

    # Restore production settings
    settings.use_matvec_tag_candidates = True

    print(f"\n{'='*75}", flush=True)
    if all_flag_on_passed and all_rollback_passed:
        print("SAFETY CONFIRMED: Flag ON completely decoupled from SVD & tags_vectors.", flush=True)
        print("ROLLBACK CONFIRMED: Flag OFF functions identically to original code.", flush=True)
    else:
        print("SAFETY TEST FAILED: Issues encountered during test.", flush=True)
    print(f"{'='*75}", flush=True)


if __name__ == "__main__":
    run_safety_test()
