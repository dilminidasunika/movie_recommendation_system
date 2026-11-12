"""
Build offline artifact plot_suppressed.npy aligned with plot memmap rows.
A plot embedding row is suppressed if:
- has_thin_overview = true in Postgres, OR
- identical-vector group size >= 20
"""
import sys, os, time, hashlib
from collections import Counter
import numpy as np
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.config import settings
from app.db.pool import open_pool, current_pool
from app.stores import artifacts

def main():
    print("=" * 80)
    print("BUILDING OFFLINE ARTIFACT: plot_suppressed.npy")
    print("=" * 80)

    open_pool()
    artifacts.load_all()

    plot_ids = artifacts.plot_ids()
    plot_memmap, _ = artifacts.plot_memmap()
    n_plots = len(plot_ids)
    print(f"Total plot embeddings in memmap: {n_plots:,}")

    # 1. Fetch thin overview IDs from Postgres
    print("\nFetching movie IDs with has_thin_overview = true from Postgres...", flush=True)
    t0 = time.perf_counter()
    thin_ids = set()
    with current_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM movies WHERE has_thin_overview = true;")
            for row in cur.fetchall():
                thin_ids.add(row["id"])
    print(f"Found {len(thin_ids):,} movies with has_thin_overview = true in DB ({time.perf_counter() - t0:.2f}s)")

    # 2. Hash plot vectors to find groups with size >= 20
    print("\nHashing plot vectors in chunks of 50,000...", flush=True)
    t0 = time.perf_counter()
    chunk_size = 50_000
    hashes = []
    vector_counter = Counter()

    for start in range(0, n_plots, chunk_size):
        end = min(start + chunk_size, n_plots)
        chunk = plot_memmap[start:end]
        for row_vec in chunk:
            h = hashlib.md5(row_vec.tobytes()).digest()
            hashes.append(h)
            vector_counter[h] += 1
        if end % 200_000 == 0 or end == n_plots:
            print(f"  Processed {end:,} / {n_plots:,} vectors...")

    print(f"Hashed {len(hashes):,} vectors in {time.perf_counter() - t0:.2f}s")
    large_groups = {h for h, count in vector_counter.items() if count >= 20}
    print(f"Found {len(large_groups):,} vector groups with size >= 20")

    # 3. Construct boolean suppression array aligned with plot memmap rows
    print("\nConstructing plot_suppressed boolean array...", flush=True)
    suppressed = np.zeros(n_plots, dtype=bool)
    n_thin = 0
    n_group20 = 0

    for r in range(n_plots):
        mid = int(plot_ids[r])
        h = hashes[r]
        is_thin = mid in thin_ids
        is_g20 = h in large_groups
        if is_thin:
            n_thin += 1
        if is_g20:
            n_group20 += 1
        if is_thin or is_g20:
            suppressed[r] = True

    total_suppressed = int(np.sum(suppressed))
    pct = total_suppressed / n_plots * 100.0

    print("\n" + "=" * 80)
    print("SUPPRESSION SUMMARY")
    print("=" * 80)
    print(f"Total plot embeddings:           {n_plots:,}")
    print(f"Suppressed via has_thin_overview: {n_thin:,}")
    print(f"Suppressed via group size >= 20:  {n_group20:,}")
    print(f"Total Suppressed Plots:          {total_suppressed:,} ({pct:.2f}% of plot catalog)")

    out_path = settings.artifact_path / "plot_suppressed.npy"
    np.save(out_path, suppressed)
    print(f"\nSaved artifact to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("Verification:")
    loaded = np.load(out_path)
    print(f"  Loaded shape: {loaded.shape}, dtype: {loaded.dtype}, sum: {np.sum(loaded):,}")
    assert np.array_equal(suppressed, loaded)
    print("Artifact successfully verified!")

if __name__ == "__main__":
    main()
