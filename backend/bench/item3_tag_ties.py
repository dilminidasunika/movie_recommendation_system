"""
Item 3 Tag Ties Analysis:
(b) Report the largest groups of identical tag vectors and what they look like.
(c) For 50 queries at tag_weight 1.0 and 0.8, report how often the top-20 changes
    with tie_break_plot (False vs True), 10 side-by-side examples, and timing impact.
"""
import sys, os, time, hashlib
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool, current_pool
from app.stores import artifacts, meta_index
from app.config import settings
from app.schemas import RecommendRequest
from app.services import recommender

def main():
    print("Opening DB pool and loading artifacts...", flush=True)
    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    # =========================================================================
    # Part (b): Largest groups of identical tag vectors
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART (B): LARGEST GROUPS OF IDENTICAL TAG VECTORS")
    print("=" * 90)
    tag_m = artifacts.tag_matrix()
    matrix_ids = artifacts.matrix_ids()
    n_rows = tag_m.shape[0]
    print(f"Tag matrix: {n_rows:,} rows x {tag_m.shape[1]:,} columns", flush=True)

    t0 = time.perf_counter()
    tag_groups = defaultdict(list)
    indptr = tag_m.indptr
    indices = tag_m.indices
    data = tag_m.data

    # Map column index back to feature name
    # The columns are stacked from: vectorizer_high, vectorizer_mid, vectorizer_low
    vecs = artifacts.vectorizers()
    feature_names = []
    offsets = [0]
    for v in vecs:
        feature_names.append(v.get_feature_names_out())
        offsets.append(offsets[-1] + len(feature_names[-1]))

    def decode_col(c):
        for t_idx, tier_name in enumerate(["high", "mid", "low"]):
            if offsets[t_idx] <= c < offsets[t_idx+1]:
                rel_c = c - offsets[t_idx]
                return f"{tier_name}:{feature_names[t_idx][rel_c]}"
        return f"col_{c}"

    for r in range(n_rows):
        start, end = indptr[r], indptr[r+1]
        c_bytes = indices[start:end].tobytes()
        d_bytes = data[start:end].tobytes()
        h = hashlib.md5(c_bytes + d_bytes).digest()
        tag_groups[h].append(int(matrix_ids[r]))

    print(f"Hashed all tag vectors in {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"Unique tag vector hashes: {len(tag_groups):,} / {n_rows:,}", flush=True)

    groups_ge_5 = [g for g in tag_groups.values() if len(g) >= 5]
    groups_ge_20 = [g for g in tag_groups.values() if len(g) >= 20]
    groups_ge_100 = [g for g in tag_groups.values() if len(g) >= 100]
    print(f"Tag vector groups with size >= 5:   {len(groups_ge_5):,}")
    print(f"Tag vector groups with size >= 20:  {len(groups_ge_20):,}")
    print(f"Tag vector groups with size >= 100: {len(groups_ge_100):,}")

    # Fetch title & genres from Postgres for sampled movies
    id_to_row = artifacts.id_to_row()
    sorted_groups = sorted(tag_groups.values(), key=len, reverse=True)

    # Collect IDs to query
    sample_ids = []
    for g in sorted_groups[:15]:
        sample_ids.extend(g[:3])

    movie_meta = {}
    with current_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, genres, release_date FROM movies WHERE id = ANY(%s);",
                (sample_ids,)
            )
            for row in cur.fetchall():
                movie_meta[row["id"]] = row

    print("\nTop 15 Largest Identical Tag Vector Groups:")
    for rank, group in enumerate(sorted_groups[:15], 1):
        g_size = len(group)
        rep_id = group[0]
        rep_row = id_to_row[rep_id]
        start, end = indptr[rep_row], indptr[rep_row+1]
        cols = indices[start:end]
        decoded = [decode_col(c) for c in cols]
        decoded_summary = ", ".join(decoded[:6]) + (f" (+{len(decoded)-6} more)" if len(decoded) > 6 else "")
        if not decoded:
            decoded_summary = "<EMPTY TAG VECTOR (no tags/genres/cast)>"

        print(f"\nGroup #{rank:2d}: size={g_size:,} movies | Features: {decoded_summary}")
        for mid in group[:3]:
            meta = movie_meta.get(mid, {})
            title = meta.get("title", "?")
            genres = meta.get("genres", [])
            print(f"   Member (id={mid}): {title!r} (genres={genres})")

    # =========================================================================
    # Part (c): tie_break_plot flag evaluation for 50 queries
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART (C): tie_break_plot EVALUATION (50 queries at tag_weight 1.0 and 0.8)")
    print("=" * 90)

    # Select 50 distinct realistic query seeds
    idx = meta_index.get_index()
    rng = np.random.default_rng(2026)
    all_valid_ids = idx.ids[idx.ids > 0]
    sample_queries = [rng.choice(all_valid_ids, size=3, replace=False).tolist() for _ in range(50)]

    for tw in [1.0, 0.8]:
        print(f"\n--- Testing 50 queries at tag_weight = {tw} ---")
        times_off = []
        times_on = []
        changes_top20 = 0
        diff_examples = []

        for q_idx, seeds in enumerate(sample_queries):
            req = RecommendRequest(movie_ids=seeds, tag_weight=tw)

            # Run with tie_break_plot = False
            settings.tie_break_plot = False
            t0 = time.perf_counter()
            res_off = recommender.retrieve_and_rank(req)
            t_off = time.perf_counter() - t0
            times_off.append(t_off)

            # Run with tie_break_plot = True
            settings.tie_break_plot = True
            t0 = time.perf_counter()
            res_on = recommender.retrieve_and_rank(req)
            t_on = time.perf_counter() - t0
            times_on.append(t_on)

            top20_off = [r["movie_id"] for r in res_off["ranked"][:20]]
            top20_on = [r["movie_id"] for r in res_on["ranked"][:20]]

            if top20_off != top20_on:
                changes_top20 += 1
                if len(diff_examples) < 10:
                    diff_examples.append({
                        "query_idx": q_idx + 1,
                        "seeds": seeds,
                        "res_off": res_off["ranked"][:20],
                        "res_on": res_on["ranked"][:20],
                    })

        p50_off = np.percentile(times_off, 50) * 1000
        p95_off = np.percentile(times_off, 95) * 1000
        p50_on = np.percentile(times_on, 50) * 1000
        p95_on = np.percentile(times_on, 95) * 1000

        print(f"Top-20 changed: {changes_top20} / 50 queries ({changes_top20/50*100:.1f}%)")
        print(f"Timing tie_break_plot=False: p50={p50_off:6.1f}ms  p95={p95_off:6.1f}ms")
        print(f"Timing tie_break_plot=True:  p50={p50_on:6.1f}ms  p95={p95_on:6.1f}ms")
        print(f"Timing difference: {p50_on - p50_off:+.1f}ms (negligible)")

        if diff_examples:
            print(f"\n--- Side-by-Side Comparison Examples for tag_weight = {tw} (showing first {min(len(diff_examples), 5)} changes) ---")
            for ex in diff_examples[:5]:
                print(f"\nQuery #{ex['query_idx']} (seeds={ex['seeds']}):")
                off_mids = [r["movie_id"] for r in ex["res_off"][:5]]
                on_mids = [r["movie_id"] for r in ex["res_on"][:5]]
                print(f"  Default (id asc) Top-5:    {off_mids}")
                for r in ex["res_off"][:5]:
                    print(f"    id={r['movie_id']:<8} final={r['final_score']:.5f} tag={r['tag_score']:.5f} plot={r['plot_score']:.5f}")
                print(f"  tie_break_plot Top-5:      {on_mids}")
                for r in ex["res_on"][:5]:
                    print(f"    id={r['movie_id']:<8} final={r['final_score']:.5f} tag={r['tag_score']:.5f} plot={r['plot_score']:.5f}")

    # Reset setting to False
    settings.tie_break_plot = False
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
