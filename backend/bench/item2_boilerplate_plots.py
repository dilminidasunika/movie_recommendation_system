"""
Item 2 Analysis: Duplicate/Boilerplate Plots (report only)
1. Group 1,023,484 plot embeddings by exact vector (hash float32 bytes)
2. Group by exact overview-text hash from Postgres
3. Report number of groups >= 5, 20 largest groups with group size,
   overview text of 3 members, and count of has_thin_overview = True
4. For query with seeds [1108020, 470285, 156627], show overview text of top 10 results
5. Propose a rule and report how many movies it would affect
"""
import sys, os, time, hashlib
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool, current_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import recommender

def main():
    print("Opening DB pool and loading artifacts...", flush=True)
    open_pool()
    artifacts.load_all()

    # --- 1. Load plot embeddings and group by exact vector bytes ---
    print("\n--- 1. Grouping plot embeddings by exact vector hash ---", flush=True)
    t0 = time.perf_counter()
    emb, _ = artifacts.plot_memmap()
    plot_ids = np.load(artifacts._p(artifacts.PLOT_IDS))
    n_embs = len(plot_ids)
    print(f"Loaded {n_embs:,} plot embeddings ({emb.shape}) in {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    # Hash each 384 float32 vector (1536 bytes)
    # Using hashlib.md5 for collision resistance
    vec_to_mids = defaultdict(list)
    # Process in chunks for speed and progress reporting
    chunk_size = 50_000
    for start in range(0, n_embs, chunk_size):
        end = min(start + chunk_size, n_embs)
        chunk = emb[start:end]
        c_ids = plot_ids[start:end]
        for i in range(len(c_ids)):
            h = hashlib.md5(chunk[i].tobytes()).digest()
            vec_to_mids[h].append(int(c_ids[i]))
        if (start // chunk_size) % 5 == 0:
            print(f"  Processed {end:,} / {n_embs:,} vectors...", flush=True)

    print(f"Hashed all vectors in {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"Unique vector hashes: {len(vec_to_mids):,} / {n_embs:,}", flush=True)

    vec_groups_ge_5 = [g for g in vec_to_mids.values() if len(g) >= 5]
    vec_groups_ge_2 = [g for g in vec_to_mids.values() if len(g) >= 2]
    vec_groups_ge_20 = [g for g in vec_to_mids.values() if len(g) >= 20]
    print(f"Vector groups with size >= 2:  {len(vec_groups_ge_2):,}")
    print(f"Vector groups with size >= 5:  {len(vec_groups_ge_5):,}")
    print(f"Vector groups with size >= 20: {len(vec_groups_ge_20):,}")

    # --- 2. Query Postgres for overview and has_thin_overview ---
    print("\n--- 2. Querying Postgres for overview and has_thin_overview ---", flush=True)
    t0 = time.perf_counter()
    overview_map = {}       # id -> overview
    thin_overview_map = {}  # id -> bool
    text_to_ids = defaultdict(list)

    with current_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, overview, has_thin_overview FROM movies;")
            for row in cur.fetchall():
                mid = row["id"]
                ov = row["overview"] or ""
                thin = bool(row["has_thin_overview"])
                overview_map[mid] = ov
                thin_overview_map[mid] = thin
                # Group by exact overview text
                h_text = hashlib.md5(ov.strip().encode("utf-8")).digest()
                text_to_ids[h_text].append(mid)

    print(f"Fetched {len(overview_map):,} movies from Postgres in {time.perf_counter() - t0:.2f}s", flush=True)
    text_groups_ge_5 = [g for g in text_to_ids.values() if len(g) >= 5]
    text_groups_ge_20 = [g for g in text_to_ids.values() if len(g) >= 20]
    print(f"Overview text groups with size >= 5:  {len(text_groups_ge_5):,}")
    print(f"Overview text groups with size >= 20: {len(text_groups_ge_20):,}")

    # Total catalog stats for has_thin_overview
    total_thin = sum(1 for v in thin_overview_map.values() if v)
    print(f"Total movies with has_thin_overview = True in DB: {total_thin:,} / {len(thin_overview_map):,} ({total_thin/len(thin_overview_map)*100:.2f}%)")

    # --- 3. Report the 20 largest vector groups ---
    print("\n" + "=" * 90)
    print("TOP 20 LARGEST VECTOR GROUPS")
    print("=" * 90)

    # Sort vector groups by size descending
    sorted_vec_groups = sorted(vec_to_mids.values(), key=len, reverse=True)
    for rank, group in enumerate(sorted_vec_groups[:20], 1):
        g_size = len(group)
        thin_count = sum(1 for mid in group if thin_overview_map.get(mid, False))
        print(f"\nGroup #{rank:2d}: size={g_size:,} | has_thin_overview={thin_count:,}/{g_size:,} ({thin_count/g_size*100:.1f}%)")
        # Sample 3 members
        samples = group[:3]
        for s_idx, mid in enumerate(samples, 1):
            ov = overview_map.get(mid, "<NO OVERVIEW IN DB>")
            thin = thin_overview_map.get(mid, False)
            ov_preview = (ov[:100] + "...") if len(ov) > 100 else (ov or "<EMPTY>")
            print(f"   Member {s_idx} (id={mid}, thin={thin}): {ov_preview!r}")

    # Also check top 5 text groups
    print("\n" + "=" * 90)
    print("TOP 5 LARGEST OVERVIEW TEXT GROUPS")
    print("=" * 90)
    sorted_text_groups = sorted(text_to_ids.values(), key=len, reverse=True)
    for rank, group in enumerate(sorted_text_groups[:5], 1):
        g_size = len(group)
        thin_count = sum(1 for mid in group if thin_overview_map.get(mid, False))
        mid = group[0]
        ov = overview_map.get(mid, "")
        ov_preview = (ov[:100] + "...") if len(ov) > 100 else (ov or "<EMPTY>")
        print(f"Text Group #{rank}: size={g_size:,} | thin={thin_count:,} | Text: {ov_preview!r}")

    # --- 4. Query with 831 ties: seeds [1108020, 470285, 156627] ---
    print("\n" + "=" * 90)
    print("QUERY WITH 831 TIES (seeds: [1108020, 470285, 156627])")
    print("=" * 90)
    meta_index.load_index()

    req = RecommendRequest(
        movie_ids=[1108020, 470285, 156627],
        tag_weight=0.5
    )
    res = recommender.retrieve_and_rank(req)
    ranked = res["ranked"]
    print(f"Total ranked results: {len(ranked)} (pool size: {res['pool_size']})")

    # Check ties among results
    scores = [r["final_score"] for r in ranked]
    from collections import Counter
    score_counts = Counter(scores)
    most_common_score, most_common_count = score_counts.most_common(1)[0]
    print(f"Most common final_score in ranked: {most_common_score:.6f} (occurs {most_common_count} times)")

    print("\nTop 10 Results:")
    for i, r in enumerate(ranked[:10], 1):
        mid = r["movie_id"]
        final_s = r["final_score"]
        tag_s = r["tag_score"]
        plot_s = r["plot_score"]
        ov = overview_map.get(mid, "")
        thin = thin_overview_map.get(mid, False)
        ov_preview = (ov[:90] + "...") if len(ov) > 90 else (ov or "<EMPTY>")
        print(f"  #{i:2d} id={mid:<8} final={final_s:.5f} tag={tag_s:.5f} plot={plot_s:.5f} thin={str(thin):<5} | {ov_preview}")

    # Also inspect the 831 ties: what are their scores and overviews?
    tied_mids = [r["movie_id"] for r in ranked if abs(r["final_score"] - most_common_score) < 1e-6]
    print(f"\nOf the {len(tied_mids)} movies tied at score {most_common_score:.6f}:")
    tied_thin_count = sum(1 for m in tied_mids if thin_overview_map.get(m, False))
    print(f"  How many have has_thin_overview = True: {tied_thin_count} / {len(tied_mids)} ({tied_thin_count/len(tied_mids)*100:.1f}%)")
    tied_ovs = Counter(overview_map.get(m, "").strip() for m in tied_mids)
    print(f"  Distinct overview texts among tied movies: {len(tied_ovs)}")
    print(f"  Top overview texts among tied movies:")
    for ov_text, count in tied_ovs.most_common(5):
        preview = (ov_text[:80] + "...") if len(ov_text) > 80 else (ov_text or "<EMPTY>")
        print(f"    [{count}x]: {preview!r}")

    # --- 5. Rule Proposals and Catalog Impact ---
    print("\n" + "=" * 90)
    print("RULE PROPOSAL AND IMPACT ANALYSIS")
    print("=" * 90)
    # Movies in large vector groups (size >= 20)
    large_vec_group_mids = set()
    for g in vec_groups_ge_20:
        large_vec_group_mids.update(g)

    # Movies in large vector groups (size >= 5)
    ge_5_vec_group_mids = set()
    for g in vec_groups_ge_5:
        ge_5_vec_group_mids.update(g)

    # Thin overview movies
    thin_mids = set(m for m, is_thin in thin_overview_map.items() if is_thin)

    # Total catalog count
    total_catalog = len(overview_map)

    rule1_mids = thin_mids
    rule2_mids = large_vec_group_mids
    rule3_mids = thin_mids | large_vec_group_mids
    rule4_mids = thin_mids | ge_5_vec_group_mids

    print(f"Total catalog movies: {total_catalog:,}")
    print(f"Rule 1: has_thin_overview = True -> affects {len(rule1_mids):,} movies ({len(rule1_mids)/total_catalog*100:.2f}%)")
    print(f"Rule 2: vector group size >= 20   -> affects {len(rule2_mids):,} movies ({len(rule2_mids)/total_catalog*100:.2f}%)")
    print(f"Rule 3: has_thin_overview OR vector group >= 20 -> affects {len(rule3_mids):,} movies ({len(rule3_mids)/total_catalog*100:.2f}%)")
    print(f"Rule 4: has_thin_overview OR vector group >= 5  -> affects {len(rule4_mids):,} movies ({len(rule4_mids)/total_catalog*100:.2f}%)")

    # Overlap between thin and large group
    overlap = thin_mids & large_vec_group_mids
    print(f"Overlap between Rule 1 and Rule 2: {len(overlap):,} movies")

if __name__ == "__main__":
    main()
