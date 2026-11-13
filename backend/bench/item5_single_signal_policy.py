"""
Item 5: Single-Signal Policy Evaluation (Report Only)
For 100 queries at tag_weight 0.2, 0.5, 0.8:
- Evaluates score for movies with only one signal = sum(w_i * s_i) / sum(w_i)
- Edge cases: at tag_weight 0 tag-only, and tag_weight 1 plot-only, are excluded
- Measures share of single-signal movies in top 50 vs their catalog share (~21.9%)
- Displays 10 queries' top 20 side by side
- Does NOT apply any change to production code
"""
import sys, os, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "dummy")

from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion

def fuse_current(s, tag_weight: float, top_n: int = 50):
    tn = np.where(s.tag_ok, np.clip(s.tag, 0.0, 1.0), 0.0)
    pn = np.where(s.plot_ok, np.clip(s.plot, 0.0, 1.0), 0.0)
    final = tag_weight * tn + (1.0 - tag_weight) * pn
    final[~(s.tag_ok | s.plot_ok)] = -1.0
    order = np.lexsort((s.ids, -final))[:top_n]

    tag_only = s.tag_ok & (~s.plot_ok)
    plot_only = s.plot_ok & (~s.tag_ok)
    both = s.tag_ok & s.plot_ok

    out = []
    for i in order:
        if final[i] >= 0:
            stype = "both" if both[i] else ("tag_only" if tag_only[i] else "plot_only")
            out.append({
                "movie_id": int(s.ids[i]),
                "final_score": float(final[i]),
                "tag_score": float(tn[i]),
                "plot_score": float(pn[i]),
                "is_single_signal": bool(tag_only[i] or plot_only[i]),
                "signal_type": stype,
            })
    return out

def fuse_single_signal(s, tag_weight: float, top_n: int = 50):
    w_tag = tag_weight
    w_plot = 1.0 - tag_weight
    tn = np.where(s.tag_ok, np.clip(s.tag, 0.0, 1.0), 0.0)
    pn = np.where(s.plot_ok, np.clip(s.plot, 0.0, 1.0), 0.0)

    n = len(s.ids)
    final = np.full(n, -1.0, dtype=np.float32)

    both = s.tag_ok & s.plot_ok
    tag_only = s.tag_ok & (~s.plot_ok)
    plot_only = s.plot_ok & (~s.tag_ok)

    # Both signals: weighted sum divided by sum(w) = 1.0
    final[both] = w_tag * tn[both] + w_plot * pn[both]

    # Tag only: divided by w_tag -> pure tn
    if w_tag > 0.0:
        final[tag_only] = tn[tag_only]
    else:
        final[tag_only] = -1.0  # excluded at tag_weight 0

    # Plot only: divided by w_plot -> pure pn
    if w_plot > 0.0:
        final[plot_only] = pn[plot_only]
    else:
        final[plot_only] = -1.0  # excluded at tag_weight 1

    order = np.lexsort((s.ids, -final))[:top_n]

    out = []
    for i in order:
        if final[i] >= 0:
            stype = "both" if both[i] else ("tag_only" if tag_only[i] else "plot_only")
            out.append({
                "movie_id": int(s.ids[i]),
                "final_score": float(final[i]),
                "tag_score": float(tn[i]),
                "plot_score": float(pn[i]),
                "is_single_signal": bool(tag_only[i] or plot_only[i]),
                "signal_type": stype,
            })
    return out

def main():
    print("Opening DB pool and loading artifacts...", flush=True)
    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    idx = meta_index.get_index()

    # 1. Catalog breakdown
    tag_mids = set(artifacts.matrix_ids().tolist())
    plot_mids = set(artifacts.plot_ids().tolist())
    all_catalog_mids = tag_mids | plot_mids
    n_catalog = len(all_catalog_mids)

    both_mids = tag_mids & plot_mids
    tag_only_mids = tag_mids - plot_mids
    plot_only_mids = plot_mids - tag_mids
    single_signal_mids = tag_only_mids | plot_only_mids

    print("\n" + "=" * 80)
    print("CATALOG SIGNAL DISTRIBUTION")
    print("=" * 80)
    print(f"Total catalog movies with at least 1 signal: {n_catalog:,}")
    print(f"  Both signals (tag + plot):  {len(both_mids):,} ({len(both_mids)/n_catalog*100:.2f}%)")
    print(f"  Tag-only movies:            {len(tag_only_mids):,} ({len(tag_only_mids)/n_catalog*100:.2f}%)")
    print(f"  Plot-only movies:           {len(plot_only_mids):,} ({len(plot_only_mids)/n_catalog*100:.2f}%)")
    print(f"  Total Single-Signal movies: {len(single_signal_mids):,} ({len(single_signal_mids)/n_catalog*100:.2f}%)")

    # 2. Generate 100 random realistic queries
    rng = np.random.default_rng(20261112)
    # Pick seeds from movies that have both signals for rich queries
    both_arr = np.array(list(both_mids), dtype=np.int64)
    queries_seeds = [rng.choice(both_arr, size=3, replace=False).tolist() for _ in range(100)]

    print(f"\nPrepared 100 queries. Evaluating across tag_weight in [0.2, 0.5, 0.8]...", flush=True)

    # Store results per weight
    weights = [0.2, 0.5, 0.8]
    stats = {}
    side_by_side = {}

    for tw in weights:
        current_single_shares = []
        proposed_single_shares = []
        current_tag_only_shares = []
        proposed_tag_only_shares = []
        current_plot_only_shares = []
        proposed_plot_only_shares = []

        examples = []

        for q_i, seeds in enumerate(queries_seeds):
            req = RecommendRequest(movie_ids=seeds, tag_weight=tw)
            q = query_builder.build(req)
            pool = candidates.generate(req, q)
            s = scoring.score(pool, q)

            ranked_curr = fuse_current(s, tw, top_n=50)
            ranked_prop = fuse_single_signal(s, tw, top_n=50)

            # Measure top 50 shares
            c_single = sum(1 for r in ranked_curr if r["is_single_signal"]) / max(1, len(ranked_curr))
            p_single = sum(1 for r in ranked_prop if r["is_single_signal"]) / max(1, len(ranked_prop))
            c_tag = sum(1 for r in ranked_curr if r["signal_type"] == "tag_only") / max(1, len(ranked_curr))
            p_tag = sum(1 for r in ranked_prop if r["signal_type"] == "tag_only") / max(1, len(ranked_prop))
            c_plot = sum(1 for r in ranked_curr if r["signal_type"] == "plot_only") / max(1, len(ranked_curr))
            p_plot = sum(1 for r in ranked_prop if r["signal_type"] == "plot_only") / max(1, len(ranked_prop))

            current_single_shares.append(c_single)
            proposed_single_shares.append(p_single)
            current_tag_only_shares.append(c_tag)
            proposed_tag_only_shares.append(p_tag)
            current_plot_only_shares.append(c_plot)
            proposed_plot_only_shares.append(p_plot)

            if len(examples) < 10 and tw == 0.5:
                # Capture for side-by-side display
                examples.append({
                    "query_idx": q_i + 1,
                    "seeds": seeds,
                    "curr_top20": ranked_curr[:20],
                    "prop_top20": ranked_prop[:20],
                })

        stats[tw] = {
            "current_single_mean": np.mean(current_single_shares) * 100,
            "proposed_single_mean": np.mean(proposed_single_shares) * 100,
            "current_tag_mean": np.mean(current_tag_only_shares) * 100,
            "proposed_tag_mean": np.mean(proposed_tag_only_shares) * 100,
            "current_plot_mean": np.mean(current_plot_only_shares) * 100,
            "proposed_plot_mean": np.mean(proposed_plot_only_shares) * 100,
        }
        if tw == 0.5:
            side_by_side = examples

    # 3. Print Results Summary
    print("\n" + "=" * 90)
    print("SINGLE-SIGNAL SHARE IN TOP 50 (vs 21.88% Catalog Share)")
    print("=" * 90)
    header = f"{'tag_weight':<12} | {'Current Policy':<24} | {'Proposed Policy':<24} | {'Catalog Baseline'}"
    print(header)
    print("-" * len(header))
    for tw in weights:
        st = stats[tw]
        curr_str = f"{st['current_single_mean']:5.2f}% (T:{st['current_tag_mean']:4.1f}% P:{st['current_plot_mean']:4.1f}%)"
        prop_str = f"{st['proposed_single_mean']:5.2f}% (T:{st['proposed_tag_mean']:4.1f}% P:{st['proposed_plot_mean']:4.1f}%)"
        print(f"tw = {tw:<8} | {curr_str:<24} | {prop_str:<24} | 21.88%")

    # 4. Display 10 queries' top 20 side by side (at tag_weight 0.5)
    print("\n" + "=" * 90)
    print("10 QUERIES TOP-20 SIDE-BY-SIDE COMPARISON (tag_weight = 0.5)")
    print("=" * 90)

    for ex in side_by_side:
        q_idx = ex["query_idx"]
        seeds = ex["seeds"]
        print(f"\nQuery #{q_idx} (seeds={seeds}):")
        print(f"  {'Rank':<4} | {'CURRENT POLICY (Half-Penalty)':<38} | {'PROPOSED POLICY (Normalized sum(w*s)/sum(w))':<38}")
        print(f"  {'-'*4} | {'-'*38} | {'-'*38}")
        curr20 = ex["curr_top20"]
        prop20 = ex["prop_top20"]
        for r_i in range(20):
            c_item = curr20[r_i] if r_i < len(curr20) else None
            p_item = prop20[r_i] if r_i < len(prop20) else None

            c_str = f"id={c_item['movie_id']:<7} f={c_item['final_score']:.4f} [{c_item['signal_type']}]" if c_item else "-"
            p_str = f"id={p_item['movie_id']:<7} f={p_item['final_score']:.4f} [{p_item['signal_type']}]" if p_item else "-"
            print(f"  #{r_i+1:<3} | {c_str:<38} | {p_str:<38}")

if __name__ == "__main__":
    main()
