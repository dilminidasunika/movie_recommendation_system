"""
Item 4 Report-Only Analysis Script:
(a) Imputation with pool median, p25, p10 priors, and shrink factor.
    Measures single-signal share of top 50 at tag_weight in [0.2, 0.5, 0.8] vs catalog shares (tag-only 12.7%, plot-only 9.2%).
    Never displays imputed value.
(b) Thin tag vectors (<3 and <4 tokens):
    Count catalog movies, how many have no plot (unrankable under binary rule).
    Measure top-20 changes at 0.5 and 1.0 with soft penalty min(1, nnz/3) and min(1, nnz/4) vs binary rule vs baseline.
"""
import sys
import os
import argparse
import json
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion
from app.stores import artifacts, meta_index


def run_item4_analysis(n_queries: int = 100):
    open_pool()
    artifacts.load_all()
    meta_index.load_index()

    tag_m = artifacts.tag_matrix()
    tag_ids = artifacts.matrix_ids()
    plot_ids = artifacts.plot_ids()

    tag_id_set = set(tag_ids.tolist())
    plot_id_set = set(plot_ids.tolist())
    all_catalog = tag_id_set | plot_id_set
    n_cat = len(all_catalog)

    both_cat = tag_id_set & plot_id_set
    tag_only_cat = tag_id_set - plot_id_set
    plot_only_cat = plot_id_set - tag_id_set

    pct_both = len(both_cat) / n_cat * 100
    pct_tag_only = len(tag_only_cat) / n_cat * 100
    pct_plot_only = len(plot_only_cat) / n_cat * 100
    pct_single = (len(tag_only_cat) + len(plot_only_cat)) / n_cat * 100

    print("=" * 80)
    print("CATALOG SIGNAL DISTRIBUTION")
    print("=" * 80)
    print(f"Total catalog movies: {n_cat:,}")
    print(f"  Dual-signal (both tag + plot): {len(both_cat):,} ({pct_both:.2f}%)")
    print(f"  Tag-only movies:               {len(tag_only_cat):,} ({pct_tag_only:.2f}%)")
    print(f"  Plot-only movies:              {len(plot_only_cat):,} ({pct_plot_only:.2f}%)")
    print(f"  Total Single-Signal:           {len(tag_only_cat)+len(plot_only_cat):,} ({pct_single:.2f}%)")

    # ═══════════════════════════════════════════════════════════════
    # Part 4(b) Catalog Analysis: Thin Tag Vectors
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 4(b): THIN TAG VECTORS CATALOG ANALYSIS")
    print("=" * 80)

    # nnz per tag matrix row
    nnz_per_row = np.diff(tag_m.indptr)
    thin_lt3_mask = nnz_per_row < 3
    thin_lt4_mask = nnz_per_row < 4

    count_lt3 = int(np.sum(thin_lt3_mask))
    count_lt4 = int(np.sum(thin_lt4_mask))

    # Identify which of these thin-tag movies have NO plot embedding
    thin_lt3_ids = set(tag_ids[thin_lt3_mask].tolist())
    thin_lt4_ids = set(tag_ids[thin_lt4_mask].tolist())

    thin_lt3_no_plot = thin_lt3_ids - plot_id_set
    thin_lt4_no_plot = thin_lt4_ids - plot_id_set

    print(f"Movies with tag nnz < 3: {count_lt3:,} ({count_lt3 / len(tag_ids) * 100:.2f}% of tag catalog)")
    print(f"  -> Of which have NO plot embedding (unrankable under binary rule): {len(thin_lt3_no_plot):,} ({len(thin_lt3_no_plot) / len(tag_ids) * 100:.2f}%)")
    print(f"Movies with tag nnz < 4: {count_lt4:,} ({count_lt4 / len(tag_ids) * 100:.2f}% of tag catalog)")
    print(f"  -> Of which have NO plot embedding (unrankable under binary rule): {len(thin_lt4_no_plot):,} ({len(thin_lt4_no_plot) / len(tag_ids) * 100:.2f}%)")

    # Sample queries
    rng = np.random.default_rng(20261113)
    common_arr = np.array(list(both_cat), dtype=np.int64)
    query_seeds = [rng.choice(common_arr, size=3, replace=False).tolist() for _ in range(n_queries)]

    # ═══════════════════════════════════════════════════════════════
    # Part 4(a): Imputation Analysis (100 queries x 3 weights)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"PART 4(a): IMPUTATION STRATEGY COMPARISON ({n_queries} queries)")
    print("=" * 80)

    # Imputation strategies to evaluate:
    # 1. baseline (current 0-fill)
    # 2. median
    # 3. p25 prior (pure p25)
    # 4. p10 prior (pure p10)
    # 5. shrunk_median_p25 (0.7 * med + 0.3 * p25)
    # 6. shrunk_median_0.8 (0.8 * med)
    strategies = [
        "baseline",
        "median",
        "p25_prior",
        "p10_prior",
        "shrunk_p25_0.7",
        "shrink_0.8",
    ]

    weights = [0.2, 0.5, 0.8]
    imp_results = {tw: {strat: {"single": [], "tag_only": [], "plot_only": []} for strat in strategies} for tw in weights}

    # Helper for fusing with imputed scores
    def fuse_with_imputed(s, tw, imp_tag, imp_plot, top_n=50):
        # Final score computed using imputed scores
        final = tw * imp_tag + (1.0 - tw) * imp_plot
        # Drop if neither signal originally existed
        final[~(s.tag_ok | s.plot_ok)] = -1.0
        # Filter final > 0 (Item 2)
        valid = np.flatnonzero(final > 0)
        if len(valid) == 0:
            return []
        sub_ids = s.ids[valid]
        sub_final = final[valid]
        sub_pn = imp_plot[valid]
        order = np.lexsort((sub_ids, -sub_pn, -sub_final))[:top_n]
        chosen = valid[order]

        # NOTE: NEVER display the imputed value in output dict
        # Return TRUE un-imputed scores
        tn_true = np.where(s.tag_ok, np.clip(s.tag, 0.0, 1.0), 0.0)
        pn_true = np.where(s.plot_ok, np.clip(s.plot, 0.0, 1.0), 0.0)

        tag_only = s.tag_ok & (~s.plot_ok)
        plot_only = s.plot_ok & (~s.tag_ok)

        out = []
        for i in chosen:
            out.append({
                "movie_id": int(s.ids[i]),
                "final_score": round(float(final[i]), 6),
                "tag_score": round(float(tn_true[i]), 6),    # TRUE original score displayed
                "plot_score": round(float(pn_true[i]), 6),  # TRUE original score displayed
                "is_single_signal": bool(tag_only[i] or plot_only[i]),
                "is_tag_only": bool(tag_only[i]),
                "is_plot_only": bool(plot_only[i]),
            })
        return out

    # Run queries for Part 4(a)
    for q_idx, seeds in enumerate(query_seeds):
        req_base = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        q = query_builder.build(req_base)
        pool = candidates.generate(req_base, q)
        s = scoring.score(pool, q)

        dual = s.tag_ok & s.plot_ok
        has_dual = np.any(dual)
        tn = np.where(s.tag_ok, np.clip(s.tag, 0.0, 1.0), 0.0)
        pn = np.where(s.plot_ok, np.clip(s.plot, 0.0, 1.0), 0.0)

        if has_dual:
            med_t = float(np.median(tn[dual]))
            med_p = float(np.median(pn[dual]))
            p25_t = float(np.percentile(tn[dual], 25))
            p25_p = float(np.percentile(pn[dual], 25))
            p10_t = float(np.percentile(tn[dual], 10))
            p10_p = float(np.percentile(pn[dual], 10))
        else:
            med_t = med_p = p25_t = p25_p = p10_t = p10_p = 0.0

        for tw in weights:
            # 1. Baseline
            res_base = fusion.fuse(s, tw, top_n=50)
            n_base = max(1, len(res_base))
            # Determine single signals in baseline
            single_base = sum(1 for r in res_base if not (s.tag_ok[s.index(r["movie_id"])] and s.plot_ok[s.index(r["movie_id"])]))
            t_only_base = sum(1 for r in res_base if s.tag_ok[s.index(r["movie_id"])] and not s.plot_ok[s.index(r["movie_id"])])
            p_only_base = sum(1 for r in res_base if s.plot_ok[s.index(r["movie_id"])] and not s.tag_ok[s.index(r["movie_id"])])
            imp_results[tw]["baseline"]["single"].append(single_base / n_base * 100)
            imp_results[tw]["baseline"]["tag_only"].append(t_only_base / n_base * 100)
            imp_results[tw]["baseline"]["plot_only"].append(p_only_base / n_base * 100)

            # 2. Median
            imp_t_med = np.where(s.tag_ok, tn, med_t)
            imp_p_med = np.where(s.plot_ok, pn, med_p)
            res_med = fuse_with_imputed(s, tw, imp_t_med, imp_p_med, 50)
            n_med = max(1, len(res_med))
            imp_results[tw]["median"]["single"].append(sum(1 for r in res_med if r["is_single_signal"]) / n_med * 100)
            imp_results[tw]["median"]["tag_only"].append(sum(1 for r in res_med if r["is_tag_only"]) / n_med * 100)
            imp_results[tw]["median"]["plot_only"].append(sum(1 for r in res_med if r["is_plot_only"]) / n_med * 100)

            # 3. p25 Prior
            imp_t_p25 = np.where(s.tag_ok, tn, p25_t)
            imp_p_p25 = np.where(s.plot_ok, pn, p25_p)
            res_p25 = fuse_with_imputed(s, tw, imp_t_p25, imp_p_p25, 50)
            n_p25 = max(1, len(res_p25))
            imp_results[tw]["p25_prior"]["single"].append(sum(1 for r in res_p25 if r["is_single_signal"]) / n_p25 * 100)
            imp_results[tw]["p25_prior"]["tag_only"].append(sum(1 for r in res_p25 if r["is_tag_only"]) / n_p25 * 100)
            imp_results[tw]["p25_prior"]["plot_only"].append(sum(1 for r in res_p25 if r["is_plot_only"]) / n_p25 * 100)

            # 4. p10 Prior
            imp_t_p10 = np.where(s.tag_ok, tn, p10_t)
            imp_p_p10 = np.where(s.plot_ok, pn, p10_p)
            res_p10 = fuse_with_imputed(s, tw, imp_t_p10, imp_p_p10, 50)
            n_p10 = max(1, len(res_p10))
            imp_results[tw]["p10_prior"]["single"].append(sum(1 for r in res_p10 if r["is_single_signal"]) / n_p10 * 100)
            imp_results[tw]["p10_prior"]["tag_only"].append(sum(1 for r in res_p10 if r["is_tag_only"]) / n_p10 * 100)
            imp_results[tw]["p10_prior"]["plot_only"].append(sum(1 for r in res_p10 if r["is_plot_only"]) / n_p10 * 100)

            # 5. Shrunk towards p25 (0.7 * med + 0.3 * p25)
            shrunk_t_p25 = 0.7 * med_t + 0.3 * p25_t
            shrunk_p_p25 = 0.7 * med_p + 0.3 * p25_p
            imp_t_shrp = np.where(s.tag_ok, tn, shrunk_t_p25)
            imp_p_shrp = np.where(s.plot_ok, pn, shrunk_p_p25)
            res_shrp = fuse_with_imputed(s, tw, imp_t_shrp, imp_p_shrp, 50)
            n_shrp = max(1, len(res_shrp))
            imp_results[tw]["shrunk_p25_0.7"]["single"].append(sum(1 for r in res_shrp if r["is_single_signal"]) / n_shrp * 100)
            imp_results[tw]["shrunk_p25_0.7"]["tag_only"].append(sum(1 for r in res_shrp if r["is_tag_only"]) / n_shrp * 100)
            imp_results[tw]["shrunk_p25_0.7"]["plot_only"].append(sum(1 for r in res_shrp if r["is_plot_only"]) / n_shrp * 100)

            # 6. Shrink factor 0.8 on median
            imp_t_s08 = np.where(s.tag_ok, tn, 0.8 * med_t)
            imp_p_s08 = np.where(s.plot_ok, pn, 0.8 * med_p)
            res_s08 = fuse_with_imputed(s, tw, imp_t_s08, imp_p_s08, 50)
            n_s08 = max(1, len(res_s08))
            imp_results[tw]["shrink_0.8"]["single"].append(sum(1 for r in res_s08 if r["is_single_signal"]) / n_s08 * 100)
            imp_results[tw]["shrink_0.8"]["tag_only"].append(sum(1 for r in res_s08 if r["is_tag_only"]) / n_s08 * 100)
            imp_results[tw]["shrink_0.8"]["plot_only"].append(sum(1 for r in res_s08 if r["is_plot_only"]) / n_s08 * 100)

    # Print Part 4(a) Summary Table
    print(f"\nSingle-Signal Share of Top 50 across {n_queries} queries (Catalog Baseline: Single={pct_single:.2f}%, Tag-only={pct_tag_only:.2f}%, Plot-only={pct_plot_only:.2f}%)")
    print(f"{'tag_weight':<10} | {'Strategy':<18} | {'Single-Signal Share':<20} | {'Tag-Only Share':<16} | {'Plot-Only Share':<16}")
    print("-" * 90)
    for tw in weights:
        for strat in strategies:
            s_mean = np.mean(imp_results[tw][strat]["single"])
            t_mean = np.mean(imp_results[tw][strat]["tag_only"])
            p_mean = np.mean(imp_results[tw][strat]["plot_only"])
            print(f"{tw:<10.1f} | {strat:<18} | {s_mean:<20.2f}% | {t_mean:<16.2f}% | {p_mean:<16.2f}%")
        print("-" * 90)

    # ═══════════════════════════════════════════════════════════════
    # Part 4(b): Soft Penalty vs Binary Rule Evaluation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 4(b): SOFT PENALTY FOR THIN TAG VECTORS (TOP-20 CHANGES)")
    print("=" * 80)

    # Map candidate IDs to their nnz in tag_m
    tag_id_to_r = artifacts.id_to_row()

    # We evaluate for tw in [0.5, 1.0]:
    # 1. Baseline
    # 2. Binary rule (tag_ok = False for nnz < 3)
    # 3. Soft penalty nnz/3 (tag score *= min(1.0, nnz/3))
    # 4. Soft penalty nnz/4 (tag score *= min(1.0, nnz/4))
    b_weights = [0.5, 1.0]
    b_changes = {tw: {"binary_lt3": 0, "soft_div3": 0, "soft_div4": 0} for tw in b_weights}
    b_side_by_side = []

    for q_idx, seeds in enumerate(query_seeds):
        req_base = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        q = query_builder.build(req_base)
        pool = candidates.generate(req_base, q)
        s = scoring.score(pool, q)

        # Candidate nnz
        c_rows = tag_id_to_r.rows_for(s.ids)
        c_nnz = np.zeros(len(s.ids), dtype=np.int32)
        valid_tag_cand = c_rows >= 0
        c_nnz[valid_tag_cand] = nnz_per_row[c_rows[valid_tag_cand]]

        for tw in b_weights:
            # Baseline
            res_base = fusion.fuse(s, tw, top_n=20)
            base_top20_ids = [r["movie_id"] for r in res_base]

            # Binary rule: tag_ok False if nnz < 3
            s_bin = scoring.Scores(
                ids=s.ids.copy(),
                tag=s.tag.copy(),
                tag_ok=s.tag_ok.copy() & (c_nnz >= 3),
                plot=s.plot.copy(),
                plot_ok=s.plot_ok.copy(),
            )
            res_bin = fusion.fuse(s_bin, tw, top_n=20)
            bin_top20_ids = [r["movie_id"] for r in res_bin]

            # Soft penalty nnz / 3.0
            mult3 = np.clip(c_nnz / 3.0, 0.0, 1.0).astype(np.float32)
            s_soft3 = scoring.Scores(
                ids=s.ids.copy(),
                tag=s.tag.copy() * mult3,
                tag_ok=s.tag_ok.copy(),
                plot=s.plot.copy(),
                plot_ok=s.plot_ok.copy(),
            )
            res_soft3 = fusion.fuse(s_soft3, tw, top_n=20)
            soft3_top20_ids = [r["movie_id"] for r in res_soft3]

            # Soft penalty nnz / 4.0
            mult4 = np.clip(c_nnz / 4.0, 0.0, 1.0).astype(np.float32)
            s_soft4 = scoring.Scores(
                ids=s.ids.copy(),
                tag=s.tag.copy() * mult4,
                tag_ok=s.tag_ok.copy(),
                plot=s.plot.copy(),
                plot_ok=s.plot_ok.copy(),
            )
            res_soft4 = fusion.fuse(s_soft4, tw, top_n=20)
            soft4_top20_ids = [r["movie_id"] for r in res_soft4]

            if base_top20_ids != bin_top20_ids:
                b_changes[tw]["binary_lt3"] += 1
            if base_top20_ids != soft3_top20_ids:
                b_changes[tw]["soft_div3"] += 1
            if base_top20_ids != soft4_top20_ids:
                b_changes[tw]["soft_div4"] += 1

            if len(b_side_by_side) < 5 and tw == 0.5 and (base_top20_ids != soft3_top20_ids):
                b_side_by_side.append({
                    "seeds": seeds,
                    "base": res_base[:10],
                    "bin": res_bin[:10],
                    "soft3": res_soft3[:10],
                    "soft4": res_soft4[:10],
                })

    print(f"\nTop-20 List Change Rates (out of {n_queries} queries):")
    print(f"{'tag_weight':<10} | {'Binary (nnz<3 tag_ok=False)':<30} | {'Soft min(1, nnz/3)':<25} | {'Soft min(1, nnz/4)':<25}")
    print("-" * 95)
    for tw in b_weights:
        b_ch = b_changes[tw]["binary_lt3"]
        s3_ch = b_changes[tw]["soft_div3"]
        s4_ch = b_changes[tw]["soft_div4"]
        print(f"{tw:<10.1f} | {b_ch:>3}/{n_queries} ({b_ch/n_queries*100:.1f}%)"
              f"{'':<14} | {s3_ch:>3}/{n_queries} ({s3_ch/n_queries*100:.1f}%)"
              f"{'':<10} | {s4_ch:>3}/{n_queries} ({s4_ch/n_queries*100:.1f}%)")

    close_pool()

    return {
        "catalog": {
            "total": n_cat,
            "pct_tag_only": round(pct_tag_only, 2),
            "pct_plot_only": round(pct_plot_only, 2),
            "pct_single": round(pct_single, 2),
            "thin_lt3_total": count_lt3,
            "thin_lt3_no_plot": len(thin_lt3_no_plot),
            "thin_lt4_total": count_lt4,
            "thin_lt4_no_plot": len(thin_lt4_no_plot),
        },
        "imputation": imp_results,
        "thin_tag_changes": b_changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    q_count = 5 if args.smoke else args.queries
    run_item4_analysis(q_count)
