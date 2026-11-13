"""
Item 5: Report-Only Analysis
(a) Impute missing signal with the pool median of that signal among dual-signal candidates.
    Report single-signal share of top 50 at tag_weight 0.2, 0.5, 0.8 for 100 queries.
(b) Count movies with < 3 tokens in tag vector.
    What happens to top 20 at tag_weight 0.5 and 1.0 if treated as tag_ok=False.
    Show 10 side-by-side examples with titles.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from app.config import settings
from app.db.pool import open_pool
from app.db.queries import get_movies
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import query_builder, candidates, scoring, fusion
from app.services.scoring import Scores

# Boot
open_pool()
artifacts.load_all()
meta_index.load_index()

# 100 reproducible seed queries
tag_ids = set(artifacts.matrix_ids().tolist())
plot_ids_set = set(artifacts.plot_ids().tolist())
common_ids = list(tag_ids & plot_ids_set)
random.seed(42)
sample_seeds = random.sample(common_ids, min(100, len(common_ids)))

def run_query(seed_ids, tw):
    req = RecommendRequest(movie_ids=seed_ids, tags=[], plot_text=None, tag_weight=tw)
    q = query_builder.build(req)
    pool_ids = candidates.generate(req, q)
    s = scoring.score(pool_ids, q)
    return s, q, pool_ids

# ═══════════════════════════════════════════════════════════
# Part 5a: Missing Signal Imputation with Pool Median
# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("Part 5a: Missing signal imputation with pool median (100 queries)")
print("=" * 80)

results_5a = {}

for tw in [0.2, 0.5, 0.8]:
    baseline_single_counts = []
    imputed_single_counts = []
    imputed_tag_only_counts = []
    imputed_plot_only_counts = []
    total_items = 0

    for i, seed in enumerate(sample_seeds):
        try:
            s, q, pool_ids = run_query([seed], tw)
        except Exception as e:
            continue

        dual = s.tag_ok & s.plot_ok

        # Baseline fusion (no imputation)
        ranked_base = fusion.fuse(s, tw, top_n=50)
        base_top50_mids = [r["movie_id"] for r in ranked_base]
        base_single = sum(1 for mid in base_top50_mids if not (s.tag_ok[s.index(mid)] and s.plot_ok[s.index(mid)]))
        baseline_single_counts.append(base_single)

        # Median imputation among dual-signal candidates in pool
        if np.any(dual):
            med_tag = float(np.median(s.tag[dual]))
            med_plot = float(np.median(s.plot[dual]))
        else:
            med_tag = 0.0
            med_plot = 0.0

        # Impute
        imp_tag = np.where(s.tag_ok, s.tag, med_tag)
        imp_plot = np.where(s.plot_ok, s.plot, med_plot)
        imp_tag_scaled = np.clip(imp_tag, 0.0, 1.0)
        imp_plot_scaled = np.clip(imp_plot, 0.0, 1.0)

        final_imputed = tw * imp_tag_scaled + (1.0 - tw) * imp_plot_scaled
        final_imputed[~(s.tag_ok | s.plot_ok)] = -1.0

        ids = s.ids
        order = np.lexsort((ids, -imp_plot_scaled, -final_imputed))[:50]
        imp_top50_ids = [int(ids[idx]) for idx in order if final_imputed[idx] >= 0]

        # Count single-signal in imputed top 50
        imp_single = 0
        imp_tag_only = 0
        imp_plot_only = 0
        for mid in imp_top50_ids:
            idx = s.index(mid)
            is_tag_ok = bool(s.tag_ok[idx])
            is_plot_ok = bool(s.plot_ok[idx])
            if not (is_tag_ok and is_plot_ok):
                imp_single += 1
                if is_tag_ok and not is_plot_ok:
                    imp_tag_only += 1
                elif is_plot_ok and not is_tag_ok:
                    imp_plot_only += 1

        imputed_single_counts.append(imp_single)
        imputed_tag_only_counts.append(imp_tag_only)
        imputed_plot_only_counts.append(imp_plot_only)
        total_items += len(imp_top50_ids)

        if (i + 1) % 25 == 0:
            print(f"  tw={tw}: processed {i+1}/100 queries...")

    n_q = len(imputed_single_counts)
    results_5a[str(tw)] = {
        "queries_evaluated": n_q,
        "baseline_single_in_top50_avg": round(float(np.mean(baseline_single_counts)), 2),
        "baseline_single_pct": round(float(np.mean(baseline_single_counts) / 50 * 100), 2),
        "imputed_single_in_top50_avg": round(float(np.mean(imputed_single_counts)), 2),
        "imputed_single_pct": round(float(np.mean(imputed_single_counts) / 50 * 100), 2),
        "imputed_tag_only_avg": round(float(np.mean(imputed_tag_only_counts)), 2),
        "imputed_plot_only_avg": round(float(np.mean(imputed_plot_only_counts)), 2),
    }
    print(f"  tw={tw} Summary:")
    print(f"    Baseline without imputation : {results_5a[str(tw)]['baseline_single_in_top50_avg']}/50 ({results_5a[str(tw)]['baseline_single_pct']}%)")
    print(f"    With median imputation     : {results_5a[str(tw)]['imputed_single_in_top50_avg']}/50 ({results_5a[str(tw)]['imputed_single_pct']}%)")
    print(f"      (tag-only: {results_5a[str(tw)]['imputed_tag_only_avg']}, plot-only: {results_5a[str(tw)]['imputed_plot_only_avg']})")

# ═══════════════════════════════════════════════════════════
# Part 5b: Thin Tag Vectors (< 3 tokens)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("Part 5b: Tag vectors with < 3 tokens")
print("=" * 80)

m = artifacts.tag_matrix()
nnz_per_row = np.diff(m.indptr)
thin_mask = nnz_per_row < 3
n_thin = int(thin_mask.sum())
n_total = m.shape[0]
print(f"Movies with < 3 tokens in tag vector: {n_thin} / {n_total} ({n_thin/n_total*100:.2f}%)")

# Pre-gather all thin movie IDs
thin_movie_ids = set(artifacts.matrix_ids()[thin_mask].tolist())

results_5b = {
    "total_tag_catalog": n_total,
    "thin_count": n_thin,
    "thin_pct": round(n_thin / n_total * 100, 2),
}

for tw in [0.5, 1.0]:
    print(f"\nEvaluating tw={tw}...")
    queries_changed = 0
    all_examples = []

    for i, seed in enumerate(sample_seeds):
        try:
            s, q, pool_ids = run_query([seed], tw)
        except Exception as e:
            continue

        # Baseline ranking
        ranked_orig = fusion.fuse(s, tw, top_n=20)
        orig_ids = [r["movie_id"] for r in ranked_orig]

        # Suppressed tag ranking: treat thin-tag movies as tag_ok=False
        s_supp_tag_ok = s.tag_ok.copy()
        for idx_row, mid in enumerate(s.ids):
            if int(mid) in thin_movie_ids:
                s_supp_tag_ok[idx_row] = False

        s_supp = Scores(ids=s.ids, tag=s.tag, plot=s.plot, tag_ok=s_supp_tag_ok, plot_ok=s.plot_ok)
        ranked_supp = fusion.fuse(s_supp, tw, top_n=20)
        supp_ids = [r["movie_id"] for r in ranked_supp]

        if orig_ids != supp_ids:
            queries_changed += 1
            if len(all_examples) < 10:
                all_examples.append({
                    "seed": seed,
                    "tw": tw,
                    "orig_ranked": ranked_orig[:10],
                    "supp_ranked": ranked_supp[:10],
                    "orig_ids": orig_ids[:10],
                    "supp_ids": supp_ids[:10],
                })

        if (i + 1) % 25 == 0:
            print(f"  tw={tw}: processed {i+1}/100, changed={queries_changed}...")

    # Fetch movie titles for the examples
    needed_mids = set()
    for ex in all_examples:
        needed_mids.add(ex["seed"])
        for r in ex["orig_ranked"]:
            needed_mids.add(r["movie_id"])
        for r in ex["supp_ranked"]:
            needed_mids.add(r["movie_id"])

    movie_meta = get_movies(list(needed_mids))

    examples_with_titles = []
    for ex in all_examples:
        seed_title = movie_meta.get(ex["seed"], {}).get("title", f"ID {ex['seed']}")
        orig_list = []
        for rank, r in enumerate(ex["orig_ranked"], 1):
            mid = r["movie_id"]
            meta = movie_meta.get(mid, {})
            orig_list.append({
                "rank": rank,
                "id": mid,
                "title": meta.get("title", "Unknown"),
                "is_thin": mid in thin_movie_ids,
                "final_score": round(r["final_score"], 4),
                "tag_score": round(r["tag_score"], 4),
                "plot_score": round(r["plot_score"], 4),
            })
        supp_list = []
        for rank, r in enumerate(ex["supp_ranked"], 1):
            mid = r["movie_id"]
            meta = movie_meta.get(mid, {})
            supp_list.append({
                "rank": rank,
                "id": mid,
                "title": meta.get("title", "Unknown"),
                "is_thin": mid in thin_movie_ids,
                "final_score": round(r["final_score"], 4),
                "tag_score": round(r["tag_score"], 4),
                "plot_score": round(r["plot_score"], 4),
            })
        examples_with_titles.append({
            "seed_id": ex["seed"],
            "seed_title": seed_title,
            "orig_top10": orig_list,
            "supp_top10": supp_list,
        })

    results_5b[f"tw_{tw}"] = {
        "queries_changed": queries_changed,
        "queries_total": len(sample_seeds),
        "pct_changed": round(queries_changed / len(sample_seeds) * 100, 2),
        "examples": examples_with_titles,
    }
    print(f"tw={tw}: {queries_changed} / {len(sample_seeds)} ({results_5b[f'tw_{tw}']['pct_changed']}%) top-20 lists changed")

# Write to JSON
final_output = {
    "item_5a_median_imputation": results_5a,
    "item_5b_thin_tag_vectors": results_5b,
}
with open("bench/item5_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)

print("\nAll Item 5 analysis results successfully written to bench/item5_analysis_results.json!")
