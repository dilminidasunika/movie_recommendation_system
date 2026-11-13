"""
Test and analysis script for boilerplate plot suppression.
Tests that seeds 1108020, 470285, 156627 top 10 are no longer
dominated by "Mexican feature film" movies when suppression is on.
Also runs 100-query analysis comparing with/without suppression.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch
from app.config import settings
from app.db.pool import open_pool
from app.stores import artifacts, meta_index
from app.schemas import RecommendRequest
from app.services import recommender

# Boot
open_pool()
artifacts.load_all()
meta_index.load_index()

PROBLEM_SEEDS = [1108020, 470285, 156627]

def run_rec(seeds, suppress):
    req = RecommendRequest(movie_ids=seeds, tags=[], plot_text=None, tag_weight=0.5)
    with patch.object(settings, "suppress_boilerplate_plots", suppress):
        return recommender.retrieve_and_rank(req)

# -- Part 1: Check the 3 problem seeds --
print("=" * 80)
print("Part 1: Problem seeds - top 10 with/without suppression")
print("=" * 80)

for seed in PROBLEM_SEEDS:
    print(f"\n--- Seed: {seed} ---")
    
    # Without suppression
    res_off = run_rec([seed], suppress=False)
    top10_off = res_off["ranked"][:10]
    
    # With suppression
    res_on = run_rec([seed], suppress=True)
    top10_on = res_on["ranked"][:10]
    
    print(f"\n  WITHOUT suppression (top 10):")
    for i, r in enumerate(top10_off):
        print(f"    {i+1}. [{r['movie_id']}] final={r['final_score']:.4f} tag={r['tag_score']:.4f} plot={r['plot_score']:.4f}")
    
    print(f"\n  WITH suppression (top 10):")
    for i, r in enumerate(top10_on):
        print(f"    {i+1}. [{r['movie_id']}] final={r['final_score']:.4f} tag={r['tag_score']:.4f} plot={r['plot_score']:.4f}")


# -- Part 2: 100-query analysis --
print("\n" + "=" * 80)
print("Part 2: 100-query analysis")
print("=" * 80)

import random, numpy as np

# Get 100 random seed movie IDs that exist in both tag and plot catalogs
tag_ids = set(artifacts.matrix_ids().tolist())
plot_ids_set = set(artifacts.plot_ids().tolist())
common_ids = list(tag_ids & plot_ids_set)
random.seed(42)
sample_seeds = random.sample(common_ids, min(100, len(common_ids)))

changed_count = 0
examples = []

for i, seed in enumerate(sample_seeds):
    try:
        res_off = run_rec([seed], suppress=False)
        res_on = run_rec([seed], suppress=True)
    except Exception as e:
        continue
    
    top50_off = set(r["movie_id"] for r in res_off["ranked"][:50])
    top50_on = set(r["movie_id"] for r in res_on["ranked"][:50])
    
    if top50_off != top50_on:
        changed_count += 1
        if len(examples) < 10:
            # Get the differences
            only_off = top50_off - top50_on
            only_on = top50_on - top50_off
            examples.append({
                "seed": seed,
                "removed_count": len(only_off),
                "added_count": len(only_on),
                "sample_removed": list(only_off)[:5],
                "sample_added": list(only_on)[:5],
            })
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/100 queries, {changed_count} changed so far...")

print(f"\nTotal queries with changed top-50: {changed_count} / {len(sample_seeds)}")
print(f"\n--- 10 example changes ---")
for ex in examples:
    print(f"  Seed {ex['seed']}: -{ex['removed_count']} +{ex['added_count']} "
          f"removed={ex['sample_removed']}, added={ex['sample_added']}")

# Save results for report
results = {
    "changed_count": changed_count,
    "total": len(sample_seeds),
    "examples": examples,
}
with open("bench/boilerplate_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to bench/boilerplate_analysis_results.json")
