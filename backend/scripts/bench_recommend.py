#!/usr/bin/env python3
"""
Benchmark script for retrieve_and_rank.
Measures latency (median and p95) across fine-grained pipeline stages and verifies ranking/scoring integrity.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Suppress overly chatty library logs during benchmark
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("posthog").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from app.config import settings
from app.db.pool import open_pool, close_pool
from app.db import queries
from app.db.queries import Filters, _where
from app.schemas import RecommendRequest, TagIn
from app.services import recommender
from app.stores import artifacts, chroma, text_embedder

FIXED_MOVIE_IDS = [238, 240, 278]  # The Godfather, The Godfather Part II, The Shawshank Redemption

TIMING_KEYS = [
    # query_builder
    "tag_part",
    "plot_centroid",
    "embed_plot_text",
    "query_build",
    # candidates
    "gate",
    "ann_tag",
    "ann_plot",
    "filter_ids",
    "candidates",
    # scoring
    "score_tag",
    "score_plot",
    "scoring",
    # fusion
    "fusion",
    # total
    "total",
]


def get_match_count(req: RecommendRequest) -> int | None:
    """Count matching movies in Postgres for a request's filters."""
    f = Filters.from_request(req)
    if not f.active:
        return None
    clauses, params = _where(f)
    sql = "SELECT COUNT(*) FROM movies"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    with queries.current_pool().connection() as conn:
        return conn.execute(sql, params).fetchone()["count"]


def get_cases() -> dict[str, dict]:
    """Define benchmark cases a through f."""
    cases = {
        "a": {
            "description": "3 selected movies, no filters",
            "is_slow": False,
            "request": RecommendRequest(
                movie_ids=FIXED_MOVIE_IDS,
                tag_weight=0.5,
            ),
        },
        "b": {
            "description": "3 selected movies + genre + language + date-range filters (large branch)",
            "is_slow": True,
            "request": RecommendRequest(
                movie_ids=FIXED_MOVIE_IDS,
                original_language=["en"],
                genres=["Drama"],
                date_from=date(1970, 1, 1),
                date_to=date(2020, 12, 31),
                tag_weight=0.5,
            ),
        },
        "c": {
            "description": "filters only, no movies (small branch)",
            "is_slow": False,
            "request": RecommendRequest(
                tags=[TagIn(tag_type="genres", name="Animation")],
                original_language=["ja"],
                genres=["Animation"],
                tag_weight=1.0,
            ),
        },
        "d": {
            "description": "3 movies + plot_text",
            "is_slow": False,
            "request": RecommendRequest(
                movie_ids=FIXED_MOVIE_IDS,
                plot_text="A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption across generations.",
                tag_weight=0.5,
            ),
        },
        "e": {
            "description": "3 movies + TIGHT filter between 50k and 200k (rare language + year range)",
            "is_slow": True,
            "request": RecommendRequest(
                movie_ids=FIXED_MOVIE_IDS,
                original_language=["fr"],
                date_from=date(1950, 1, 1),
                date_to=date(2024, 12, 31),
                tag_weight=0.5,
            ),
        },
        "f": {
            "description": "3 movies + LOOSE filter (single common genre)",
            "is_slow": True,
            "request": RecommendRequest(
                movie_ids=FIXED_MOVIE_IDS,
                genres=["Drama"],
                tag_weight=0.5,
            ),
        },
    }
    return cases


def run_benchmark(case_keys: list[str], n_fast: int, n_slow: int) -> tuple[dict, dict]:
    """Run warm-up and timed runs for selected cases. Return (results, stats)."""
    all_cases = get_cases()
    results = {}
    stats = {}

    for key in case_keys:
        if key not in all_cases:
            continue
        case_info = all_cases[key]
        req = case_info["request"]
        is_slow = case_info["is_slow"]
        n_runs = n_slow if is_slow else n_fast

        match_count = get_match_count(req)
        match_str = f", match_count={match_count}" if match_count is not None else ""
        print(f"\n[{key}] {case_info['description']}{match_str}")
        print(f"     Warm-up run (discarded)...", end="", flush=True)

        t_warm_start = time.perf_counter()
        recommender.retrieve_and_rank(req)
        print(f" done in {time.perf_counter() - t_warm_start:.2f}s")

        print(f"     Running {n_runs} timed iterations...", end="", flush=True)
        run_timings = []
        ranked_output = None
        pool_size = 0

        for r in range(n_runs):
            t_run_start = time.perf_counter()
            out = recommender.retrieve_and_rank(req)
            if ranked_output is None:
                ranked_output = [
                    {
                        "movie_id": item["movie_id"],
                        "final_score": round(item["final_score"], 6),
                        "tag_score": round(item["tag_score"], 6),
                        "plot_score": round(item["plot_score"], 6),
                    }
                    for item in out["ranked"]
                ]
                pool_size = out.get("pool_size", 0)

            # Ensure all standard timing keys exist
            timings = out.get("timings", {})
            run_timings.append(timings)
            print(".", end="", flush=True)
        print(f" completed.")

        # Compute median and p95 per key
        case_stats = {"n": n_runs, "match_count": match_count, "pool_size": pool_size, "metrics": {}}
        for tkey in TIMING_KEYS:
            vals = [t.get(tkey, 0.0) for t in run_timings if tkey in t]
            if vals:
                # convert to milliseconds for reporting
                ms_vals = [v * 1000.0 for v in vals]
                case_stats["metrics"][tkey] = {
                    "median_ms": round(float(np.percentile(ms_vals, 50)), 2),
                    "p95_ms": round(float(np.percentile(ms_vals, 95)), 2),
                    "raw_median_s": round(float(np.percentile(vals, 50)), 4),
                    "raw_p95_s": round(float(np.percentile(vals, 95)), 4),
                }

        results[key] = {
            "description": case_info["description"],
            "match_count": match_count,
            "pool_size": pool_size,
            "ranked": ranked_output,
        }
        stats[key] = case_stats

    return results, stats


def print_stats_table(stats: dict):
    """Print clean formatted table of benchmark latency stats."""
    cases = list(stats.keys())
    if not cases:
        return

    header_cols = [f"Case {c} (N={stats[c]['n']})" for c in cases]
    col_w = 20
    row_fmt = "{:<18} " + " ".join([f"{{:>{col_w}}}" for _ in cases])

    print("\n" + "=" * (20 + (col_w + 1) * len(cases)))
    print("BENCHMARK TIMINGS: MEDIAN / P95 (ms)")
    print("=" * (20 + (col_w + 1) * len(cases)))
    print(row_fmt.format("Metric", *header_cols))
    print("-" * (20 + (col_w + 1) * len(cases)))

    # Pool size and match count
    pool_sizes = [str(stats[c]["pool_size"]) for c in cases]
    print(row_fmt.format("pool_size", *pool_sizes))
    match_counts = [str(stats[c]["match_count"]) if stats[c]["match_count"] is not None else "-" for c in cases]
    print(row_fmt.format("match_count", *match_counts))
    print("-" * (20 + (col_w + 1) * len(cases)))

    for tkey in TIMING_KEYS:
        row_vals = []
        for c in cases:
            m = stats[c]["metrics"].get(tkey)
            if m is not None:
                row_vals.append(f"{m['median_ms']:.1f} / {m['p95_ms']:.1f}")
            else:
                row_vals.append("-")
        prefix = "  " if tkey not in ("query_build", "candidates", "scoring", "fusion", "total") else ""
        label = prefix + tkey
        print(row_fmt.format(label, *row_vals))
        if tkey in ("query_build", "candidates", "scoring", "fusion"):
            print("-" * (20 + (col_w + 1) * len(cases)))

    print("=" * (20 + (col_w + 1) * len(cases)) + "\n")


def compare_strict(current_results: dict, current_stats: dict, baseline_file: Path):
    """Compare against baseline with strict exact equality (same ids, same order, scores within 1e-6)."""
    with open(baseline_file, "r") as f:
        baseline_data = json.load(f)

    baseline_cases = baseline_data.get("cases", {})
    baseline_stats = baseline_data.get("stats", {})

    print(f"\n========================================================")
    print(f"STRICT COMPARISON MODE against: {baseline_file.name}")
    print(f"========================================================")

    for case_key, cur_case in current_results.items():
        if case_key not in baseline_cases:
            print(f"Warning: Case {case_key} not present in baseline. Skipping.")
            continue

        base_case = baseline_cases[case_key]
        cur_ranked = cur_case["ranked"]
        base_ranked = base_case["ranked"]

        if len(cur_ranked) != len(base_ranked):
            raise AssertionError(
                f"Case '{case_key}' output count mismatch: current={len(cur_ranked)}, baseline={len(base_ranked)}"
            )

        for i, (cur_item, base_item) in enumerate(zip(cur_ranked, base_ranked)):
            cid, bid = cur_item["movie_id"], base_item["movie_id"]
            if cid != bid:
                raise AssertionError(
                    f"Case '{case_key}' rank #{i} ID mismatch: current={cid}, baseline={bid}"
                )
            for score_field in ("final_score", "tag_score", "plot_score"):
                cs, bs = cur_item[score_field], base_item[score_field]
                if abs(cs - bs) > 1e-6:
                    raise AssertionError(
                        f"Case '{case_key}' rank #{i} (movie {cid}) {score_field} mismatch: "
                        f"current={cs}, baseline={bs}, diff={abs(cs-bs):.8f}"
                    )

        print(f" Case {case_key}: IDENTICAL output verified ({len(cur_ranked)} items, IDs, order, scores <= 1e-6)")

    # Print comparative timing table
    print_comparative_table(current_stats, baseline_stats)


def compare_scores_mode(current_results: dict, current_stats: dict, baseline_file: Path):
    """Check score integrity for common items and report overlap@50 and overlap@200."""
    with open(baseline_file, "r") as f:
        baseline_data = json.load(f)

    baseline_cases = baseline_data.get("cases", {})
    baseline_stats = baseline_data.get("stats", {})

    print(f"\n========================================================")
    print(f"SCORE INTEGRITY & OVERLAP MODE against: {baseline_file.name}")
    print(f"========================================================")

    for case_key, cur_case in current_results.items():
        if case_key not in baseline_cases:
            print(f"Warning: Case {case_key} not present in baseline. Skipping.")
            continue

        base_case = baseline_cases[case_key]
        cur_ranked = cur_case["ranked"]
        base_ranked = base_case["ranked"]

        cur_ids = [item["movie_id"] for item in cur_ranked]
        base_ids = [item["movie_id"] for item in base_ranked]

        base_map = {item["movie_id"]: item for item in base_ranked}
        cur_map = {item["movie_id"]: item for item in cur_ranked}

        # Check every common item for score equality
        common_ids = set(cur_map.keys()) & set(base_map.keys())
        for mid in common_ids:
            cur_item = cur_map[mid]
            base_item = base_map[mid]
            for score_field in ("final_score", "tag_score", "plot_score"):
                cs, bs = cur_item[score_field], base_item[score_field]
                if abs(cs - bs) > 1e-6:
                    raise AssertionError(
                        f"Case '{case_key}' movie {mid} {score_field} mismatch on common item: "
                        f"current={cs}, baseline={bs}, diff={abs(cs - bs):.8f}"
                    )

        k50 = min(50, len(cur_ids), len(base_ids))
        overlap50 = len(set(cur_ids[:k50]) & set(base_ids[:k50])) / k50 if k50 > 0 else 1.0
        k200 = min(200, len(cur_ids), len(base_ids))
        overlap200 = len(set(cur_ids[:k200]) & set(base_ids[:k200])) / k200 if k200 > 0 else 1.0

        print(
            f" Case {case_key}: Verified {len(common_ids)} common item scores (tol 1e-6) | "
            f"Overlap@50: {overlap50:.2f} ({int(overlap50 * k50)}/{k50}) | "
            f"Overlap@200: {overlap200:.2f} ({int(overlap200 * k200)}/{k200})"
        )

    print_comparative_table(current_stats, baseline_stats)


def print_comparative_table(current_stats: dict, baseline_stats: dict):
    """Print comparative table showing before/after median and percentage change."""
    cases = [c for c in current_stats if c in baseline_stats]
    if not cases:
        return

    col_w = 26
    row_fmt = "{:<16} " + " ".join([f"{{:>{col_w}}}" for _ in cases])
    header_cols = [f"Case {c} (Base -> Now)" for c in cases]

    print("\n" + "=" * (18 + (col_w + 1) * len(cases)))
    print("COMPARATIVE TIMINGS: MEDIAN ms (Baseline -> Current [% Diff])")
    print("=" * (18 + (col_w + 1) * len(cases)))
    print(row_fmt.format("Metric", *header_cols))
    print("-" * (18 + (col_w + 1) * len(cases)))

    for tkey in TIMING_KEYS:
        row_vals = []
        for c in cases:
            bm = baseline_stats[c]["metrics"].get(tkey)
            cm = current_stats[c]["metrics"].get(tkey)
            if bm and cm:
                b_med = bm["median_ms"]
                c_med = cm["median_ms"]
                diff = c_med - b_med
                pct = (diff / b_med * 100.0) if b_med > 0 else 0.0
                sign = "+" if diff > 0 else ""
                row_vals.append(f"{b_med:.1f} -> {c_med:.1f} ({sign}{pct:.1f}%)")
            elif cm:
                row_vals.append(f"- -> {cm['median_ms']:.1f}")
            else:
                row_vals.append("-")
        prefix = "  " if tkey not in ("query_build", "candidates", "scoring", "fusion", "total") else ""
        label = prefix + tkey
        print(row_fmt.format(label, *row_vals))
        if tkey in ("query_build", "candidates", "scoring", "fusion"):
            print("-" * (18 + (col_w + 1) * len(cases)))

    print("=" * (18 + (col_w + 1) * len(cases)) + "\n")


def resolve_file(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # Try relative to cwd or backend or repo root
    cwd_p = Path.cwd() / path_str
    if cwd_p.exists():
        return cwd_p
    backend_p = BACKEND_DIR / path_str
    if backend_p.exists():
        return backend_p
    root_p = BACKEND_DIR.parent / path_str
    if root_p.exists():
        return root_p
    return cwd_p


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieve_and_rank latency and verify scores.")
    parser.add_argument("--save-baseline", nargs="?", const="bench_baseline.json", default=None,
                        help="Save output to baseline JSON file (default: bench_baseline.json).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Strict comparison against baseline JSON file.")
    parser.add_argument("--compare-scores", type=str, default=None,
                        help="Score integrity and overlap comparison against baseline JSON file.")
    parser.add_argument("-N", "--n", type=int, default=None,
                        help="Set number of iterations for all cases.")
    parser.add_argument("--n-fast", type=int, default=30,
                        help="Number of iterations for fast cases a, c, d (default: 30).")
    parser.add_argument("--n-slow", type=int, default=3,
                        help="Number of iterations for slow cases b, e, f (default: 3).")
    parser.add_argument("--cases", nargs="+", default=["a", "b", "c", "d", "e", "f"],
                        help="Cases to run (default: a b c d e f).")
    args = parser.parse_args()

    if args.n is not None:
        args.n_fast = args.n
        args.n_slow = args.n

    if args.save_baseline is None and not args.compare and not args.compare_scores:
        args.save_baseline = "bench_baseline.json"

    # Reproducibility seeds
    random.seed(42)
    np.random.seed(42)

    print("Initializing environment...")
    open_pool()
    artifacts.load_all()
    chroma.get_collection(settings.plots_collection)
    chroma.get_collection(settings.tags_collection)
    text_embedder.get_sentence_transformer()
    print("Environment initialized.")

    try:
        results, stats = run_benchmark(args.cases, args.n_fast, args.n_slow)
        print_stats_table(stats)

        if args.compare:
            base_file = resolve_file(args.compare)
            compare_strict(results, stats, base_file)
        elif args.compare_scores:
            base_file = resolve_file(args.compare_scores)
            compare_scores_mode(results, stats, base_file)

        # If saving baseline requested
        if args.save_baseline:
            out_file = resolve_file(args.save_baseline)
            with open(out_file, "w") as f:
                json.dump({"cases": results, "stats": stats}, f, indent=2)
            print(f"Saved baseline results and stats to {out_file.resolve()}")

    finally:
        close_pool()


if __name__ == "__main__":
    main()
