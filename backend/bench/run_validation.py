#!/usr/bin/env python3
"""
Controlled Validation Harness for Movie Recommendation System.

Features:
- Configures single-threaded BLAS to measure pure pipeline parallelism.
- Tests 4 key retrieval cases: unfiltered, plot_text, 55k_filter, 295k_filter.
- Uses fixed seeds for determinism and reproducible query sets.
- Executes each case twice with randomized run order.
- Runs an 8-request concurrent burst through PIPELINE_SEM=4.
- Collects per-request latency, pool size, throughput, RSS, free RAM (psutil), and page faults.
- Disables the rate limiter only inside the benchmark (app.state.limiter.enabled = False).
- Cross-platform: fully compatible with Linux and Windows.
- Generates a single, consolidated validation_results.json.
"""

import os
import sys

# Configure BLAS/OpenMP threads BEFORE importing numpy/scipy/torch/onnxruntime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import psutil

# Ensure backend root is on sys.path
BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.main import app
from app.stores import artifacts


def get_free_ram_gb() -> float:
    """Return available system RAM in gigabytes."""
    return psutil.virtual_memory().available / (1024.0 ** 3)


def get_process_rss_mb(process: psutil.Process) -> float:
    """Return process Resident Set Size in megabytes."""
    return process.memory_info().rss / (1024.0 ** 2)


def get_page_faults(process: psutil.Process) -> Optional[int]:
    """
    Return cumulative page faults cross-platform.
    On Windows: process.memory_info().num_page_faults
    On Linux/Unix: ru_majflt + ru_minflt via resource module.
    """
    try:
        mem = process.memory_info()
        if hasattr(mem, "num_page_faults"):
            return int(mem.num_page_faults)
    except Exception:
        pass

    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_majflt + ru.ru_minflt)
    except Exception:
        pass

    return None


def get_system_snapshot(process: psutil.Process) -> Dict[str, Any]:
    """Capture current system and process memory metrics."""
    return {
        "free_ram_gb": round(get_free_ram_gb(), 3),
        "rss_mb": round(get_process_rss_mb(process), 2),
        "page_faults": get_page_faults(process),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        import subprocess
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(BACKEND_DIR))
        return res.decode().strip()
    except Exception:
        return "unknown"


def get_active_config_flags() -> Dict[str, Any]:
    """Return dictionary of active settings and configuration flags."""
    attrs = [
        "pipeline_sem",
        "exact_limit",
        "starvation_hard_cap",
        "min_survivors",
        "pool_cap",
        "max_results",
        "k_tag",
        "k_plot",
        "use_matvec_tag_candidates",
        "use_uncapped_pool",
        "tie_break_plot",
        "tag_dense_matvec_threshold",
        "suppress_boilerplate_plots",
        "app_env",
        "ann_k",
    ]
    flags: Dict[str, Any] = {}
    for attr in attrs:
        if hasattr(settings, attr):
            flags[attr] = getattr(settings, attr)
    return flags


def get_plot_suppressed_count() -> int:
    """Return the count of rows marked True in plot_suppressed.npy mask."""
    try:
        suppressed = artifacts.plot_suppressed()
        return int(np.sum(suppressed))
    except Exception as e:
        return -1


async def send_recommend_request(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a single recommendation request and record timings."""
    t0 = time.perf_counter()
    resp = await client.post("/api/recommend", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result: Dict[str, Any] = {
        "status_code": resp.status_code,
        "elapsed_ms": round(elapsed_ms, 2),
        "success": resp.status_code == 200,
        "results_count": 0,
        "pool_size": None,
    }

    if resp.status_code == 200:
        try:
            data = resp.json()
            items = data.get("results", [])
            result["results_count"] = len(items)
            result["pool_size"] = data.get("pool_size")
        except Exception:
            pass
    else:
        result["error"] = resp.text[:200]

    return result


async def run_benchmark(
    queries_per_case: int,
    concurrency_iterations: int,
    seed: int,
    output_path: Path,
    smoke_mode: bool,
) -> None:
    process = psutil.Process()
    stage_metrics: Dict[str, Any] = {}
    metrics_start = get_system_snapshot(process)
    stage_metrics["start"] = metrics_start

    print("=" * 80)
    print("CONTROLLED VALIDATION BENCHMARK HARNESS")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()} | CPU Cores: {os.cpu_count()}")
    print(f"RNG Seed: {seed} | Queries per case: {queries_per_case} | Smoke mode: {smoke_mode}")
    print("=" * 80)

    # Force pipeline settings
    settings.use_matvec_tag_candidates = True
    settings.use_uncapped_pool = True
    settings.k_tag = 10000
    settings.k_plot = 5000
    settings.exact_limit = 300000

    print("Starting FastAPI application lifespan (loading artifacts, meta index, pools)...")
    async with app.router.lifespan_context(app):
        stage_metrics["after_lifespan"] = get_system_snapshot(process)
        # Explicit requirement: disable rate limiter only inside benchmark
        rate_limiter = getattr(app.state, "limiter", None)
        original_limiter_enabled = True
        if rate_limiter is not None:
            original_limiter_enabled = getattr(rate_limiter, "enabled", True)
            rate_limiter.enabled = False
            print("[INFO] Rate limiter disabled for benchmark run.")

        # Discover common IDs
        tag_ids = set(artifacts.matrix_ids())
        _, plot_id_to_r = artifacts.plot_memmap()
        plot_ids = set(plot_id_to_r.keys())
        common_ids = np.array(sorted(tag_ids & plot_ids), dtype=np.int64)
        print(f"[INFO] Common movies in tag matrix and plot embeddings: {len(common_ids):,}")

        rng = np.random.default_rng(seed)

        # Build cases definitions
        case_definitions = {
            "unfiltered": lambda seeds: {
                "movie_ids": seeds,
                "tag_weight": 0.5,
            },
            "plot_text": lambda seeds: {
                "movie_ids": seeds,
                "plot_text": "A thrilling tale of organized crime, family loyalty, betrayal, and a quest for redemption.",
                "tag_weight": 0.5,
            },
            "55k_filter": lambda seeds: {
                "movie_ids": seeds,
                "original_language": ["ja"],
                "tag_weight": 0.5,
            },
            "295k_filter": lambda seeds: {
                "movie_ids": seeds,
                "genres": ["Drama"],
                "tag_weight": 0.5,
            },
        }

        # Setup randomized execution order: 4 cases × 2 runs = 8 runs
        case_names = ["unfiltered", "plot_text", "55k_filter", "295k_filter"]
        run_plan: List[Dict[str, Any]] = [
            {"case_name": name, "pass_num": p}
            for p in (1, 2)
            for name in case_names
        ]
        # Deterministically shuffle the run order
        rng.shuffle(run_plan)

        print("\nRandomized execution sequence for sequential runs:")
        for idx, item in enumerate(run_plan, 1):
            print(f"  Stage {idx}/8: {item['case_name']} (pass {item['pass_num']})")

        metrics_start = get_system_snapshot(process)
        print(f"\n[System Start] Free RAM: {metrics_start['free_ram_gb']:.2f} GB | Process RSS: {metrics_start['rss_mb']:.1f} MB | Page Faults: {metrics_start['page_faults']}")

        sequential_results: List[Dict[str, Any]] = []

        # Execute using in-process ASGI Transport (no network/port binding required)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://benchmark-harness",
            timeout=180.0,
        ) as client:

            # -------------------------------------------------------------
            # Part 1: Sequential Benchmark Runs
            # -------------------------------------------------------------
            print("\n" + "-" * 80)
            print("PART 1: SEQUENTIAL RUNS (EACH CASE RUN TWICE IN RANDOMIZED ORDER)")
            print("-" * 80)

            for step_idx, step in enumerate(run_plan, 1):
                case_name = step["case_name"]
                pass_num = step["pass_num"]
                req_builder = case_definitions[case_name]

                stage_metrics[f"seq_{step_idx}_{case_name}_pass{pass_num}_start"] = get_system_snapshot(process)
                print(f"\n--> [{step_idx}/8] Running {case_name} (Pass {pass_num}) - {queries_per_case} distinct queries...")
                stage_ram_start = get_free_ram_gb()
                stage_rss_start = get_process_rss_mb(process)

                run_latencies: List[float] = []
                run_pools: List[int] = []
                detailed_records: List[Dict[str, Any]] = []

                # Fixed per-case deterministic seed stream
                case_seed = seed + (hash((case_name, pass_num)) % 100_000)
                case_rng = np.random.default_rng(case_seed)

                for q_idx in range(queries_per_case):
                    seeds = case_rng.choice(common_ids, size=3, replace=False).tolist()
                    payload = req_builder(seeds)

                    res = await send_recommend_request(client, payload)
                    run_latencies.append(res["elapsed_ms"])
                    if res["pool_size"] is not None:
                        run_pools.append(res["pool_size"])
                    detailed_records.append(res)

                    if (q_idx + 1) % max(1, queries_per_case // 5) == 0 or (q_idx + 1) == queries_per_case:
                        print(f"    Completed {q_idx + 1}/{queries_per_case} queries (latest: {res['elapsed_ms']:.1f} ms)")

                lats_arr = np.array(run_latencies, dtype=np.float64)
                p50 = float(np.percentile(lats_arr, 50))
                p95 = float(np.percentile(lats_arr, 95))
                pmax = float(np.max(lats_arr))
                pmin = float(np.min(lats_arr))
                pmean = float(np.mean(lats_arr))

                pool_arr = np.array(run_pools, dtype=np.float64) if run_pools else np.array([0])
                pool_p50 = float(np.percentile(pool_arr, 50))

                stage_ram_end = get_free_ram_gb()
                stage_rss_end = get_process_rss_mb(process)
                stage_metrics[f"seq_{step_idx}_{case_name}_pass{pass_num}_end"] = get_system_snapshot(process)

                print(f"    Results: p50={p50:.1f}ms | p95={p95:.1f}ms | max={pmax:.1f}ms | mean={pmean:.1f}ms | pool p50={pool_p50:.0f}")
                print(f"    Memory: RSS {stage_rss_start:.1f} -> {stage_rss_end:.1f} MB | Free RAM {stage_ram_start:.2f} -> {stage_ram_end:.2f} GB")

                sequential_results.append({
                    "step_index": step_idx,
                    "case_name": case_name,
                    "pass_num": pass_num,
                    "queries_count": queries_per_case,
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "max_ms": round(pmax, 2),
                    "min_ms": round(pmin, 2),
                    "mean_ms": round(pmean, 2),
                    "pool_p50": pool_p50,
                    "latencies_ms": run_latencies,
                    "rss_start_mb": round(stage_rss_start, 2),
                    "rss_end_mb": round(stage_rss_end, 2),
                    "free_ram_start_gb": round(stage_ram_start, 3),
                    "free_ram_end_gb": round(stage_ram_end, 3),
                })

            # -------------------------------------------------------------
            # Part 2: 8-Request Concurrency Test at PIPELINE_SEM=4
            # -------------------------------------------------------------
            print("\n" + "-" * 80)
            print("PART 2: CONCURRENCY TEST (8 DISTINCT REQUESTS THROUGH PIPELINE_SEM=4)")
            print("-" * 80)

            concurrency_results: List[Dict[str, Any]] = []

            for it in range(concurrency_iterations):
                print(f"\n--> Concurrency Iteration {it + 1}/{concurrency_iterations} (Burst of 8 simultaneous requests)...")
                burst_rng = np.random.default_rng(seed + 9999 + it * 100)

                # Construct 8 distinct requests (2 unfiltered, 2 plot_text, 2 55k, 2 295k)
                burst_payloads: List[Tuple[str, Dict[str, Any]]] = []
                for cname in case_names:
                    for _ in range(2):
                        s = burst_rng.choice(common_ids, size=3, replace=False).tolist()
                        burst_payloads.append((cname, case_definitions[cname](s)))

                # Fire all 8 requests simultaneously into ASGI pipeline
                stage_metrics[f"concurrency_burst_{it + 1}_start"] = get_system_snapshot(process)
                c_ram_start = get_free_ram_gb()
                c_rss_start = get_process_rss_mb(process)
                t_burst_start = time.perf_counter()

                tasks = [
                    send_recommend_request(client, p)
                    for _, p in burst_payloads
                ]
                responses = await asyncio.gather(*tasks)

                wall_elapsed_ms = (time.perf_counter() - t_burst_start) * 1000.0
                c_ram_end = get_free_ram_gb()
                c_rss_end = get_process_rss_mb(process)
                stage_metrics[f"concurrency_burst_{it + 1}_end"] = get_system_snapshot(process)

                c_lats = [r["elapsed_ms"] for r in responses]
                c_lats_arr = np.array(c_lats, dtype=np.float64)
                throughput_rps = (len(responses) / (wall_elapsed_ms / 1000.0)) if wall_elapsed_ms > 0 else 0.0

                c_p50 = float(np.percentile(c_lats_arr, 50))
                c_p95 = float(np.percentile(c_lats_arr, 95))
                c_mean = float(np.mean(c_lats_arr))
                success_count = sum(1 for r in responses if r["success"])

                print(f"    Burst Wall Time: {wall_elapsed_ms:.1f} ms | Throughput: {throughput_rps:.2f} req/s")
                print(f"    Per-request: p50={c_p50:.1f}ms | p95={c_p95:.1f}ms | mean={c_mean:.1f}ms | success={success_count}/8")
                print(f"    Memory: RSS {c_rss_start:.1f} -> {c_rss_end:.1f} MB | Free RAM {c_ram_start:.2f} -> {c_ram_end:.2f} GB")

                concurrency_results.append({
                    "iteration": it + 1,
                    "batch_size": 8,
                    "pipeline_sem": 4,
                    "wall_ms": round(wall_elapsed_ms, 2),
                    "throughput_rps": round(throughput_rps, 2),
                    "success_count": success_count,
                    "per_request_p50_ms": round(c_p50, 2),
                    "per_request_p95_ms": round(c_p95, 2),
                    "per_request_mean_ms": round(c_mean, 2),
                    "requests": [
                        {
                            "case": burst_payloads[i][0],
                            "elapsed_ms": responses[i]["elapsed_ms"],
                            "status_code": responses[i]["status_code"],
                            "pool_size": responses[i]["pool_size"],
                        }
                        for i in range(len(responses))
                    ],
                    "rss_start_mb": round(c_rss_start, 2),
                    "rss_end_mb": round(c_rss_end, 2),
                    "free_ram_start_gb": round(c_ram_start, 3),
                    "free_ram_end_gb": round(c_ram_end, 3),
                })

        # Re-enable rate limiter
        if rate_limiter is not None:
            rate_limiter.enabled = original_limiter_enabled
            print("[INFO] Rate limiter restored to original state.")

        metrics_end = get_system_snapshot(process)
        stage_metrics["end"] = metrics_end
        page_faults_start = metrics_start["page_faults"]
        page_faults_end = metrics_end["page_faults"]
        pf_delta = (page_faults_end - page_faults_start) if (page_faults_start is not None and page_faults_end is not None) else None

        # Consolidate complete results payload
        payload = {
            "metadata": {
                "benchmark_name": "Controlled Validation Harness",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": get_git_commit(),
                "plot_suppressed_marked_count": get_plot_suppressed_count(),
                "smoke_mode": smoke_mode,
                "rng_seed": seed,
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                    "cpu_count": os.cpu_count(),
                    "total_ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
                },
                "active_config_flags": get_active_config_flags(),
                "benchmark_config": {
                    "pipeline_sem": 4,
                    "blas_threads": 1,
                    "exact_limit": settings.exact_limit,
                    "use_matvec_tag_candidates": settings.use_matvec_tag_candidates,
                    "use_uncapped_pool": settings.use_uncapped_pool,
                    "k_tag": settings.k_tag,
                    "k_plot": settings.k_plot,
                    "queries_per_case": queries_per_case,
                    "concurrency_batch_size": 8,
                },
            },
            "system_metrics": {
                "start": metrics_start,
                "end": metrics_end,
                "delta": {
                    "free_ram_delta_gb": round(metrics_end["free_ram_gb"] - metrics_start["free_ram_gb"], 3),
                    "rss_delta_mb": round(metrics_end["rss_mb"] - metrics_start["rss_mb"], 2),
                    "page_faults_delta": pf_delta,
                },
                "free_ram_per_stage": stage_metrics,
            },
            "sequential_runs": sequential_results,
            "concurrency_runs": concurrency_results,
        }

        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("\n" + "=" * 80)
        print(f"BENCHMARK COMPLETE - RESULTS SAVED TO: {output_path}")
        print("=" * 80)
        print(f"Git Commit:       {payload['metadata']['git_commit']}")
        print(f"Plot Suppressed:  {payload['metadata']['plot_suppressed_marked_count']:,} marked rows")
        print(f"Initial Free RAM: {metrics_start['free_ram_gb']:.2f} GB | Final Free RAM: {metrics_end['free_ram_gb']:.2f} GB")
        print(f"Initial RSS:      {metrics_start['rss_mb']:.1f} MB | Final RSS:      {metrics_end['rss_mb']:.1f} MB")
        if pf_delta is not None:
            print(f"Page Faults:      +{pf_delta:,}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run controlled validation benchmark harness.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test (1 query per case, 1 concurrency burst) safe for dev laptops.",
    )
    parser.add_argument(
        "--queries-per-case",
        type=int,
        default=25,
        help="Number of distinct queries per case run (default: 25). Ignored in --smoke mode.",
    )
    parser.add_argument(
        "--concurrency-iterations",
        type=int,
        default=2,
        help="Number of 8-request concurrent bursts to execute (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH_DIR / "validation_results.json",
        help="Destination JSON file path.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute the benchmark inside a Docker container with --memory=6g.",
    )
    parser.add_argument(
        "--docker-memory",
        type=str,
        default="6g",
        help="Memory constraint for docker run (default: 6g).",
    )

    args = parser.parse_args()

    if args.docker:
        import subprocess
        docker_cmd = [
            "docker", "run", "--rm",
            f"--memory={args.docker_memory}",
            "--network=host",
            "-v", f"{BACKEND_DIR.resolve()}:/srv",
            "-w", "/srv",
            "-e", "PYTHONPATH=.",
            "movie_recommender_backend:latest",
            "python", "bench/run_validation.py",
        ]
        if args.smoke:
            docker_cmd.append("--smoke")
        if args.queries_per_case != 25 and not args.smoke:
            docker_cmd.extend(["--queries-per-case", str(args.queries_per_case)])
        if args.concurrency_iterations != 2 and not args.smoke:
            docker_cmd.extend(["--concurrency-iterations", str(args.concurrency_iterations)])
        if args.seed != 42:
            docker_cmd.extend(["--seed", str(args.seed)])

        print("=" * 80)
        print(f"DOCKER MODE ENABLED (--memory={args.docker_memory})")
        print("Constructed Docker command:")
        print(" ".join(docker_cmd))
        print("=" * 80)
        try:
            res = subprocess.run(docker_cmd)
            sys.exit(res.returncode)
        except Exception as e:
            print(f"[ERROR] Failed to run Docker command: {e}")
            sys.exit(1)

    queries = 1 if args.smoke else args.queries_per_case
    conc_iters = 1 if args.smoke else args.concurrency_iterations

    asyncio.run(
        run_benchmark(
            queries_per_case=queries,
            concurrency_iterations=conc_iters,
            seed=args.seed,
            output_path=args.output,
            smoke_mode=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
