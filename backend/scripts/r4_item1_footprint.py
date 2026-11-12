"""
Round 4 Item 1: Measured footprint — Flag OFF vs ON.
Measures private bytes, working set, free system RAM via PowerShell / ctypes.
Run with: python scripts/r4_item1_footprint.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, time, json, ctypes, gc, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_system_mem():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_phys_gb": stat.ullTotalPhys / (1024**3),
        "avail_phys_gb": stat.ullAvailPhys / (1024**3),
        "mem_load_pct": stat.dwMemoryLoad,
    }


def get_process_mem():
    """Use tasklist and wmic to measure process memory."""
    pid = os.getpid()
    mem = {"pid": pid}

    # Try wmic
    try:
        result = subprocess.run(
            ["wmic", "process", "where", f"ProcessId={pid}", "get",
             "WorkingSetSize,PrivatePageCount", "/format:csv"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("Node"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        mem["private_bytes_mb"] = int(parts[1]) / (1024**2)
                        mem["working_set_mb"] = int(parts[2]) / (1024**2)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        mem["wmic_error"] = str(e)

    # Also try tasklist as backup
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/V"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            if str(pid) in line:
                mem["tasklist_raw"] = line.strip()
    except Exception as e:
        mem["tasklist_error"] = str(e)

    # Also try PowerShell Get-Process
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-Process -Id {pid} | Select-Object WorkingSet64,PrivateMemorySize64,PagedMemorySize64,VirtualMemorySize64 | ConvertTo-Json"],
            capture_output=True, text=True, timeout=10
        )
        pdata = json.loads(result.stdout.strip())
        mem["ps_working_set_mb"] = pdata.get("WorkingSet64", 0) / (1024**2)
        mem["ps_private_mb"] = pdata.get("PrivateMemorySize64", 0) / (1024**2)
        mem["ps_paged_mb"] = pdata.get("PagedMemorySize64", 0) / (1024**2)
        mem["ps_virtual_mb"] = pdata.get("VirtualMemorySize64", 0) / (1024**2)
    except Exception as e:
        mem["ps_error"] = str(e)

    return mem


def measure_component_sizes():
    """Measure individual data structure sizes in memory."""
    import numpy as np
    import scipy.sparse as sp
    from app.stores import artifacts, meta_index

    sizes = {}

    # Tag CSR matrix
    m = artifacts.tag_matrix()
    sizes["tag_csr_data_mb"] = m.data.nbytes / (1024**2)
    sizes["tag_csr_indices_mb"] = m.indices.nbytes / (1024**2)
    sizes["tag_csr_indptr_mb"] = m.indptr.nbytes / (1024**2)
    sizes["tag_csr_total_mb"] = (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / (1024**2)
    sizes["tag_csr_shape"] = list(m.shape)

    # Row norms
    rn = artifacts.row_norms()
    sizes["tag_row_norms_mb"] = rn.nbytes / (1024**2)

    # Matrix IDs
    mids = artifacts.matrix_ids()
    sizes["tag_matrix_ids_mb"] = mids.nbytes / (1024**2)

    # id_to_row dict (estimate: ~120 bytes per entry for Python dict)
    i2r = artifacts.id_to_row()
    sizes["id_to_row_entries"] = len(i2r)
    sizes["id_to_row_est_mb"] = len(i2r) * 120 / (1024**2)

    # Plot memmap
    emb, id_to_r = artifacts.plot_memmap()
    sizes["plot_memmap_shape"] = list(emb.shape)
    sizes["plot_memmap_dtype"] = str(emb.dtype)
    sizes["plot_memmap_virtual_mb"] = (emb.shape[0] * emb.shape[1] * emb.dtype.itemsize) / (1024**2)

    # Plot id_to_row dict
    sizes["plot_id_to_row_entries"] = len(id_to_r)
    sizes["plot_id_to_row_est_mb"] = len(id_to_r) * 120 / (1024**2)

    # MetaIndex
    idx = meta_index.get_index()
    meta_total = 0
    for attr_name in ["ids", "languages", "years", "dates", "genres"]:
        arr = getattr(idx, attr_name, None)
        if arr is not None and hasattr(arr, "nbytes"):
            mb = arr.nbytes / (1024**2)
            sizes[f"meta_{attr_name}_mb"] = mb
            meta_total += mb
    sizes["meta_total_arrays_mb"] = meta_total
    sizes["meta_id_to_row_entries"] = len(idx.id_to_row)
    sizes["meta_id_to_row_est_mb"] = len(idx.id_to_row) * 120 / (1024**2)

    # Vectorizers (rough estimate)
    vecs = artifacts.vectorizers()
    sizes["vectorizers_count"] = len(vecs)
    total_vocab = sum(len(v.vocabulary_) for v in vecs)
    sizes["vectorizers_total_vocab"] = total_vocab

    return sizes


def run_footprint(flag_on: bool):
    label = "ON" if flag_on else "OFF"
    print(f"\n{'='*70}", flush=True)
    print(f"MEASURING FOOTPRINT WITH use_matvec_tag_candidates = {label}", flush=True)
    print(f"{'='*70}", flush=True)

    # Force GC before measurement
    gc.collect()

    sys_before = get_system_mem()
    proc_before = get_process_mem()
    print(f"BEFORE loading:", flush=True)
    print(f"  System: {sys_before['avail_phys_gb']:.2f} GB free / {sys_before['total_phys_gb']:.2f} GB total ({sys_before['mem_load_pct']}% used)", flush=True)
    print(f"  Process: {json.dumps({k:v for k,v in proc_before.items() if 'mb' in k.lower()}, indent=2)}", flush=True)

    # Import and configure
    from app.config import settings
    settings.use_matvec_tag_candidates = flag_on

    from app.db.pool import open_pool
    open_pool()

    from app.stores import artifacts, meta_index, chroma

    t0 = time.perf_counter()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    if not flag_on:
        chroma.get_collection(settings.tags_collection)

    load_time = time.perf_counter() - t0
    print(f"Loading took {load_time:.2f}s", flush=True)

    # Warmup: do a few recommendations
    import numpy as np
    from app.schemas import RecommendRequest
    from app.services import recommender
    rng = np.random.default_rng(42)
    all_ids = list(artifacts.id_to_row().keys())
    for _ in range(3):
        seeds = rng.choice(all_ids, size=3, replace=False).tolist()
        try:
            recommender.retrieve_and_rank(RecommendRequest(movie_ids=seeds, tag_weight=0.5))
        except Exception:
            pass

    gc.collect()
    time.sleep(1)

    sys_after = get_system_mem()
    proc_after = get_process_mem()
    print(f"\nAFTER loading + warmup:", flush=True)
    print(f"  System: {sys_after['avail_phys_gb']:.2f} GB free / {sys_after['total_phys_gb']:.2f} GB total ({sys_after['mem_load_pct']}% used)", flush=True)
    print(f"  Process: {json.dumps({k:v for k,v in proc_after.items() if 'mb' in k.lower()}, indent=2)}", flush=True)

    comp = measure_component_sizes()
    print(f"\nComponent sizes:", flush=True)
    for k, v in comp.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)

    return {
        "flag": label,
        "sys_before": sys_before,
        "sys_after": sys_after,
        "proc_before": proc_before,
        "proc_after": proc_after,
        "components": comp,
        "load_time_s": load_time,
    }


if __name__ == "__main__":
    # We can only run one flag state per process (artifacts are cached).
    # Default to current config (production: ON).
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", choices=["on", "off"], default="on")
    args = parser.parse_args()

    result = run_footprint(flag_on=(args.flag == "on"))
    out_file = SCRIPT_DIR / f"r4_footprint_{args.flag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_file}", flush=True)
