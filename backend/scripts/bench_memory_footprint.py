import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import ctypes
from ctypes import wintypes
import time
import json
import sqlite3
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

psapi = ctypes.windll.psapi
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def get_mem():
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize, pmc.PageFaultCount

def get_chroma_segment_size(coll_name):
    from app.config import settings
    sqlite_path = os.path.join(settings.chroma_path, "chroma.sqlite3")
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT s.id FROM segments s JOIN collections c ON s.collection = c.id WHERE c.name = ? AND s.type = 'urn:chroma:segment/vector/hnsw-local-persisted'",
        (coll_name,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return 0
    seg_dir = os.path.join(settings.chroma_path, row[0])
    size = 0
    if os.path.exists(seg_dir):
        for root, dirs, files in os.walk(seg_dir):
            for f in files:
                size += os.path.getsize(os.path.join(root, f))
    return size

def main():
    ws_base, _ = get_mem()
    print("=" * 70, flush=True)
    print("MEMORY FOOTPRINT TABLE (RESIDENT / ALLOCATED BYTES AFTER WARM-UP)", flush=True)
    print("=" * 70, flush=True)
    print(f"Python Baseline Working Set: {ws_base / 1024**2:.2f} MB ({ws_base:,} bytes)", flush=True)

    # 1. meta_index
    t0 = time.perf_counter()
    from app.db.pool import open_pool, close_pool
    open_pool()
    from app.stores import meta_index
    meta_index.load_index()
    idx = meta_index.get_index()
    meta_bytes = (
        idx.ids.nbytes +
        idx.languages.nbytes +
        idx.years.nbytes +
        idx.dates.nbytes +
        idx.genres.nbytes +
        sys.getsizeof(idx.id_to_row) +
        sys.getsizeof(idx.lang_to_code) +
        sys.getsizeof(idx.code_to_lang) +
        sys.getsizeof(idx.genre_to_bit) +
        sys.getsizeof(idx.bit_to_genre)
    )
    ws_meta, _ = get_mem()
    delta_meta = ws_meta - ws_base
    print(f"meta_index: Arrays={meta_bytes / 1024**2:.2f} MB, Resident Delta={delta_meta / 1024**2:.2f} MB", flush=True)

    # 2. artifacts (tag CSR matrix, SVD, row_norms, plot memmap)
    from app.stores import artifacts
    tag_m = artifacts.tag_matrix()
    tag_csr_bytes = tag_m.data.nbytes + tag_m.indices.nbytes + tag_m.indptr.nbytes
    tag_ids_bytes = artifacts.matrix_ids().nbytes
    tag_norms_bytes = artifacts.row_norms().nbytes
    total_tag_bytes = tag_csr_bytes + tag_ids_bytes + tag_norms_bytes
    ws_tag, _ = get_mem()
    delta_tag = ws_tag - ws_meta
    print(f"tag CSR matrix + norms + ids: Allocated={total_tag_bytes / 1024**2:.2f} MB (data={tag_m.data.nbytes/1024**2:.1f}MB, indices={tag_m.indices.nbytes/1024**2:.1f}MB, indptr={tag_m.indptr.nbytes/1024**2:.1f}MB, norms={tag_norms_bytes/1024**2:.1f}MB), Resident Delta={delta_tag / 1024**2:.2f} MB", flush=True)

    svd_model = artifacts.svd()
    svd_bytes = svd_model.components_.nbytes
    print(f"SVD components ({svd_model.components_.shape}): Allocated={svd_bytes / 1024**2:.2f} MB", flush=True)

    # plot memmap
    plot_mem, plot_id_to_r = artifacts.plot_memmap()
    plot_mem_bytes = plot_mem.nbytes
    print(f"plot memmap ({plot_mem.shape}, float32): File/Buffer={plot_mem_bytes / 1024**2:.2f} MB ({plot_mem_bytes / 1024**3:.2f} GB)", flush=True)

    # 3. sentence-transformer
    from app.stores import text_embedder
    m = text_embedder.get_sentence_transformer()
    st_param_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    st_buffer_bytes = sum(b.numel() * b.element_size() for b in m.buffers())
    total_st_bytes = st_param_bytes + st_buffer_bytes
    ws_st, _ = get_mem()
    delta_st = ws_st - ws_tag
    print(f"sentence-transformer ({type(m).__name__}): Model Parameters={total_st_bytes / 1024**2:.2f} MB, Resident Delta={delta_st / 1024**2:.2f} MB", flush=True)

    # 4. Chroma HNSW collections
    from app.config import settings
    tags_hnsw_disk = get_chroma_segment_size(settings.tags_collection)
    plots_hnsw_disk = get_chroma_segment_size(settings.plots_collection)
    print(f"tags_vectors HNSW index: On Disk={tags_hnsw_disk / 1024**2:.2f} MB ({tags_hnsw_disk / 1024**3:.2f} GB)", flush=True)
    print(f"plots_embeddings HNSW index: On Disk={plots_hnsw_disk / 1024**2:.2f} MB ({plots_hnsw_disk / 1024**3:.2f} GB)", flush=True)

    # 5. Full Resident Footprint Summary
    # To keep everything fully resident in RAM:
    # Everything mapped or allocated must stay in memory without page-out
    print("\n" + "=" * 70, flush=True)
    print("MINIMUM RESIDENT RAM CALCULATION", flush=True)
    print("=" * 70, flush=True)

    table = [
        ("Python Baseline & App Runtime", ws_base),
        ("meta_index (in-memory arrays)", meta_bytes),
        ("tag CSR matrix, norms, & IDs", total_tag_bytes),
        ("SVD components (300 x 99,116)", svd_bytes),
        ("sentence-transformer model", total_st_bytes),
        ("plot memmap (1,023,484 x 384 f32)", plot_mem_bytes),
        ("tags_vectors HNSW index (Chroma)", tags_hnsw_disk),
        ("plots_embeddings HNSW index (Chroma)", plots_hnsw_disk),
    ]

    total_resident = 0
    print(f"{'Component':<40} {'Bytes':<15} {'MB':<12} {'GB':<10}")
    print("-" * 75)
    for name, b in table:
        total_resident += b
        print(f"{name:<40} {b:<15,d} {b / 1024**2:<12.2f} {b / 1024**3:<10.3f}")
    print("-" * 75)
    print(f"{'Total App Data Resident Footprint':<40} {total_resident:<15,d} {total_resident / 1024**2:<12.2f} {total_resident / 1024**3:<10.3f}")
    
    os_headroom = 1024**3  # 1 GB
    min_ram_with_headroom = total_resident + os_headroom
    print(f"{'+ 1 GB OS Headroom':<40} {os_headroom:<15,d} {os_headroom / 1024**2:<12.2f} {os_headroom / 1024**3:<10.3f}")
    print("=" * 75)
    print(f"{'MINIMUM SYSTEM RAM REQUIRED':<40} {min_ram_with_headroom:<15,d} {min_ram_with_headroom / 1024**2:<12.2f} {min_ram_with_headroom / 1024**3:<10.3f}")
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
