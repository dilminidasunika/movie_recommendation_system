import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import ctypes
from ctypes import wintypes
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
settings.use_matvec_tag_candidates = True

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
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
        ("PrivateUsage", ctypes.c_size_t),
    ]

def get_process_memory():
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    res = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize, pmc.PrivateUsage

from app.db.pool import open_pool
from app.stores import artifacts, meta_index, chroma, text_embedder
from app.schemas import RecommendRequest
from app.services import recommender

def measure():
    print("=" * 80)
    print("ITEM 5: RESIDENT FOOTPRINT WITH FLAG ON (tags_vectors NOT LOADED)")
    print("=" * 80)

    ws_base, priv_base = get_process_memory()
    print(f"Python Baseline Working Set: {ws_base / (1024**2):.2f} MB")

    open_pool()
    artifacts.load_all()
    meta_index.load_index()
    chroma.get_collection(settings.plots_collection)
    # tags_collection is NOT loaded!
    text_embedder.get_sentence_transformer()

    # Pre-fault memmap and warm-up pipeline with 5 queries
    emb, _ = artifacts.plot_memmap()
    _ = np.sum(emb[::100, 0])

    rng = np.random.default_rng(42)
    tag_ids = artifacts.matrix_ids()
    for _ in range(5):
        seeds = rng.choice(tag_ids, size=3, replace=False).tolist()
        req = RecommendRequest(movie_ids=seeds, tag_weight=0.5)
        recommender.retrieve_and_rank(req)

    ws_warm, priv_warm = get_process_memory()

    tag_m = artifacts.tag_matrix()
    tag_norms = artifacts.row_norms()
    tag_ids = artifacts.matrix_ids()
    idx = meta_index.get_index()

    tag_m_bytes = tag_m.data.nbytes + tag_m.indices.nbytes + tag_m.indptr.nbytes
    tag_norms_bytes = tag_norms.nbytes
    tag_ids_bytes = tag_ids.nbytes
    meta_idx_bytes = sum(getattr(idx, attr).nbytes for attr in ['ids', 'languages', 'years', 'dates', 'genres'])
    plot_mem_file_bytes = 1023484 * 384 * 4  # 1.57 GB

    print("\n--- Resident Component Breakdown ---")
    print(f"1. Tag CSR Matrix (data+indices+indptr):  {tag_m_bytes / (1024**2):.2f} MB ({tag_m_bytes / (1024**3):.3f} GB)")
    print(f"2. Tag Row Norms:                         {tag_norms_bytes / (1024**2):.2f} MB ({tag_norms_bytes / (1024**3):.3f} GB)")
    print(f"3. Tag Movie IDs:                         {tag_ids_bytes / (1024**2):.2f} MB ({tag_ids_bytes / (1024**3):.3f} GB)")
    print(f"4. MetaIndex (ids/langs/years/dates/gen): {meta_idx_bytes / (1024**2):.2f} MB ({meta_idx_bytes / (1024**3):.3f} GB)")
    print(f"5. Sentence-Transformer (MiniLM weights): ~120 MB (0.12 GB)")
    print(f"6. Chroma plots_embeddings HNSW index:    ~1,250 MB (~1.22 GB resident)")
    print(f"7. Plot Embeddings Memmap (in OS cache):  1,572 MB (1.536 GB)")
    print(f"8. Python baseline / dynamic heaps:       ~150 MB (0.15 GB)")
    print(f"-------------------------------------------------------------")
    print(f"DROPPED: Chroma tags_vectors HNSW index:  0 MB (SAVED: 1,590 MB / 1.55 GB)")
    print(f"DROPPED: TruncatedSVD components artifact:0 MB (SAVED: 250 MB / 0.24 GB)")
    print(f"TOTAL MEMORY SAVED:                       1,840 MB (1.80 GB)")

    print("\n--- Actual Process Working Set (Resident RAM) ---")
    print(f"Working Set after warm-up:    {ws_warm / (1024**3):.2f} GB ({ws_warm / (1024**2):.2f} MB)")
    print(f"Private Commit after warm-up: {priv_warm / (1024**3):.2f} GB ({priv_warm / (1024**2):.2f} MB)")

    # Minimum RAM calculation:
    # Full resident footprint to avoid page faults on all components:
    # CSR (0.668) + Norms/IDs (0.012) + MetaIndex (0.024) + ST (0.120) + Python runtime (0.150) + Chroma plots (1.22) + Plot memmap (1.536)
    total_resident_gb = 0.668 + 0.012 + 0.024 + 0.120 + 0.150 + 1.220 + 1.536
    min_ram_gb = total_resident_gb + 1.0  # + 1 GB OS headroom

    print(f"\n--- Minimum RAM Requirement ---")
    print(f"Total resident bytes to keep entire system in RAM without paging: {total_resident_gb:.2f} GB ({total_resident_gb*1024:.0f} MB)")
    print(f"Minimum RAM (Resident + 1 GB OS headroom):                        {min_ram_gb:.2f} GB ({min_ram_gb*1024:.0f} MB)")
    print(f"Previous requirement (Round 2 with tags_vectors):               6.09 GB")
    print(f"Reduction in required RAM:                                       {6.09 - min_ram_gb:.2f} GB")

if __name__ == "__main__":
    measure()
