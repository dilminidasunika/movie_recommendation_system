"""
Step 1: Run this on WINDOWS to export ChromaDB data to portable numpy files.
These are temporary transfer files, NOT used at runtime.

Usage: python scripts/export_chroma.py
Output: backend/artifacts/v1/_chroma_export_plots.npz
        backend/artifacts/v1/_chroma_export_tags.npz
"""
import chromadb
import numpy as np
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "v1")


def export_collection(client, name, out_path):
    col = client.get_collection(name)
    total = col.count()
    print(f"\n[{name}] total records: {total}")

    all_ids = []
    all_embs = []
    all_metas = []
    batch_size = 5000

    for offset in range(0, total, batch_size):
        res = col.get(limit=batch_size, offset=offset, include=["embeddings", "metadatas"])
        all_ids.extend(res["ids"])
        all_embs.extend(res["embeddings"])
        all_metas.extend(res["metadatas"] if res["metadatas"] else [None] * len(res["ids"]))
        print(f"  {min(offset + batch_size, total)}/{total}")

    ids_arr = np.array(all_ids, dtype="U20")  # string IDs
    embs_arr = np.array(all_embs, dtype=np.float32)

    # Save metadatas as structured arrays if they exist
    has_meta = any(m is not None for m in all_metas)

    if has_meta:
        import json
        meta_json = np.array([json.dumps(m) if m else "{}" for m in all_metas], dtype="U500")
        np.savez_compressed(out_path, ids=ids_arr, embeddings=embs_arr, metadatas=meta_json)
    else:
        np.savez_compressed(out_path, ids=ids_arr, embeddings=embs_arr)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Shape: ids={ids_arr.shape}, embeddings={embs_arr.shape}")


def main():
    print(f"ChromaDB path: {os.path.abspath(CHROMA_PATH)}")
    client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_PATH))

    collections = client.list_collections()
    print(f"Found {len(collections)} collections: {[c.name for c in collections]}")

    for col_info in collections:
        out_file = os.path.join(OUTPUT_DIR, f"_chroma_export_{col_info.name}.npz")
        export_collection(client, col_info.name, out_file)

    print("\nDone! Now rebuild Docker image and start containers.")
    print("The init script will re-ingest into Linux-native ChromaDB on first startup.")


if __name__ == "__main__":
    main()
