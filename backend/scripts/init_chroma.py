"""
Step 2: Runs INSIDE the Docker container to re-ingest exported data into
Linux-native ChromaDB. Called from entrypoint.sh on first startup only.

Usage: python -m scripts.init_chroma
"""
import json
import os
import sys

import chromadb
import numpy as np

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/chroma_data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/artifacts/v1")
MARKER = os.path.join(CHROMA_DIR, ".linux_ready")


def ingest_collection(client, name: str, export_path: str):
    print(f"\n[{name}] Loading from {export_path} ...")
    data = np.load(export_path, allow_pickle=False)

    ids = data["ids"].tolist()
    embeddings = data["embeddings"].tolist()

    metadatas = None
    if "metadatas" in data:
        metadatas = [json.loads(m) if m != "{}" else None for m in data["metadatas"]]
        # Filter out all-None metadatas
        if all(m is None for m in metadatas):
            metadatas = None

    total = len(ids)
    print(f"  Records to ingest: {total}, embedding dim: {data['embeddings'].shape[1]}")

    # Delete existing collection if present
    try:
        client.delete_collection(name)
        print(f"  Deleted existing collection '{name}'")
    except Exception:
        pass

    col = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 5000
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        kwargs = {
            "ids": ids[i:end],
            "embeddings": embeddings[i:end],
        }
        if metadatas:
            batch_meta = metadatas[i:end]
            # Only include if there are non-None values
            if any(m is not None for m in batch_meta):
                kwargs["metadatas"] = [m or {} for m in batch_meta]

        col.add(**kwargs)
        print(f"  {end}/{total}")

    print(f"  Done. Collection '{name}' has {col.count()} records.")


def main():
    if os.path.exists(MARKER):
        print("ChromaDB already initialized for Linux. Skipping.")
        return

    print(f"Initializing ChromaDB at {CHROMA_DIR} from exports in {ARTIFACTS_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Find all export files
    export_files = sorted(
        f for f in os.listdir(ARTIFACTS_DIR)
        if f.startswith("_chroma_export_") and f.endswith(".npz")
    )

    if not export_files:
        print(f"No export files found in {ARTIFACTS_DIR}. Skipping ChromaDB init.")
        return

    for fname in export_files:
        # Extract collection name: _chroma_export_<name>.npz
        col_name = fname.replace("_chroma_export_", "").replace(".npz", "")
        ingest_collection(client, col_name, os.path.join(ARTIFACTS_DIR, fname))

    # Write marker so we don't re-ingest on next startup
    with open(MARKER, "w") as f:
        f.write("ok")

    print("\nAll collections ingested successfully!")


if __name__ == "__main__":
    main()
